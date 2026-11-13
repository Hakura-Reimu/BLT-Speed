from __future__ import annotations

"""Synchronized Track A timing. Bolmo timings use the unmodified HF generation loop."""

from contextlib import contextmanager
import hashlib
import time
from typing import Any

import numpy as np
import torch

from .bench import paired_speed_interval
from .eval_tokens import TokenBytes
from .trackA import ReferenceModel, bolmo_ids, free_bolmo_cache, olmo_ids


def forced_boundary(byte_number: int, p: float) -> bool:
    """One-based byte positions; 4.4 uses the repeating gaps 4, 4, 5, 4, 5."""
    if byte_number < 1:
        return False
    if p == 4.4:
        return byte_number % 22 in (0, 4, 8, 13, 17)
    if p not in (6, 8, 10, 12, 16):
        raise ValueError("unsupported forced patch length")
    return byte_number % int(p) == 0


def forced_mask(input_ids: torch.Tensor, p: float) -> torch.Tensor:
    if input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError("expected BOS followed by bytes")
    length = input_ids.shape[1] - 1
    pattern = [True, *(forced_boundary(i, p) for i in range(1, length + 1))]
    pattern[-1] = False  # stock generate rolls this byte back
    return torch.tensor(pattern, dtype=torch.bool, device=input_ids.device)[None, :].expand(input_ids.shape[0], -1).clone()


def _tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensors(item)


def _cache_tensors(cache: Any) -> list[torch.Tensor]:
    if cache is None:
        return []
    if hasattr(cache, "layers"):
        return [t for layer in cache.layers for t in (getattr(layer, "keys", None), getattr(layer, "values", None))
                if isinstance(t, torch.Tensor)]
    return list(_tensors([getattr(cache, "key_cache", []), getattr(cache, "value_cache", [])]))


def cache_statistics(cache: Any, model: Any, family: str, context_bytes: int) -> dict[str, Any]:
    global_tensors = _cache_tensors(cache)
    local = []
    if family == "bolmo":
        for module in (model.model.local_encoder, model.model.local_decoder):
            local.extend(_tensors([getattr(module, "layer_states", None), getattr(module, "last_h", None),
                                   getattr(module, "last_value", None)]))
    def logical(tensors):
        return sum(t.numel() * t.element_size() for t in tensors)
    storages = {}
    for tensor in global_tensors + local:
        storage = tensor.untyped_storage()
        storages[(tensor.device.type, tensor.device.index, storage.data_ptr())] = storage.nbytes()
    global_bytes, state_bytes = logical(global_tensors), logical(local)
    total = global_bytes + state_bytes
    positions = int(cache.get_seq_length()) if cache is not None else 0
    config = model.config
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    per_position = 2 * int(config.num_hidden_layers) * int(config.num_key_value_heads) * int(head_dim) * 2
    return {"global_kv_bytes": global_bytes, "local_kv_bytes": 0, "state_tensor_bytes": state_bytes,
            "host_hash_history_bytes": 0, "logical_bytes": total, "physical_cache_bytes": sum(storages.values()),
            "logical_bytes_per_context_byte": total / context_bytes, "global_positions": positions,
            "theoretical_bf16_global_kv_bytes_per_position_per_row": per_position,
            "host_python_object_overhead_in_cache_bytes": False}


@contextmanager
def module_events(model: Any):
    """Optional inclusive CUDA event timings; routing is nested inside encoder/decoder."""
    records: dict[str, list[tuple[Any, Any]]] = {key: [] for key in ("local_encoder", "global_trunk", "local_decoder", "routing_pooling", "depool_inclusive")}
    originals = []
    targets = [(model.model.local_encoder, "forward", "local_encoder"),
               (model.model.local_decoder, "forward", "local_decoder"),
               (model.model.local_encoder, "_pool", "routing_pooling"),
               (model.model.local_decoder, "_depool", "depool_inclusive"),
               (model.model.local_encoder.boundary_predictor_module, "forward", "routing_pooling")]
    targets.extend((layer, "forward", "global_trunk") for layer in model.model.layers)
    for obj, name, group in targets:
        original = getattr(obj, name)
        originals.append((obj, name, original))
        def wrapper(*args, _original=original, _group=group, **kwargs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                return _original(*args, **kwargs)
            finally:
                end.record()
                records[_group].append((start, end))
        setattr(obj, name, wrapper)
    try:
        yield records
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)


def event_milliseconds(records: dict[str, list[tuple[Any, Any]]]) -> dict[str, float]:
    torch.cuda.synchronize()
    return {name: sum(start.elapsed_time(end) for start, end in pairs) for name, pairs in records.items()}


@torch.inference_mode()
def run_bolmo(bundle: ReferenceModel, prompt: bytes, batch: int, output_bytes: int, *, p: float | None = None,
              profile: bool = False, resident: bool = True) -> dict[str, Any]:
    from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
    model = bundle.model
    ids = bolmo_ids(prompt, bundle.device, batch) if resident else None
    if len(prompt) < 2 or output_bytes < 1:
        raise ValueError("Bolmo benchmark needs at least two prompt bytes and one output byte")

    class BytesOnly(LogitsProcessor):
        def __init__(self):
            self.calls = 0
        def __call__(self, _input, scores):
            if scores.shape[0] != batch:
                raise RuntimeError("batch boundaries diverged; this forced-layout benchmark requires synchronized rows")
            pos = len(prompt) + self.calls
            self.calls += 1
            allowed = torch.zeros_like(scores, dtype=torch.bool)
            if p is None or not forced_boundary(pos, p):
                allowed[:, 4:260] = True
            if p is None or forced_boundary(pos, p):
                allowed[:, 264:520] = True
            return scores.masked_fill(~allowed, -torch.inf)

    class Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return torch.full((input_ids.shape[0],), input_ids.shape[1] >= output_bytes + 2,
                              dtype=torch.bool, device=input_ids.device)

    processor = BytesOnly()
    original_forward = model.forward
    saved: dict[str, Any] = {"calls": 0}
    profile_records = None
    def prediction(_module, _args, output):
        logprobs, mask = output
        if logprobs.shape[0] != batch or logprobs.shape[1] > ids.shape[1]:
            raise RuntimeError("cannot align forced layout with Bolmo predictor output")
        return logprobs, forced_mask(ids[:, :logprobs.shape[1]], p)
    def forward(*args, **kwargs):
        out = original_forward(*args, **kwargs)
        saved["calls"] += 1
        saved["cache_object"] = out.past_key_values
        if saved["calls"] == 2:
            torch.cuda.synchronize(bundle.device)
            saved["prefill_end"] = time.perf_counter()
            saved["prefill_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(bundle.device)
            saved["prefill_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(bundle.device)
            saved["cache"] = cache_statistics(out.past_key_values, model, "bolmo", len(prompt) * batch)
            if profile_records is not None:
                saved["event_split"] = {name: len(pairs) for name, pairs in profile_records.items()}
            saved["decode_start"] = time.perf_counter()
        return out
    model.forward = forward
    handle = model.model.local_encoder.boundary_predictor_module.register_forward_hook(prediction) if p is not None else None
    try:
        torch.cuda.synchronize(bundle.device)
        baseline_allocated = torch.cuda.memory_allocated(bundle.device)
        baseline_reserved = torch.cuda.memory_reserved(bundle.device)
        torch.cuda.reset_peak_memory_stats(bundle.device)
        start = time.perf_counter()
        if ids is None:
            ids = bolmo_ids(prompt, bundle.device, batch)
        if profile:
            with module_events(model) as events:
                profile_records = events
                output = model.generate(ids, do_sample=False, max_new_tokens=output_bytes + 4,
                                        logits_processor=LogitsProcessorList([processor]),
                                        stopping_criteria=StoppingCriteriaList([Stop()]))
        else:
            output = model.generate(ids, do_sample=False, max_new_tokens=output_bytes + 4,
                                    logits_processor=LogitsProcessorList([processor]),
                                    stopping_criteria=StoppingCriteriaList([Stop()]))
        torch.cuda.synchronize(bundle.device)
        end = time.perf_counter()
        if saved["calls"] < 2 or output.shape != (batch, ids.shape[1] + output_bytes):
            raise RuntimeError("stock Bolmo generate stopped at an unexpected output length")
        if not torch.equal(output[:, :ids.shape[1]], ids):
            raise RuntimeError("stock Bolmo generate changed the prompt")
        if batch > 1 and not torch.equal(output, output[:1].expand_as(output)):
            raise RuntimeError("benchmark requires identical synchronized batch rows")
        if ((output[:, ids.shape[1]:] < 4) | (output[:, ids.shape[1]:] >= 260)).any():
            raise RuntimeError("stock Bolmo generate produced a non-byte token")
        result = {"prefill_seconds": saved["prefill_end"] - start, "decode_seconds": end - saved["decode_start"],
                  "bytes": [len(prompt) * batch, output_bytes * batch], "cache": saved["cache"],
                  "cache_after_decode": cache_statistics(saved["cache_object"], model, "bolmo",
                                                          (len(prompt) + output_bytes - 1) * batch),
                  "bytes_per_patch": len(prompt) / saved["cache"]["global_positions"],
                  "prefill_peak_allocated_bytes": saved["prefill_peak_allocated_bytes"],
                  "prefill_peak_reserved_bytes": saved["prefill_peak_reserved_bytes"],
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(bundle.device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(bundle.device),
                  "baseline_allocated_bytes": baseline_allocated, "baseline_reserved_bytes": baseline_reserved,
                  "overshoot_bytes": [0] * batch, "output_bytes_per_row": [output_bytes] * batch}
        if profile:
            result["inclusive_cuda_event_ms"] = {
                "prefill": event_milliseconds({name: pairs[:saved["event_split"][name]] for name, pairs in events.items()}),
                "decode": event_milliseconds({name: pairs[saved["event_split"][name]:] for name, pairs in events.items()})}
        return result
    finally:
        model.forward = original_forward
        if handle is not None:
            handle.remove()
        free_bolmo_cache(model)


@torch.inference_mode()
def run_olmo(bundle: ReferenceModel, prompt: bytes, batch: int, output_bytes: int, vocabulary: TokenBytes,
             *, resident: bool = True) -> dict[str, Any]:
    token_ids = olmo_ids(bundle.tokenizer, prompt)
    represented = [vocabulary.values[token] for token in token_ids]
    if any(value is None for value in represented) or b"".join(represented) != prompt:
        raise ValueError("OLMo token-byte mapping does not reproduce the prompt")
    if len(token_ids) + 1 > bundle.model.config.max_position_embeddings:
        raise ValueError("OLMo prompt exceeds its position limit")
    inputs = (torch.tensor([bundle.bos_id, *token_ids], device=bundle.device, dtype=torch.long)[None, :].repeat(batch, 1)
              if resident else None)
    _, allowed = vocabulary.tensors(bundle.device)
    torch.cuda.synchronize(bundle.device)
    baseline_allocated, baseline_reserved = torch.cuda.memory_allocated(bundle.device), torch.cuda.memory_reserved(bundle.device)
    torch.cuda.reset_peak_memory_stats(bundle.device)
    start = time.perf_counter()
    if inputs is None:
        token_ids = bundle.tokenizer.encode(prompt.decode("utf-8"), add_special_tokens=False)
        inputs = torch.tensor([bundle.bos_id, *token_ids], device=bundle.device, dtype=torch.long)[None, :].repeat(batch, 1)
    out = bundle.model(input_ids=inputs, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize(bundle.device)
    prefill_end = time.perf_counter()
    prefill_peak_allocated, prefill_peak_reserved = torch.cuda.max_memory_allocated(bundle.device), torch.cuda.max_memory_reserved(bundle.device)
    cache = cache_statistics(out.past_key_values, bundle.model, "olmo", len(prompt) * batch)
    decode_start = time.perf_counter()
    lengths, units = [0] * batch, [0] * batch
    cached_output_bytes = 0
    while min(lengths) < output_bytes:
        selected = out.logits[:, -1].masked_fill(~allowed, -torch.inf).argmax(-1)
        for row, token in enumerate(selected.tolist()):
            if lengths[row] < output_bytes:
                value = vocabulary.values[token]
                if not value:
                    raise RuntimeError("OLMo produced a token without raw bytes")
                lengths[row] += len(value)
                units[row] += 1
        if min(lengths) < output_bytes:
            if out.past_key_values.get_seq_length() >= bundle.model.config.max_position_embeddings:
                raise ValueError("OLMo decode exceeds its position limit")
            cached_output_bytes += sum(len(vocabulary.values[token]) for token in selected.tolist())
            out = bundle.model(input_ids=selected[:, None], past_key_values=out.past_key_values, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize(bundle.device)
    end = time.perf_counter()
    return {"prefill_seconds": prefill_end - start, "decode_seconds": end - decode_start,
            "bytes": [len(prompt) * batch, sum(lengths)], "cache": cache,
            "cache_after_decode": cache_statistics(out.past_key_values, bundle.model, "olmo",
                                                    len(prompt) * batch + cached_output_bytes),
            "prefill_peak_allocated_bytes": prefill_peak_allocated, "prefill_peak_reserved_bytes": prefill_peak_reserved,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(bundle.device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(bundle.device),
            "baseline_allocated_bytes": baseline_allocated, "baseline_reserved_bytes": baseline_reserved,
            "bytes_per_token": len(prompt) / len(token_ids), "generated_units": units,
            "output_bytes_per_row": lengths, "overshoot_bytes": [length - output_bytes for length in lengths]}


def benchmark_pair(bolmo: ReferenceModel, olmo: ReferenceModel, prompts: list[bytes], *, batch: int,
                   output_bytes: int = 256, warmup: int = 10, repetitions: int = 30,
                   forced_p: float | None = None, seed: int = 0, resident: bool = True) -> dict[str, Any]:
    if warmup < 0 or repetitions < 1 or len(prompts) < warmup + repetitions:
        raise ValueError("insufficient paired prompts or repetitions")
    vocab = TokenBytes(olmo.tokenizer.backend_tokenizer, olmo.model.config.vocab_size)
    for token in olmo.tokenizer.all_special_ids:
        if 0 <= token < len(vocab.values):
            vocab.values[token] = None
    records = [{"identity": model.identity, "trials": []} for model in (bolmo, olmo)]
    for trial, prompt in enumerate(prompts[:warmup + repetitions]):
        order = (0, 1) if trial % 2 == 0 else (1, 0)
        for index in order:
            row = (run_bolmo(bolmo, prompt, batch, output_bytes, p=forced_p, resident=resident) if index == 0 else
                   run_olmo(olmo, prompt, batch, output_bytes, vocab, resident=resident))
            if trial >= warmup:
                row["prompt_sha256"] = hashlib.sha256(prompt).hexdigest()
                row["prompt_bytes"] = len(prompt)
                records[index]["trials"].append(row)
    result: dict[str, Any] = {"batch": batch, "warmup": warmup, "repetitions": repetitions,
                              "requested_output_bytes_per_row": output_bytes, "forced_p": forced_p,
                              "models": records, "variants": {}}
    for phase, number in (("prefill", 0), ("decode", 1)):
        rows = []
        for model in records:
            times = [r[f"{phase}_seconds"] for r in model["trials"]]
            sizes = [r["bytes"][number] for r in model["trials"]]
            rows.append({"identity": model["identity"], "seconds": times, "bytes": sizes,
                         "median_seconds": float(np.median(times)), "p95_seconds": float(np.quantile(times, 0.95)),
                         "bytes_per_second": float(np.median(np.asarray(sizes) / np.asarray(times))),
                         "cache": [r["cache"] if phase == "prefill" else r["cache_after_decode"] for r in model["trials"]],
                         "peak_allocated_bytes": max(r["prefill_peak_allocated_bytes" if phase == "prefill" else "peak_allocated_bytes"] for r in model["trials"]),
                         "peak_reserved_bytes": max(r["prefill_peak_reserved_bytes" if phase == "prefill" else "peak_reserved_bytes"] for r in model["trials"]),
                         "baseline_allocated_bytes": [r["baseline_allocated_bytes"] for r in model["trials"]],
                         "baseline_reserved_bytes": [r["baseline_reserved_bytes"] for r in model["trials"]],
                         "prompt_sha256": [r["prompt_sha256"] for r in model["trials"]],
                         "prompt_bytes": [r["prompt_bytes"] for r in model["trials"]],
                         "overshoot_bytes": [r["overshoot_bytes"] for r in model["trials"]],
                         "realised_bytes_per_unit": [r.get("bytes_per_patch", r.get("bytes_per_token")) for r in model["trials"]]})
        result["variants"][phase] = {"models": rows, "paired_bootstrap": paired_speed_interval(
            rows[0]["seconds"], rows[1]["seconds"], a_bytes=rows[0]["bytes"], b_bytes=rows[1]["bytes"], seed=seed)}
    return result


def benchmark_bolmo(bolmo: ReferenceModel, prompts: list[bytes], *, batch: int, output_bytes: int,
                    warmup: int, repetitions: int, forced_p: float | None, resident: bool = True) -> dict[str, Any]:
    """Keep Bolmo measurements available if an OLMo prompt exceeds its context limit."""
    if warmup < 0 or repetitions < 1 or len(prompts) < warmup + repetitions:
        raise ValueError("insufficient prompts or repetitions")
    rows = []
    for trial, prompt in enumerate(prompts[:warmup + repetitions]):
        row = run_bolmo(bolmo, prompt, batch, output_bytes, p=forced_p, resident=resident)
        if trial >= warmup:
            row.update({"prompt_sha256": hashlib.sha256(prompt).hexdigest(), "prompt_bytes": len(prompt)})
            rows.append(row)
    variants = {}
    for phase, index in (("prefill", 0), ("decode", 1)):
        times = [row[f"{phase}_seconds"] for row in rows]
        sizes = [row["bytes"][index] for row in rows]
        variants[phase] = {"models": [{"identity": bolmo.identity, "seconds": times, "bytes": sizes,
                                       "median_seconds": float(np.median(times)),
                                       "p95_seconds": float(np.quantile(times, 0.95)),
                                       "bytes_per_second": float(np.median(np.asarray(sizes) / np.asarray(times))),
                                       "cache": [r["cache"] if phase == "prefill" else r["cache_after_decode"] for r in rows],
                                       "peak_allocated_bytes": max(r["prefill_peak_allocated_bytes" if phase == "prefill" else "peak_allocated_bytes"] for r in rows),
                                       "peak_reserved_bytes": max(r["prefill_peak_reserved_bytes" if phase == "prefill" else "peak_reserved_bytes"] for r in rows),
                                       "baseline_allocated_bytes": [r["baseline_allocated_bytes"] for r in rows],
                                       "baseline_reserved_bytes": [r["baseline_reserved_bytes"] for r in rows],
                                       "prompt_sha256": [r["prompt_sha256"] for r in rows],
                                       "prompt_bytes": [r["prompt_bytes"] for r in rows],
                                       "overshoot_bytes": [r["overshoot_bytes"] for r in rows],
                                       "realised_bytes_per_unit": [r["bytes_per_patch"] for r in rows]}]}
    return {"batch": batch, "warmup": warmup, "repetitions": repetitions, "forced_p": forced_p,
            "requested_output_bytes_per_row": output_bytes, "models": [{"identity": bolmo.identity, "trials": rows}],
            "variants": variants}
