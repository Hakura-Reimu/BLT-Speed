from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blt.atomic import atomic_json
from blt.bench import (InferenceCase, _check_budget, _gpu_telemetry, _software_environment,
                       cache_statistics, paired_speed_interval, val_prompts)
from blt.config import BPEConfig
from blt.data import verify_supply
from blt.decode import (CUDAGraphStepper, CUDASplitGraphStepper, StaticBPEState, StaticByteState, _delimiter,
                        speculate_bpe, speculate_byte)
from blt.eval_io import EvaluationModel, load_evaluation_model
from blt.eval_tokens import TokenBytes
from blt.score import NUMERICS_POLICY, autocast_for, configure_matmul_precision


POLL_INTERVAL = 8
NON_SPECULATIVE_MODES = ("eager", "device", "device-conditional", "graph", "graph-split")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _static_cache_statistics(state: StaticBPEState | StaticByteState, context_bytes: int) -> dict[str, Any]:
    def physical(tensors):
        return sum(t.numel() * t.element_size() for t in tensors)

    def logical(cache, lengths):
        return sum(2 * sum(lengths) * key.shape[1] * key.shape[3] * key.element_size() for key in cache.keys)

    if isinstance(state, StaticBPEState):
        positions = state.positions.cpu().tolist()
        global_bytes = logical(state.cache, positions)
        local_bytes = 0
        state_bytes = physical([state.positions, state.logits, state.current])
        physical_bytes = physical(state.cache.tensors()) + state_bytes
        routing_bytes = 0
    else:
        positions = state.global_positions.cpu().tolist()
        byte_positions = state.routing.positions.cpu().tolist()
        block = state.model.config.block_size
        encoder_lengths = [((position - 1) % block + 1) if position else 0 for position in byte_positions]
        decoder_lengths = [position % block + 1 for position in byte_positions]
        global_bytes = logical(state.globals, positions)
        local_bytes = logical(state.encoder, encoder_lengths) + logical(state.decoder, decoder_lengths)
        routing_bytes = physical(state.routing.tensors()) + physical([state.global_positions])
        state_bytes = physical([state.conditioning, state.logits, state.current]) + routing_bytes
        physical_bytes = physical(state.encoder.tensors() + state.decoder.tensors() + state.globals.tensors()) + state_bytes
    logical_bytes = global_bytes + local_bytes + state_bytes
    return {"global_kv_bytes": global_bytes, "local_kv_bytes": local_bytes, "state_tensor_bytes": state_bytes,
            "host_hash_history_bytes": 0, "device_routing_bytes": routing_bytes, "logical_bytes": logical_bytes,
            "physical_cache_bytes": physical_bytes, "logical_bytes_per_context_byte": logical_bytes / context_bytes,
            "global_positions": positions}


def _validate_drafter(bundle: EvaluationModel, drafter: EvaluationModel | None) -> None:
    if drafter is None or not isinstance(drafter.config, BPEConfig):
        raise ValueError("BPE speculation requires a BPE drafter")
    tokenizer_hash = bundle.identity.get("tokenizer_sha256")
    if (bundle.device != drafter.device or not tokenizer_hash or
            tokenizer_hash != drafter.identity.get("tokenizer_sha256") or
            bundle.config.vocab_size != drafter.config.vocab_size or bundle.config.bos_id != drafter.config.bos_id):
        raise ValueError("target and drafter must use the same tokenizer and device")
    if sum(p.numel() for p in drafter.model.parameters()) >= sum(p.numel() for p in bundle.model.parameters()):
        raise ValueError("the BPE drafter must be smaller than its target")


def _prepare(case: InferenceCase, mode: str, drafter: EvaluationModel | None, k: int):
    # Up to seven more native units may be generated before the next poll.
    state, logits, _ = case.prefill(resident=True, headroom=case.output_bytes + POLL_INTERVAL)
    is_bpe = isinstance(case.bundle.config, BPEConfig)
    case._output_sizes = (torch.tensor([len(value) if value else 0 for value in case.token_bytes.values],
                                       dtype=torch.long, device=case.bundle.device) if is_bpe else None)
    if mode == "eager":
        case.decode_logits = logits
        if not is_bpe:
            case._initial_patch_counts = [patcher.count for patcher in state.patchers]
        scores = logits[:, -1, :]
        if is_bpe:
            scores = scores.masked_fill(~case.allowed, -torch.inf)
        else:
            scores = scores[:, :256]
        return state, scores.argmax(dim=-1), None, None, None
    static = (StaticBPEState(case.bundle.model, state, logits, case.allowed) if is_bpe
              else StaticByteState(case.bundle.model, state))
    if not is_bpe:
        case._initial_patch_counts = static.routing.patch_counts.clone()
    graph = (CUDASplitGraphStepper(static) if mode == "graph-split" and not is_bpe else
             CUDAGraphStepper(static) if mode in {"graph", "graph-split"} else None)
    draft_static = draft_graph = None
    if mode == "speculative" and is_bpe:
        if drafter is None:
            raise ValueError("BPE speculation requires a smaller BPE drafter")
        draft_case = InferenceCase(drafter, case.prompts, case.batch, case.output_bytes, case.token_bytes)
        draft_state, draft_logits, _ = draft_case.prefill(resident=True, headroom=case.output_bytes + POLL_INTERVAL)
        draft_static = StaticBPEState(drafter.model, draft_state, draft_logits, case.allowed)
        if case.bundle.device.type == "cuda":
            draft_graph = CUDAGraphStepper(draft_static)
    elif mode == "speculative" and case.bundle.device.type == "cuda":
        draft_graph = CUDAGraphStepper(static, byte_draft=True)
    return static, static.current, graph, draft_static, draft_graph


def _decode(case: InferenceCase, mode: str, state: Any, current: torch.Tensor, graph: Any,
            drafter: StaticBPEState | None, draft_graph: CUDAGraphStepper | None, k: int) -> dict[str, Any]:
    is_bpe = isinstance(case.bundle.config, BPEConfig)
    if mode == "speculative":
        lengths, units = [0] * case.batch, [0] * case.batch
        cached_output_bytes = 0
        rounds = []
        while min(lengths) < case.output_bytes:
            # At least one output byte is produced by every allowed BPE token.
            draft_count = min(k, case.output_bytes - min(lengths))
            result = (speculate_bpe(state, drafter, draft_count, draft_graph=draft_graph) if is_bpe
                      else speculate_byte(state, draft_count, draft_graph=draft_graph))
            ids = result.tokens.cpu().tolist()
            accepted_bytes = 0
            for row, values in enumerate(ids):
                cached_output_bytes += sum(len(case.token_bytes.values[value]) if is_bpe else 1 for value in values)
                for index, value in enumerate(values):
                    if lengths[row] >= case.output_bytes:
                        break
                    size = len(case.token_bytes.values[value]) if is_bpe else 1
                    lengths[row] += size
                    units[row] += 1
                    if index < result.accepted_draft_positions:
                        accepted_bytes += size
            rounds.append({"accepted_draft_positions": result.accepted_draft_positions,
                           "accepted_bytes": accepted_bytes,
                           "emitted_positions": result.tokens.shape[1],
                           "draft_seconds": result.draft_seconds, "verify_seconds": result.verify_seconds,
                           "rollback_seconds": result.rollback_seconds})
        return {"output_bytes": sum(lengths), "output_bytes_per_row": lengths,
                "overshoot_bytes": [size - case.output_bytes for size in lengths], "generated_units": units,
                "cached_output_bytes": cached_output_bytes, "speculation_rounds": rounds, "polls": None}

    # Record every unit actually generated, including units after a row first
    # crossed its target and during the final polling interval. No token IDs or
    # counters are transferred to the host between completion polls.
    lengths = torch.zeros(case.batch, dtype=torch.long, device=case.bundle.device)
    units = torch.zeros_like(lengths)
    polls = steps = 0
    while True:
        selected = current if mode == "eager" else state.current
        sizes = case._output_sizes[selected] if is_bpe else torch.ones_like(lengths)
        lengths.add_(sizes)
        units.add_(1)
        steps += 1
        if steps % POLL_INTERVAL == 0:
            polls += 1
            if bool((lengths >= case.output_bytes).all().item()):
                break
        if mode == "eager":
            logits = case.bundle.model.step(state, selected)
            case.decode_logits = logits
            scores = logits[:, -1, :]
            current = (scores.masked_fill(~case.allowed, -torch.inf).argmax(dim=-1) if is_bpe
                       else scores[:, :256].argmax(dim=-1))
        elif graph is not None:
            graph.step()
        elif mode == "device-conditional" and not is_bpe:
            state.conditional_step()
        else:
            state.step()
    # Conversions are deliberately deferred until after the timed sync.
    return {"_lengths": lengths, "_units": units, "_last_sizes": sizes, "_last_selected": selected,
            "polls": polls, "native_steps": steps,
            "routing_polls": steps - 1 if not is_bpe and mode in {"device-conditional", "graph-split"} else 0,
            "speculation_rounds": []}


def _finish_decode(case: InferenceCase, mode: str, state: Any, generated: dict[str, Any]) -> dict[str, Any]:
    """Materialize counters and calculate byte patch closures after timing."""
    if "_lengths" in generated:
        lengths = generated.pop("_lengths").cpu().tolist()
        units = generated.pop("_units").cpu().tolist()
        last_sizes = generated.pop("_last_sizes").cpu().tolist()
        last = generated.pop("_last_selected")
        generated.update(output_bytes=sum(lengths), output_bytes_per_row=lengths,
                         overshoot_bytes=[value - case.output_bytes for value in lengths],
                         generated_units=units, cached_output_bytes=sum(lengths) - sum(last_sizes))
    else:
        last = None  # Speculative decoding has already consumed its last output.
    if not isinstance(case.bundle.config, BPEConfig):
        if mode == "eager":
            counts = [p.count - before for p, before in zip(state.patchers, case._initial_patch_counts)]
            if last is not None:
                for row, value in enumerate(last.cpu().tolist()):
                    counts[row] += int(copy.copy(state.patchers[row]).step(value))
        else:
            counts_tensor = state.routing.patch_counts - case._initial_patch_counts
            if last is not None:
                routing = state.routing
                length = routing.patch_lengths + 1
                closed = ((_delimiter(last) & (length >= routing.min_len)) | (length == routing.max_len) |
                          ((routing.positions + 1).remainder(routing.block_size) == 0))
                counts_tensor = counts_tensor + closed.long()
            counts = counts_tensor.cpu().tolist()
        generated["generated_patch_closures"] = counts
    else:
        generated["generated_patch_closures"] = None
    return generated


TIE_ULPS = 4  # numerical-tie tolerance, in bf16 units in the last place of the reference's top logit
MAX_TIE_FRACTION = 0.01  # at most 1% of a cell's checked decisions may be tolerated ties


def bf16_ulp(value: float) -> float:
    """Spacing of bf16 numbers (8 significant bits) at |value|."""
    magnitude = abs(float(value))
    if not np.isfinite(magnitude):
        return float("inf")
    if magnitude == 0:
        return 2.0 ** -133
    return max(2.0 ** (np.floor(np.log2(magnitude)) - 7), 2.0 ** -133)


def tie_flip(reference: torch.Tensor, tested: torch.Tensor, expected: int, selected: int) -> dict[str, Any] | None:
    """Classify a greedy mismatch on one row (rule fixed 2026-11-12 00:10, after the first run's parity failures were
    examined and before any decode timing was read).

    It is a tolerated numerical tie only if the reference prefers its choice by at most TIE_ULPS bf16 ulps of its
    top logit and the tested path prefers its own choice by at most the same amount. Returns the record of a
    tolerated tie, or None for a real mismatch.
    """
    if tested is None:
        return None
    top = float(reference[expected])
    tolerance = TIE_ULPS * bf16_ulp(top)
    reference_margin = top - float(reference[selected])
    tested_margin = float(tested[selected] - tested[expected])
    if not (np.isfinite(top) and np.isfinite(reference_margin) and np.isfinite(tested_margin)
            and 0 <= reference_margin <= tolerance and 0 <= tested_margin <= tolerance):
        return None
    return {"reference_margin": reference_margin, "tested_margin": tested_margin, "tolerance": tolerance,
            "reference_top_logit": top, "expected": expected, "selected": selected}


@torch.inference_mode()
def verify_decode(bundle: EvaluationModel, prompts: list[bytes], *, batch: int, output_bytes: int,
                  mode: str, k: int = 0, drafter: EvaluationModel | None = None) -> dict[str, Any]:
    """Compare at least output_bytes per row on each distinct-row batch."""
    if (not prompts or min(batch, output_bytes) <= 0 or len(prompts) % batch or
            mode not in {*NON_SPECULATIVE_MODES, "speculative"}):
        raise ValueError("invalid verification settings")
    if len(set(prompts)) != len(prompts):
        raise ValueError("verification prompts must be distinct")
    if mode == "speculative" and k <= 0:
        raise ValueError("speculation verification requires a positive k")
    is_bpe = isinstance(bundle.config, BPEConfig)
    if is_bpe and mode == "speculative":
        _validate_drafter(bundle, drafter)
    checked_bytes = checked_units = 0
    ties: list[dict[str, Any]] = []
    prompt_hashes, prompt_sizes, per_row_bytes, per_row_units, patch_closures, block_resets = [], [], [], [], [], []
    for offset in range(0, len(prompts), batch):
        group = prompts[offset:offset + batch]
        case = InferenceCase(bundle, group, batch, output_bytes)
        with autocast_for(bundle.device):
            reference, logits, _ = case.prefill(resident=True, headroom=output_bytes + POLL_INTERVAL)
            if not is_bpe:
                start_counts = [p.count for p in reference.patchers]
                start_positions = [p.position for p in reference.patchers]
            state, current, graph, draft_state, draft_graph = _prepare(case, mode, drafter, k)
            eager_scores = case.decode_logits[:, -1, :] if mode == "eager" else None
            lengths = [0] * batch
            unit_counts = [0] * batch
            step = 0
            while min(lengths) < output_bytes:
                if mode == "speculative":
                    draft_count = min(k, output_bytes - min(lengths))
                    result = (speculate_bpe(state, draft_state, draft_count, draft_graph=draft_graph) if is_bpe
                              else speculate_byte(state, draft_count, draft_graph=draft_graph))
                    proposals = result.tokens.unbind(dim=1)
                else:
                    proposals = ((current if mode == "eager" else state.current).clone(),)
                for selected in proposals:
                    scores = logits[:, -1, :]
                    expected = (scores.masked_fill(~case.allowed, -torch.inf).argmax(dim=-1) if is_bpe
                                else scores[:, :256].argmax(dim=-1))
                    # Matching argmax values could otherwise hide matching NaNs on both paths.
                    finite = [torch.isfinite(value[:, :256] if not is_bpe else value.masked_fill(~case.allowed, 0)).all()
                              for value in ((scores,) if mode == "speculative" else
                                            (scores, eager_scores if mode == "eager" else state.logits[:, -1, :]))]
                    if not all(bool(flag) for flag in finite):
                        raise AssertionError(f"{mode}: non-finite logits after {step} units for prompt "
                                             f"batch {_batch_digest(group)}")
                    if not torch.equal(selected, expected):
                        if mode == "speculative":
                            raise AssertionError(f"{mode} greedy mismatch after {step} units for prompt "
                                                 f"batch {_batch_digest(group)}")
                        tested = eager_scores if mode == "eager" else state.logits[:, -1, :]
                        masked = [(value.masked_fill(~case.allowed, -torch.inf) if is_bpe else value[:, :256]).float()
                                  for value in (scores, tested)]
                        for row in (selected != expected).nonzero().flatten().tolist():
                            record = tie_flip(masked[0][row], masked[1][row], int(expected[row]), int(selected[row]))
                            if record is None:
                                raise AssertionError(f"{mode} greedy mismatch after {step} units for prompt "
                                                     f"batch {_batch_digest(group)} (row {row}, not a numerical tie)")
                            ties.append({"prompt_batch": _batch_digest(group), "step": step, "row": row, **record})
                        # Teacher-force the tested path to the reference token and keep checking.
                        selected = expected.clone()
                        if mode != "eager":
                            state.current.copy_(expected)
                    ids = selected.tolist() if is_bpe else None
                    for row in range(batch):
                        lengths[row] += len(case.token_bytes.values[ids[row]]) if is_bpe else 1
                        unit_counts[row] += 1
                        checked_units += 1
                    step += 1
                    # Advance the reference even for the final byte, so its
                    # patch and local-block coverage records include that byte.
                    logits = bundle.model.step(reference, selected)
                if mode != "speculative" and min(lengths) < output_bytes:
                    if mode == "eager":
                        actual = bundle.model.step(state, selected)[:, -1, :]
                        eager_scores = actual
                        current = (actual.masked_fill(~case.allowed, -torch.inf).argmax(dim=-1) if is_bpe
                                   else actual[:, :256].argmax(dim=-1))
                    elif graph is not None:
                        graph.step()
                    elif mode == "device-conditional" and not is_bpe:
                        state.conditional_step()
                    else:
                        state.step()
            checked_bytes += sum(lengths)
            prompt_hashes.extend(hashlib.sha256(prompt).hexdigest() for prompt in group)
            prompt_sizes.append([len(prompt) for prompt in group])
            per_row_bytes.append(lengths)
            per_row_units.append(unit_counts)
            if not is_bpe:
                patch_closures.append([p.count - start for p, start in zip(reference.patchers, start_counts)])
                block = bundle.config.block_size
                block_resets.append([p.position // block - start // block
                                     for p, start in zip(reference.patchers, start_positions)])
            del reference, state, graph, draft_state, draft_graph
    if len(ties) > MAX_TIE_FRACTION * checked_units:
        raise AssertionError(f"{mode}: {len(ties)} numerical ties in {checked_units} checked decisions exceed "
                             f"{MAX_TIE_FRACTION:.0%}")
    return {"decoder_mode": mode, "k": k if mode == "speculative" else None,
            "checked_output_bytes": checked_bytes, "checked_units": checked_units,
            "checked_output_bytes_per_row": per_row_bytes, "checked_units_per_row": per_row_units,
            "generated_patch_closures_per_row": patch_closures if not is_bpe else None,
            "local_block_resets_per_row": block_resets if not is_bpe else None,
            "prompt_sha256": prompt_hashes, "prompt_bytes_per_row": prompt_sizes, "greedy_mismatches": 0,
            "raw_argmax_disagreements": len(ties), "tolerated_ties": len(ties), "tie_records": ties,
            "ties_per_row": [sum(1 for t in ties if t["row"] == row) for row in range(batch)],
            "tie_rule": f"a mismatch is tolerated only if both paths prefer their own choice by <= {TIE_ULPS} bf16 "
                        f"ulps of the reference top logit; the tested path is then forced to the reference token; "
                        f"at most {MAX_TIE_FRACTION:.0%} of checked decisions"}


def _batch_digest(prompts: list[bytes]) -> str:
    digests = [hashlib.sha256(prompt).digest() for prompt in prompts]
    return (digests[0] if len(digests) == 1 else hashlib.sha256(b"".join(digests)).digest()).hex()


@torch.inference_mode()
def benchmark_decode(bundles: list[EvaluationModel], prompts: list[bytes], *, batch: int, output_bytes: int,
                     modes: list[str], drafter: EvaluationModel | None = None, warmup: int = 10,
                     repetitions: int = 30, seed: int = 0, selected_modes: list[str] | None = None,
                     memory_budget_bytes: int | None = 10_500_000_000) -> list[dict[str, Any]]:
    needed = (warmup + repetitions) * batch
    if (len(bundles) not in (1, 2) or min(batch, output_bytes, repetitions) <= 0 or warmup < 0 or
            len(prompts) < needed or not modes or (memory_budget_bytes is not None and memory_budget_bytes <= 0) or
            any(mode not in {*NON_SPECULATIVE_MODES, "speculative"} for mode in modes) or
            (selected_modes is not None and (len(selected_modes) != len(bundles) or
                                             any(mode not in NON_SPECULATIVE_MODES or mode not in modes
                                                 for mode in selected_modes)))):
        raise ValueError("invalid decode benchmark settings")
    prompts = prompts[:needed]
    if any(not prompt for prompt in prompts) or len(set(prompts)) != len(prompts):
        raise ValueError("all warm-up and timed row prompts must be distinct")
    groups = [prompts[i:i + batch] for i in range(0, needed, batch)]
    device = bundles[0].device
    if any(bundle.device != device for bundle in bundles) or (drafter is not None and drafter.device != device):
        raise ValueError("all models must use the same device")
    if any(mode in {"graph", "graph-split"} for mode in (selected_modes or modes)) and device.type != "cuda":
        raise ValueError("graph mode requires CUDA")
    for bundle in bundles:
        if isinstance(bundle.config, BPEConfig) and "speculative" in modes:
            _validate_drafter(bundle, drafter)
    tokenizers = [TokenBytes(bundle.tokenizer, bundle.config.vocab_size) if isinstance(bundle.config, BPEConfig) else None
                  for bundle in bundles]
    # Each speculative variant covers one point of both families' requested k grids.
    configurations = [(mode, None, None) for mode in modes if mode != "speculative"]
    if "speculative" in modes:
        configurations.extend(("speculative", byte_k, bpe_k) for byte_k, bpe_k in ((4, 2), (8, 4), (16, 8)))
    if selected_modes is not None:
        configurations = [("selected", None, None)]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _check_budget(device, memory_budget_bytes, "before decode")
    gpu_before = _gpu_telemetry(device)
    variants = {}
    for mode, byte_k, bpe_k in configurations:
        label = mode if mode != "speculative" else f"speculative_byte{byte_k}_bpe{bpe_k}"
        rows = []
        for index, bundle in enumerate(bundles):
            row = {"identity": bundle.identity, "seconds": [], "bytes": [], "peak_allocated_bytes": 0,
                   "peak_reserved_bytes": 0, "baseline_allocated_bytes": [], "baseline_reserved_bytes": [],
                   "compile_new_graphs": [], "cache": [], "realised_bytes_per_unit": [], "overshoot_bytes": [],
                   "output_bytes_per_row": [], "generated_units_per_row": [], "generated_patch_closures_per_row": [],
                   "polls": [], "routing_polls": [], "native_steps": [], "prompt_statistics": [],
                   "prompt_sha256": [], "prompt_sha256_per_row": [], "prompt_bytes": [],
                   "prompt_bytes_per_row": [], "speculation_rounds": [],
                   "decoder_mode": selected_modes[index] if selected_modes is not None else mode}
            if mode == "speculative":
                row["k"] = bpe_k if isinstance(bundle.config, BPEConfig) else byte_k
                if isinstance(bundle.config, BPEConfig):
                    row["drafter_identity"] = drafter.identity
            rows.append(row)
        variants[label] = {"mode": mode, "byte_k": byte_k, "bpe_k": bpe_k, "models": rows}
    tasks = [(label, index) for label in variants for index in range(len(bundles))]
    for trial in range(warmup + repetitions):
        order = tasks if trial % 2 == 0 else list(reversed(tasks))
        for label, index in order:
            variant, bundle = variants[label], bundles[index]
            record = variant["models"][index]
            mode = record["decoder_mode"]
            if "error" in record:
                continue
            k = (variant["bpe_k"] if isinstance(bundle.config, BPEConfig) else variant["byte_k"]) or 0
            state = selected = graph = draft_state = draft_graph = None
            try:
                group = groups[trial]
                case = InferenceCase(bundle, group, batch, output_bytes, tokenizers[index])
                with autocast_for(device):
                    state, selected, graph, draft_state, draft_graph = _prepare(case, mode, drafter, k)
                    _sync(device)
                    _check_budget(device, memory_budget_bytes, "decode preparation")
                    if device.type == "cuda" and trial >= warmup:
                        record["baseline_allocated_bytes"].append(torch.cuda.memory_allocated(device))
                        record["baseline_reserved_bytes"].append(torch.cuda.memory_reserved(device))
                    started = time.perf_counter()
                    generated = _decode(case, mode, state, selected, graph, draft_state, draft_graph, k)
                    _sync(device)
                    elapsed = time.perf_counter() - started
                    _check_budget(device, memory_budget_bytes, "decode timing")
                    generated = _finish_decode(case, mode, state, generated)
                if trial < warmup:
                    continue
                record["seconds"].append(elapsed)
                # The registered speed counts the requested work, not polling or token overshoot.
                record["bytes"].append(batch * output_bytes)
                record["prompt_sha256"].append(_batch_digest(group))
                record["prompt_sha256_per_row"].append([hashlib.sha256(p).hexdigest() for p in group])
                record["prompt_bytes"].append(sum(map(len, group)) / batch)
                record["prompt_bytes_per_row"].append([len(p) for p in group])
                record["overshoot_bytes"].append(generated["overshoot_bytes"])
                record["output_bytes_per_row"].append(generated["output_bytes_per_row"])
                record["generated_units_per_row"].append(generated["generated_units"])
                record["generated_patch_closures_per_row"].append(generated["generated_patch_closures"])
                record["polls"].append(generated["polls"])
                record["routing_polls"].append(generated.get("routing_polls"))
                record["native_steps"].append(generated.get("native_steps"))
                record["speculation_rounds"].append(generated["speculation_rounds"])
                context = sum(map(len, group)) + generated["cached_output_bytes"]
                prompt_stats = {"bytes_per_row": [len(p) for p in group]}
                if isinstance(bundle.config, BPEConfig):
                    prompt_stats["tokens_per_row"] = [length - 1 for length in case.real_lengths]
                else:
                    initial = case._initial_patch_counts
                    prompt_stats["patch_closures_per_row"] = (initial if isinstance(initial, list) else initial.cpu().tolist())
                record["prompt_statistics"].append(prompt_stats)
                if mode == "eager":
                    record["cache"].append(cache_statistics(state, context, case.decode_logits))
                else:
                    record["cache"].append(_static_cache_statistics(state, context))
                if isinstance(bundle.config, BPEConfig):
                    record["realised_bytes_per_unit"].append([n / u for n, u in zip(generated["output_bytes_per_row"],
                                                                                    generated["generated_units"])])
                else:
                    record["realised_bytes_per_unit"].append([n / c if c else None for n, c in
                                                              zip(generated["output_bytes_per_row"], generated["generated_patch_closures"])])
                if device.type == "cuda":
                    record["peak_allocated_bytes"] = max(record["peak_allocated_bytes"], torch.cuda.max_memory_allocated(device))
                    record["peak_reserved_bytes"] = max(record["peak_reserved_bytes"], torch.cuda.max_memory_reserved(device))
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
                record["failed_trial"] = trial
            finally:
                del state, selected, graph, draft_state, draft_graph
    for label, variant in variants.items():
        rows = variant["models"]
        for bundle, row in zip(bundles, rows):
            if "error" in row:
                continue
            times = np.asarray(row["seconds"])
            row["median_seconds"] = float(np.median(times))
            row["p95_seconds"] = float(np.quantile(times, .95))
            row["bytes_per_second"] = float(np.median(np.asarray(row["bytes"]) / times))
            row["generated_bytes_per_token" if isinstance(bundle.config, BPEConfig)
                else "generated_bytes_per_patch_closure"] = row["realised_bytes_per_unit"]
            if isinstance(bundle.config, BPEConfig):
                row["bytes_per_token"] = row["generated_bytes_per_token"]
            if variant["mode"] == "speculative":
                all_rounds = [r for trial in row["speculation_rounds"] for r in trial]
                row["accepted_bytes_per_verification"] = sum(r["accepted_bytes"] for r in all_rounds) / len(all_rounds)
                row["accepted_bytes_per_row_per_verification"] = row["accepted_bytes_per_verification"] / batch
        if len(rows) == 2 and all("error" not in row for row in rows):
            variant["paired_bootstrap"] = paired_speed_interval(rows[0]["seconds"], rows[1]["seconds"],
                                                                  a_bytes=rows[0]["bytes"], b_bytes=rows[1]["bytes"], seed=seed)
        if label != "eager" and "eager" in variants:
            for index, row in enumerate(rows):
                reference = variants["eager"]["models"][index]
                if "error" in row or "error" in reference:
                    continue
                row["paired_vs_eager"] = paired_speed_interval(row["seconds"], reference["seconds"],
                                                                 a_bytes=row["bytes"], b_bytes=reference["bytes"], seed=seed)
        if label != "graph" and "graph" in variants:
            for index, row in enumerate(rows):
                reference = variants["graph"]["models"][index]
                if "error" in row or "error" in reference:
                    continue
                row["paired_vs_graph"] = paired_speed_interval(row["seconds"], reference["seconds"],
                                                                 a_bytes=row["bytes"], b_bytes=reference["bytes"], seed=seed)
                if variant["mode"] == "speculative":
                    ratios = []
                    for rounds, ar_seconds, ar_bytes in zip(row["speculation_rounds"], reference["seconds"], reference["bytes"]):
                        accepted = sum(r["accepted_bytes"] for r in rounds)
                        cost = sum(r["draft_seconds"] + r["verify_seconds"] + r["rollback_seconds"] for r in rounds)
                        ratios.append(accepted * (ar_seconds / ar_bytes) / cost)
                    row["speed_formula_S_per_trial"] = ratios
                    row["speed_formula_S_median"] = float(np.median(ratios))
    if len(bundles) == 2:
        # Preserve paired prompt/timing samples for all cross-family choices,
        # including e.g. byte k=8 versus BPE k=8 or byte graph versus BPE speculation.
        for variant in variants.values():
            a = variant["models"][0]
            variant["cross_decoder_paired_bootstrap"] = {}
            if "error" in a:
                continue
            for other_label, other in variants.items():
                b = other["models"][1]
                if "error" in b:
                    continue
                variant["cross_decoder_paired_bootstrap"][other_label] = paired_speed_interval(
                    a["seconds"], b["seconds"], a_bytes=a["bytes"], b_bytes=b["bytes"], seed=seed)
    gpu_after = _gpu_telemetry(device)
    peak = _check_budget(device, memory_budget_bytes, "decode completion")
    results = []
    for label, variant in variants.items():
        decode = {"models": variant["models"]}
        if "paired_bootstrap" in variant:
            decode["paired_bootstrap"] = variant["paired_bootstrap"]
        if "cross_decoder_paired_bootstrap" in variant:
            decode["cross_decoder_paired_bootstrap"] = variant["cross_decoder_paired_bootstrap"]
        results.append({"mode": "decode", "decoder_mode": variant["mode"], "decoder_variant": label,
                        "batch": batch, "warmup": warmup, "repetitions": repetitions,
                        "warmup_prompt_sha256_per_row": [[hashlib.sha256(p).hexdigest() for p in group]
                                                         for group in groups[:warmup]],
                        "warmup_prompt_bytes_per_row": [[len(p) for p in group] for group in groups[:warmup]],
                        "requested_output_bytes_per_row": output_bytes,
                        "completion_poll_interval_native_steps": POLL_INTERVAL,
                        "timing_definition": "post-prefill through completion poll; throughput uses requested bytes",
                        "memory_budget_bytes": memory_budget_bytes, "peak_reserved_bytes_whole_cell": peak,
                        "environment": _software_environment(device), "gpu_before": gpu_before, "gpu_after": gpu_after,
                        "numerics": {**NUMERICS_POLICY,
                                     "autocast": "bfloat16" if device.type == "cuda" else "disabled (float32)",
                                     "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                                     "tf32_matmul": torch.backends.cuda.matmul.allow_tf32},
                        "generation_policy": "greedy over raw-byte outputs; EOS and special tokens suppressed",
                        "compiled": False, "host_python_object_overhead_in_cache_bytes": False,
                        "variants": {"decode": decode}})
    return results


def select_modes(calibration: list[dict[str, Any]], modes: list[str], model_count: int) -> tuple[list[str], list[dict[str, Any]]]:
    """Choose independently by median elapsed time; all requested modes must succeed."""
    by_mode = {result["decoder_variant"]: result for result in calibration}
    if set(by_mode) != set(modes):
        raise ValueError("calibration did not produce every requested mode")
    choices, details = [], []
    for index in range(model_count):
        samples = {}
        reference_prompts = None
        for mode in modes:
            row = by_mode[mode]["variants"]["decode"]["models"][index]
            if "error" in row or not row.get("seconds"):
                raise ValueError(f"calibration failed for model {index}, {mode}: {row.get('error', 'no trials')}")
            if reference_prompts is None:
                reference_prompts = row["prompt_sha256"]
            elif row["prompt_sha256"] != reference_prompts:
                raise ValueError("calibration modes used different prompts")
            samples[mode] = {"seconds": row["seconds"], "median_seconds": row["median_seconds"],
                             "prompt_sha256": row["prompt_sha256"],
                             "prompt_sha256_per_row": row["prompt_sha256_per_row"],
                             "prompt_bytes_per_row": row["prompt_bytes_per_row"]}
        choice = min(modes, key=lambda mode: samples[mode]["median_seconds"])
        choices.append(choice)
        details.append({"selected_mode": choice, "modes": samples})
    return choices, details


def exact_val_prompts(supply: str, size: int, count: int, seed: int) -> list[bytes]:
    """Filter UTF-8 windows to the requested exact byte length, deterministically."""
    prompts, seen = [], set()
    for attempt in range(8):
        candidates = val_prompts(supply, size, max(count * 2, 32), seed + attempt * 1_000_003)
        for prompt in candidates:
            if len(prompt) == size and prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)
                if len(prompts) == count:
                    return prompts
    raise ValueError("validation data could not supply enough distinct exact-length prompts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate and measure paired, post-prefill greedy decode")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--run", type=Path)
    choice.add_argument("--preset")
    choice.add_argument("--compare", type=Path, nargs=2, metavar=("RUN_A", "RUN_B"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-a", type=Path)
    parser.add_argument("--checkpoint-b", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--tokenizer-b", type=Path)
    parser.add_argument("--min-len", type=int)
    draft = parser.add_mutually_exclusive_group()
    draft.add_argument("--drafter-run", type=Path)
    draft.add_argument("--drafter-preset")
    parser.add_argument("--drafter-checkpoint", type=Path)
    parser.add_argument("--supply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=(*NON_SPECULATIVE_MODES, "speculative", "all"), default="all",
                        help="all calibrates each non-speculative mode; a named mode fixes the choice")
    parser.add_argument("--prompt-kib", type=int, default=2)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--output-bytes", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--calibration-requests", type=int, default=10)
    parser.add_argument("--memory-budget-bytes", type=int, default=10_500_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--verify-bytes", type=int, default=512,
                        help="verify at least this many greedy output bytes in every row (minimum 512)")
    parser.add_argument("--verify-only", action="store_true", help="run the parity gate without timing")
    args = parser.parse_args()
    if (args.warmup < 10 or args.repetitions < 30 or min(args.output_bytes, args.prompt_kib, *args.batches) <= 0 or
            min(args.calibration_requests, args.memory_budget_bytes) <= 0 or args.verify_bytes < 512):
        parser.error("benchmark requires positive sizes, 10 warm-ups, 30 repetitions, and >=512 parity bytes per row")
    if (args.compare and args.checkpoint) or (not args.compare and (args.checkpoint_a or args.checkpoint_b)):
        parser.error("use --checkpoint for a single run or --checkpoint-a/--checkpoint-b with --compare")
    if args.drafter_checkpoint and not args.drafter_run:
        parser.error("--drafter-checkpoint requires --drafter-run")
    # Apply the same policy to model loading, prefill, capture, parity and timing.
    configure_matmul_precision()
    verify_supply(args.supply)
    if args.compare:
        bundles = [load_evaluation_model(run=run, checkpoint=checkpoint, tokenizer_path=tokenizer,
                                         device=args.device, supply=args.supply)
                   for run, checkpoint, tokenizer in zip(args.compare, (args.checkpoint_a, args.checkpoint_b),
                                                          (args.tokenizer, args.tokenizer_b))]
    else:
        bundles = [load_evaluation_model(run=args.run, checkpoint=args.checkpoint, preset_name=args.preset,
                                         min_len=args.min_len, tokenizer_path=args.tokenizer, device=args.device,
                                         supply=args.supply, seed=args.seed)]
    drafter = None
    if args.drafter_run or args.drafter_preset:
        bpe_index = next((i for i, bundle in enumerate(bundles) if isinstance(bundle.config, BPEConfig)), None)
        if bpe_index is None:
            parser.error("a BPE drafter was provided without a BPE target")
        tokenizer = args.tokenizer if bpe_index == 0 else args.tokenizer_b
        drafter = load_evaluation_model(run=args.drafter_run, checkpoint=args.drafter_checkpoint,
                                        preset_name=args.drafter_preset, tokenizer_path=tokenizer,
                                        device=args.device, supply=args.supply, seed=args.seed + 1)
    modes = (list(NON_SPECULATIVE_MODES) if args.mode == "all" else
             ["graph", "speculative"] if args.mode == "speculative" else [args.mode])
    if "speculative" in modes and any(isinstance(bundle.config, BPEConfig) for bundle in bundles) and drafter is None:
        parser.error("BPE speculative mode requires --drafter-run or --drafter-preset")
    total_batches = args.warmup + args.calibration_requests + args.repetitions + 1
    prompts = exact_val_prompts(str(args.supply), args.prompt_kib * 1024, total_batches * max(args.batches),
                                args.seed + args.prompt_kib)
    results = []
    for batch in args.batches:
        try:
            device = bundles[0].device
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _check_budget(device, args.memory_budget_bytes, "before parity")
            warm_end = args.warmup * batch
            calibration_end = warm_end + args.calibration_requests * batch
            timed_end = calibration_end + args.repetitions * batch
            warm_prompts = prompts[:warm_end]
            calibration_prompts = prompts[warm_end:calibration_end]
            timed_prompts = prompts[calibration_end:timed_end]
            verification_prompts = prompts[timed_end:timed_end + batch]
            verification = []
            for bundle in bundles:
                checks = [(mode, 0) for mode in modes if mode in NON_SPECULATIVE_MODES]
                if "speculative" in modes:
                    checks.extend(("speculative", k) for k in ((2, 4, 8) if isinstance(bundle.config, BPEConfig)
                                                                else (4, 8, 16)))
                for mode, k in checks:
                    checked = verify_decode(bundle, verification_prompts, batch=batch, output_bytes=args.verify_bytes,
                                            mode=mode, k=k, drafter=drafter)
                    checked["identity"] = bundle.identity
                    verification.append(checked)
                    _check_budget(device, args.memory_budget_bytes, "parity")
            parity_peak = _check_budget(device, args.memory_budget_bytes, "parity")
            if args.verify_only:
                batch_results = [{"mode": "verification", "batch": batch, "prompt_kib": args.prompt_kib,
                                  "verification": verification}]
            elif args.mode == "speculative":
                # Preserve the existing speculative diagnostic; it is excluded
                # from the registered non-speculative mode selection.
                batch_results = benchmark_decode(bundles, warm_prompts + timed_prompts, batch=batch,
                                                 output_bytes=args.output_bytes, modes=modes, drafter=drafter,
                                                 warmup=args.warmup, repetitions=args.repetitions, seed=args.seed,
                                                 memory_budget_bytes=args.memory_budget_bytes)
            else:
                calibration = benchmark_decode(bundles, warm_prompts + calibration_prompts, batch=batch,
                                               output_bytes=args.output_bytes, modes=modes, warmup=args.warmup,
                                               repetitions=args.calibration_requests, seed=args.seed,
                                               memory_budget_bytes=args.memory_budget_bytes)
                choices, calibration_details = select_modes(calibration, modes, len(bundles))
                calibration_peak = max(cell["peak_reserved_bytes_whole_cell"] for cell in calibration)
                batch_results = benchmark_decode(bundles, warm_prompts + timed_prompts, batch=batch,
                                                 output_bytes=args.output_bytes, modes=modes, selected_modes=choices,
                                                 warmup=args.warmup, repetitions=args.repetitions, seed=args.seed,
                                                 memory_budget_bytes=args.memory_budget_bytes)
                for result in batch_results:
                    result["calibration"] = {"requests": args.calibration_requests, "requested_modes": modes,
                                             "models": calibration_details,
                                             "peak_reserved_bytes": calibration_peak}
                    result["peak_reserved_bytes_whole_cell"] = max(result["peak_reserved_bytes_whole_cell"],
                                                                    parity_peak, calibration_peak)
            for result in batch_results:
                result["prompt_kib"] = args.prompt_kib
                result["verification"] = verification
                result["verify_bytes_per_row"] = args.verify_bytes
                result["seed"] = args.seed
        except Exception as error:
            batch_results = [{"mode": "decode", "prompt_kib": args.prompt_kib, "batch": batch,
                              "error": f"{type(error).__name__}: {error}"}]
        results.extend(batch_results)
        atomic_json(args.output, {"results": results})
    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    if any("error" in result or any("error" in model for model in
                                    result.get("variants", {}).get("decode", {}).get("models", []))
           for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
