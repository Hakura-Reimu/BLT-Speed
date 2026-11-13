from __future__ import annotations

from collections import OrderedDict
import gc
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Iterable

import numpy as np
import torch

from .config import BPEConfig, ByteConfig
from .data import iter_eval_chunks, utf8_end
from .eval_io import EvaluationModel
from .eval_tokens import TokenBytes, first_byte_probabilities
from .hashes import hashes_at_ends
from .model import BPELM, BPEState, BatchedKVCache, ByteLatentLM, ByteState
from .patcher import max_patch_count, patch_batch, patch_capacity as choose_patch_capacity, trim_patch_metadata
from .score import autocast_for


def val_prompts(supply: str, size: int, count: int, seed: int = 0) -> list[bytes]:
    if min(size, count) <= 0:
        raise ValueError("prompt size and count must be positive")
    documents: list[bytearray] = []
    previous = None
    for chunk in iter_eval_chunks(supply, "val"):
        if chunk.document != previous:
            documents.append(bytearray())
            previous = chunk.document
        documents[-1].extend(chunk.data)
    stream = b"\n".join(documents)
    if len(stream) < size + 4:
        raise ValueError("validation text is too small for the requested prompt")
    rng = np.random.default_rng(seed)
    prompts: list[bytes] = []
    seen: set[bytes] = set()
    attempts = 0
    while len(prompts) < count:
        attempts += 1
        if attempts > max(1000, count * 100):
            raise ValueError("validation text cannot supply enough distinct prompts")
        start = int(rng.integers(0, len(stream) - size - 3))
        while start < len(stream) and stream[start] & 0xC0 == 0x80:
            start += 1
        end = utf8_end(stream, start, size)
        prompt = stream[start:end]
        prompt.decode("utf-8")
        if prompt not in seen:
            prompts.append(prompt)
            seen.add(prompt)
    return prompts


def _tensor_bytes(tensors: Iterable[torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def cache_statistics(state: BPEState | ByteState, context_bytes: int, logits: torch.Tensor | None = None) -> dict[str, Any]:
    if isinstance(state, BPEState):
        cache = state.cache
        physical = _tensor_bytes([*cache.keys, *cache.values])
        lengths = (list(cache.lengths) if isinstance(cache, BatchedKVCache) else
                   [cache.length] * cache.keys[0].shape[0])
        global_bytes = sum(2 * sum(lengths) * key.shape[1] * key.shape[3] * key.element_size() for key in cache.keys)
        state_bytes = _tensor_bytes([logits]) if logits is not None else 0
        logical = global_bytes + state_bytes
        return {"global_kv_bytes": global_bytes, "local_kv_bytes": 0, "state_tensor_bytes": state_bytes,
                "host_hash_history_bytes": 0, "logical_bytes": logical, "physical_cache_bytes": physical + state_bytes,
                "logical_bytes_per_context_byte": logical / context_bytes, "global_positions": lengths}
    def sizes(cache):
        physical = _tensor_bytes([*cache.keys, *cache.values])
        logical = sum(2 * sum(cache.lengths) * key.shape[1] * key.shape[3] * key.element_size() for key in cache.keys)
        return logical, physical
    global_bytes, global_physical = sizes(state.globals)
    enc_bytes, enc_physical = sizes(state.encoder)
    dec_bytes, dec_physical = sizes(state.decoder)
    state_bytes = _tensor_bytes([state.conditioning, state.logits])
    history_bytes = sum(len(history.history) for history in state.hashes)
    logical = global_bytes + enc_bytes + dec_bytes + state_bytes + history_bytes
    return {"global_kv_bytes": global_bytes, "local_kv_bytes": enc_bytes + dec_bytes,
            "state_tensor_bytes": state_bytes, "host_hash_history_bytes": history_bytes, "logical_bytes": logical,
            "physical_cache_bytes": global_physical + enc_physical + dec_physical + state_bytes,
            "logical_bytes_per_context_byte": logical / context_bytes, "global_positions": list(state.globals.lengths)}


# Exact, faster encoders for timed BPE preprocessing, keyed by tokenizer SHA-256: (name, text -> token-ID array).
FAST_ENCODERS: dict[str, tuple[str, Callable[[str], Any]]] = {}


def register_fast_encoder(tokenizer_sha256: str, name: str, encode: Callable[[str], Any]) -> None:
    FAST_ENCODERS[tokenizer_sha256] = (name, encode)


class InferenceCase:
    def __init__(self, bundle: EvaluationModel, prompt: bytes | list[bytes], batch: int, output_bytes: int = 256,
                 token_bytes: TokenBytes | None = None, prepare_resident: bool = True,
                 prepare_preprocessed: bool = False, patch_capacity: str = "bucket64") -> None:
        self._repeated_prompt = isinstance(prompt, bytes)
        prompts = [prompt] * batch if isinstance(prompt, bytes) else list(prompt)
        if len(prompts) != batch or not all(prompts) or min(batch, output_bytes) <= 0:
            raise ValueError("prompt, batch and output size must be positive")
        if patch_capacity not in {"exact", "bucket64", "upper_bound"}:
            raise ValueError("patch_capacity must be exact, bucket64 or upper_bound")
        self.bundle, self.prompts, self.prompt = bundle, prompts, prompts[0]
        self.batch, self.output_bytes = batch, output_bytes
        self.patch_capacity = patch_capacity
        self.device_bytes = self.patch_metadata = self.tokens = None
        self.resident_payload = None
        self.actual_capacity = self.padded_capacity = None
        self.real_lengths: list[int] = []
        self.encode_seconds = 0.0
        self.encoder_name = None
        self.prompt_token_count = None
        self.total_prompt_tokens = None
        self.token_bytes = None
        self.first_bytes = self.allowed = None
        if isinstance(bundle.config, BPEConfig):
            self.token_bytes = token_bytes or TokenBytes(bundle.tokenizer, bundle.config.vocab_size)
            self.first_bytes, self.allowed = self.token_bytes.tensors(bundle.device)
        if prepare_resident:
            self.resident_payload = self.to_device(self.prepare_host(preprocessed=prepare_preprocessed))
            self.tokens = self.resident_payload.get("tokens")
            self.device_bytes = self.resident_payload.get("bytes")
            self.patch_metadata = self.resident_payload.get("patches")

    def prepare_host(self, *, preprocessed: bool = True) -> dict[str, Any]:
        """Tokenize or patch on the CPU; leave all tensors on the host."""
        c = self.bundle.config
        if isinstance(c, BPEConfig):
            fast = FAST_ENCODERS.get(getattr(self.bundle, "identity", {}).get("tokenizer_sha256"))
            started = time.perf_counter()
            if fast is None:
                encoded = [self.bundle.tokenizer.encode(prompt.decode("utf-8"), add_special_tokens=False).ids
                           for prompt in self.prompts]
            else:
                encoded = [fast[1](prompt.decode("utf-8")) for prompt in self.prompts]
            # Tokenization alone (UTF-8 decoding included); host tensor construction follows.
            self.encode_seconds = time.perf_counter() - started
            self.encoder_name = "hf-tokenizers" if fast is None else fast[0]
            if any(len(ids) == 0 for ids in encoded):
                raise ValueError("tokenizer produced no tokens")
            self.real_lengths = [len(ids) + 1 for ids in encoded]
            self.prompt_token_count = self.real_lengths[0] - 1
            self.total_prompt_tokens = sum(length - 1 for length in self.real_lengths)
            actual = max(self.real_lengths)
            upper = min(c.trunk.max_positions, max(actual, 1 + max(map(len, self.prompts))))
            padded = (actual if self._repeated_prompt else
                      choose_patch_capacity(actual, upper if self.patch_capacity == "upper_bound" else
                                            max(upper, ((actual + 63) // 64) * 64), self.patch_capacity))
            if padded > c.trunk.max_positions:
                raise ValueError("padded token capacity exceeds configured positions")
            # Position 0 of every row is BOS; the token IDs follow, then BOS padding.
            values = np.full((self.batch, padded), c.bos_id, dtype=np.int64)
            for row, ids in enumerate(encoded):
                values[row, 1:len(ids) + 1] = ids
            self.actual_capacity, self.padded_capacity = actual, padded
            return {"tokens": torch.from_numpy(values), "lengths": torch.tensor(self.real_lengths, dtype=torch.long)}

        self.real_lengths = [len(prompt) for prompt in self.prompts]
        width = (max(self.real_lengths) if self._repeated_prompt else
                 ((max(self.real_lengths) + 63) // 64) * 64)
        host = torch.zeros((self.batch, width), dtype=torch.uint8)
        for row, prompt in enumerate(self.prompts):
            host[row, :len(prompt)] = torch.from_numpy(np.frombuffer(prompt, dtype=np.uint8).copy())
        payload: dict[str, Any] = {"bytes": host, "lengths": torch.tensor(self.real_lengths, dtype=torch.long)}
        self.actual_capacity = self.padded_capacity = None
        self.encode_seconds = 0.0
        if preprocessed:
            started = time.perf_counter()
            patched = patch_batch(host.numpy(), c.min_len, c.max_len, c.block_size,
                                  lengths=np.asarray(self.real_lengths, dtype=np.int64))
            actual = int(patched.counts.max())
            upper = max_patch_count(width, c.min_len, c.block_size)
            capacity = choose_patch_capacity(actual, upper, self.patch_capacity)
            ends = torch.full((self.batch, capacity), -1, dtype=torch.long)
            if actual:
                ends[:, :actual] = torch.from_numpy(patched.ends)
            metadata = {"ends": ends, "patch_ids": torch.from_numpy(patched.patch_ids),
                        "counts": torch.from_numpy(patched.counts)}
            if c.hash_enabled:
                metadata["hash_ids"] = hashes_at_ends(host.long(), ends, c.hash_rows, c.hash_seed)
            # Patching and hashes (with their small metadata tensors), the byte counterpart of tokenization.
            self.encode_seconds = time.perf_counter() - started
            payload["patches"] = metadata
            self.actual_capacity, self.padded_capacity = actual, capacity
        return payload

    def to_device(self, host: dict[str, Any]) -> dict[str, Any]:
        return {key: ({k: v.to(self.bundle.device) for k, v in value.items()} if isinstance(value, dict)
                      else value.to(self.bundle.device)) for key, value in host.items()}

    def host_inputs(self) -> dict[str, Any]:
        return self.to_device(self.prepare_host())

    def resident_inputs(self, *, preprocessed: bool = False) -> dict[str, Any]:
        if self.resident_payload is None or (preprocessed and isinstance(self.bundle.config, ByteConfig) and
                                             "patches" not in self.resident_payload):
            raise ValueError("resident inputs were not prepared")
        return self.resident_payload

    def distribution(self, logits: torch.Tensor) -> torch.Tensor:
        if isinstance(self.bundle.config, BPEConfig):
            return first_byte_probabilities(logits, self.first_bytes)
        return logits.float().softmax(dim=-1)

    def prefill(self, *, resident: bool, headroom: int = 0, preprocessed: bool = False):
        if preprocessed and not resident:
            raise ValueError("preprocessed inputs must already be resident")
        model, c, device = self.bundle.model, self.bundle.config, self.bundle.device
        payload = self.resident_inputs(preprocessed=preprocessed) if resident else self.host_inputs()
        source = payload["tokens"] if isinstance(c, BPEConfig) else payload["bytes"]
        if isinstance(c, BPEConfig) and self._repeated_prompt:
            # The separate static-decode benchmark uses one shared prefix and
            # expects the original uniform KVCache interface.
            source = source[:, :self.real_lengths[0]]
        state = model.init_state(self.batch, source.shape[1] + headroom, device,
                                 torch.bfloat16 if device.type == "cuda" else torch.float32)
        if isinstance(c, BPEConfig):
            logits = model.prefill(state, source, lengths=None if self._repeated_prompt else payload["lengths"])
        else:
            patching = "preprocessed" if preprocessed or not resident else "device"
            logits = model.prefill(state, payload, lengths=payload["lengths"], patching=patching,
                                   patch_capacity=self.patch_capacity)
        return state, logits, None

    def decode(self, state: BPEState | ByteState, logits: torch.Tensor) -> dict[str, Any]:
        model, config = self.bundle.model, self.bundle.config
        lengths = [0] * self.batch
        tokens = [0] * self.batch
        cached_output_bytes = 0
        while min(lengths) < self.output_bytes:
            if isinstance(config, BPEConfig):
                selected = logits[:, -1, :].masked_fill(~self.allowed, -torch.inf).argmax(dim=-1)
                ids = selected.tolist()
                for row, token_id in enumerate(ids):
                    if lengths[row] < self.output_bytes:
                        value = self.token_bytes.values[token_id]
                        if not value:
                            raise ValueError("generated token has no byte representation")
                        lengths[row] += len(value)
                        tokens[row] += 1
            else:
                selected = logits[:, -1, :256].argmax(dim=-1)
                for row in range(self.batch):
                    if lengths[row] < self.output_bytes:
                        lengths[row] += 1
                        tokens[row] += 1
            if min(lengths) < self.output_bytes:
                if isinstance(config, BPEConfig):
                    cached_output_bytes += sum(len(self.token_bytes.values[token_id]) for token_id in ids)
                else:
                    cached_output_bytes += self.batch
                logits = model.step(state, selected)
        return {"output_bytes": sum(lengths), "output_bytes_per_row": lengths,
                "overshoot_bytes": [length - self.output_bytes for length in lengths], "generated_units": tokens,
                "cached_output_bytes": cached_output_bytes}


class CUDAGraphPatcher:
    """Fixed-shape stage A: device patching and hashes with no host reads."""

    @torch.inference_mode()
    def __init__(self, model: ByteLatentLM, inputs: torch.Tensor | dict[str, torch.Tensor], *, warmup: int = 3) -> None:
        source = inputs["bytes"] if isinstance(inputs, dict) else inputs
        if source.device.type != "cuda" or not torch.cuda.is_available() or warmup < 1:
            raise ValueError("CUDA graph patching needs CUDA inputs and at least one warm-up")
        if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
            raise ValueError("CUDA graph patching requires nonempty [batch, time] inputs")
        self.device = source.device
        self.inputs = CUDAGraphPrefill._clone(inputs)

        def run():
            if isinstance(self.inputs, dict):
                return model.device_patch_metadata(self.inputs["bytes"], lengths=self.inputs["lengths"])
            return model.device_patch_metadata(self.inputs)

        torch.cuda.synchronize(self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                run()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        with torch.cuda.device(self.device):
            self.graph = torch.cuda.CUDAGraph()
            capture_started = time.perf_counter()
            with torch.cuda.graph(self.graph, stream=stream):
                self.metadata = run()
            torch.cuda.synchronize(self.device)
            self.capture_seconds = time.perf_counter() - capture_started
        # First replay can have one-off costs; it must never be timed.
        self.replay(inputs)

    @torch.inference_mode()
    def replay(self, inputs: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        with torch.cuda.device(self.device):
            CUDAGraphPrefill._copy_inputs(self.inputs, inputs)
            self.graph.replay()
        return self.metadata

    def __getstate__(self):
        raise TypeError("CUDA graphs are process-local; recreate and recapture after resume")


class CUDAGraphPrefill:
    """A process-local graph for one fixed input shape and patching definition.

    Construction includes warm-up, capture and an untimed replay. replay() copies
    already-resident inputs into fixed-address buffers, fills the same KV caches
    and returns the next-step logits. Recreate it when shape, model or precision
    changes. Device patching uses a separate graph and a fixed captured K.
    """

    @torch.inference_mode()
    def __init__(self, model: BPELM | ByteLatentLM, inputs: torch.Tensor | dict[str, Any], *,
                 max_len: int, patching: str = "device", warmup: int = 3,
                 patch_capacity: str = "bucket64") -> None:
        source = ((inputs["tokens"] if isinstance(model, BPELM) else inputs["bytes"])
                  if isinstance(inputs, dict) else inputs)
        if source.device.type != "cuda" or not torch.cuda.is_available() or warmup < 1:
            raise ValueError("CUDA graph prefill needs CUDA inputs and at least one warm-up")
        if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0 or max_len < source.shape[1]:
            raise ValueError("graph prefill needs a nonempty prefix and sufficient cache capacity")
        if patch_capacity not in {"exact", "bucket64", "upper_bound"}:
            raise ValueError("patch_capacity must be exact, bucket64 or upper_bound")
        self.device_patcher = None
        self.patch_capacity = patch_capacity
        if isinstance(model, BPELM):
            if isinstance(inputs, dict) and set(inputs) != {"tokens", "lengths"}:
                raise ValueError("BPE graph inputs must contain tokens and lengths")
        else:
            if patching not in {"device", "preprocessed"} or (patching == "preprocessed") != isinstance(inputs, dict):
                raise ValueError("byte graph inputs must match the patching definition")
            if patching == "device":
                self.device_patcher = CUDAGraphPatcher(model, source, warmup=warmup)
                metadata = self.device_patcher.replay(source)
                count = int(metadata["counts"].max().item()) if patch_capacity != "upper_bound" else 0
                self.capacity = choose_patch_capacity(count, metadata["ends"].shape[1], patch_capacity)
                inputs = {"bytes": source, "patches": trim_patch_metadata(metadata, self.capacity)}
                patching = "preprocessed"
        self.inputs = self._clone(inputs)
        self.model, self.patching = model, patching
        self.device = source.device
        self.prefix_length = source.shape[1]
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled() else next(model.parameters()).dtype
        self.state = model.init_state(source.shape[0], max_len, source.device, dtype)

        def reset() -> None:
            if isinstance(self.state, BPEState):
                if isinstance(self.state.cache, BatchedKVCache):
                    self.state.cache.lengths = [0] * source.shape[0]
                else:
                    self.state.cache.length = 0
            else:
                self.state.position = 0
                self.state._routing = None
                for cache in (self.state.encoder, self.state.decoder, self.state.globals):
                    cache._lengths_loader = None

        def run() -> torch.Tensor:
            if isinstance(model, BPELM):
                if isinstance(self.inputs, dict):
                    return model.prefill(self.state, self.inputs["tokens"], lengths=self.inputs["lengths"])
                return model.prefill(self.state, self.inputs)
            lengths = self.inputs.get("lengths") if isinstance(self.inputs, dict) else None
            return model.prefill(self.state, self.inputs, lengths=lengths, patching=patching)

        device = source.device
        torch.cuda.synchronize(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                reset()
                run()
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        reset()
        with torch.cuda.device(device):
            self.graph = torch.cuda.CUDAGraph()
            capture_started = time.perf_counter()
            with torch.cuda.graph(self.graph, stream=stream):
                self.logits = run()
            torch.cuda.synchronize(device)
            self.capture_seconds = time.perf_counter() - capture_started
        if isinstance(self.state, ByteState):
            assert self.state._routing is not None
            self.routing = self.state._routing
            self.conditioning = self.state.conditioning
        else:
            self.routing = None
            self.conditioning = None
        self.length_loader = (self.state.cache._lengths_loader if isinstance(self.state, BPEState) and
                              isinstance(self.state.cache, BatchedKVCache) else None)
        self.replay(source if self.device_patcher is not None else inputs)

    @staticmethod
    def _copy(target: torch.Tensor, source: torch.Tensor) -> None:
        if target.shape != source.shape or target.dtype != source.dtype or target.device != source.device:
            raise ValueError("graph replay inputs must retain their captured shapes, dtypes and device")
        target.copy_(source)

    @staticmethod
    def _clone(value):
        return {key: CUDAGraphPrefill._clone(item) for key, item in value.items()} if isinstance(value, dict) else value.clone()

    @staticmethod
    def _copy_inputs(target, source) -> None:
        if isinstance(target, dict):
            if not isinstance(source, dict) or target.keys() != source.keys():
                raise ValueError("graph replay metadata does not match capture")
            for key in target:
                CUDAGraphPrefill._copy_inputs(target[key], source[key])
        elif isinstance(source, dict):
            raise ValueError("graph replay input does not match capture")
        else:
            CUDAGraphPrefill._copy(target, source)

    @torch.inference_mode()
    def replay(self, inputs: torch.Tensor | dict[str, Any]) -> tuple[BPEState | ByteState, torch.Tensor]:
        if self.device_patcher is not None:
            if isinstance(inputs, dict):
                raise ValueError("device-patching graph inputs must be byte tensors")
            metadata = self.device_patcher.replay(inputs)
            count = int(metadata["counts"].max().item()) if self.patch_capacity != "upper_bound" else 0
            capacity = choose_patch_capacity(count, metadata["ends"].shape[1], self.patch_capacity)
            if capacity != self.capacity:
                raise ValueError("patch capacity changed; recapture the trunk graph")
            inputs = {"bytes": inputs, "patches": trim_patch_metadata(metadata, capacity)}
        with torch.cuda.device(self.device):
            self._copy_inputs(self.inputs, inputs)
            self.graph.replay()
        if isinstance(self.state, ByteState):
            self.state.position = self.prefix_length
            self.state._conditioning = self.conditioning
            self.state._logits = self.logits
            self.state._routing = self.routing
            for cache in (self.state.encoder, self.state.decoder, self.state.globals):
                cache._lengths_loader = self.state._materialize_routing
        else:
            if isinstance(self.state.cache, BatchedKVCache):
                self.state.cache._lengths_loader = self.length_loader
            else:
                self.state.cache.length = self.prefix_length
        return self.state, self.logits

    def __getstate__(self):
        raise TypeError("CUDA graphs are process-local; recreate and recapture after resume")


def compile_inference(model: torch.nn.Module) -> None:
    """Compile the tensor paths; Python routing and tokenization remain timed."""
    trunks = [model.trunk] if hasattr(model, "trunk") else [model.encoder, model.global_trunk, model.decoder]
    for trunk in trunks:
        trunk.forward = torch.compile(trunk.forward, dynamic=True)
        trunk.step = torch.compile(trunk.step, dynamic=True)
        trunk.step_batched = torch.compile(trunk.step_batched, dynamic=True)
        trunk.step_batched_tensor = torch.compile(trunk.step_batched_tensor, dynamic=True)


def paired_speed_interval(a_seconds: list[float], b_seconds: list[float], *, a_bytes: list[int] | None = None,
                          b_bytes: list[int] | None = None, seed: int = 0, samples: int = 2000) -> dict[str, float]:
    if len(a_seconds) != len(b_seconds) or not a_seconds or samples <= 0:
        raise ValueError("paired timing samples must be nonempty and aligned")
    a, b = np.asarray(a_seconds), np.asarray(b_seconds)
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("timings must be positive")
    a_sizes = np.ones_like(a) if a_bytes is None else np.asarray(a_bytes)
    b_sizes = np.ones_like(b) if b_bytes is None else np.asarray(b_bytes)
    if a_sizes.shape != a.shape or b_sizes.shape != b.shape or np.any(a_sizes <= 0) or np.any(b_sizes <= 0):
        raise ValueError("byte counts must be positive and aligned with timings")
    a_speed, b_speed = a_sizes / a, b_sizes / b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(a), size=(samples, len(a)))
    ratios = np.median(a_speed[indices], axis=1) / np.median(b_speed[indices], axis=1)
    return {"speed_a_over_b": float(np.median(a_speed) / np.median(b_speed)),
            "ci95_low": float(np.quantile(ratios, 0.025)), "ci95_high": float(np.quantile(ratios, 0.975))}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class GraphCache:
    """A bounded LRU shared by the patcher and trunk graphs of one variant/model."""

    def __init__(self, max_size: int) -> None:
        if max_size <= 0:
            raise ValueError("graph cache size must be positive")
        self.max_size = max_size
        self.entries: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self.captures: list[dict[str, Any]] = []
        self.hits = self.misses = 0

    def get(self, key: tuple[Any, ...]):
        value = self.entries.get(key)
        if value is not None:
            self.entries.move_to_end(key)
        return value

    def put(self, key: tuple[Any, ...], value: Any):
        self.entries[key] = value
        self.entries.move_to_end(key)
        return self.entries.popitem(last=False)[0] if len(self.entries) > self.max_size else None

    def make_room(self):
        if len(self.entries) < self.max_size:
            return None
        key, graph = self.entries.popitem(last=False)
        del graph
        return key

    def record_trial(self, hit: bool) -> None:
        if hit:
            self.hits += 1
        else:
            self.misses += 1


def _run_graph_or_eager(cache: GraphCache, key: tuple[Any, ...], inputs: Any, eager, *, eligible: bool = True):
    """Lookup and run inside the caller's timed interval; capture is deliberately separate."""
    runner = cache.get(key) if eligible else None
    if runner is None:
        return eager(), False
    return runner.replay(inputs), True


def _capture_missing(cache: GraphCache, key: tuple[Any, ...], factory, device: torch.device, *,
                     phase: str, trial: int) -> bool:
    if cache.get(key) is not None:
        return False
    _sync(device)
    evicted = cache.make_room()
    if evicted is not None:
        gc.collect()  # Release static graph/cache cycles before allocating a replacement.
    started = time.perf_counter()
    graph = factory()  # Constructors include one untimed replay after capture.
    _sync(device)
    duration = time.perf_counter() - started
    cache.put(key, graph)
    cache.captures.append({"stage": key[0], "key": list(key), "phase": phase, "trial": trial,
                           "seconds": getattr(graph, "capture_seconds", duration),
                           "total_setup_seconds": duration, "cache_size": len(cache.entries),
                           "evicted": list(evicted) if evicted else None})
    return True


def _trunk_key(inputs: dict[str, Any]) -> tuple[Any, ...]:
    if "tokens" in inputs:
        return ("trunk", "bpe", *inputs["tokens"].shape)
    return ("trunk", "byte", *inputs["bytes"].shape, inputs["patches"]["ends"].shape[1])


def _patcher_key(inputs: dict[str, Any]) -> tuple[Any, ...]:
    return ("patcher", *inputs["bytes"].shape)


def _eager_preprocessed(case: InferenceCase, inputs: dict[str, Any]):
    model, device = case.bundle.model, case.bundle.device
    is_bpe = isinstance(case.bundle.config, BPEConfig)
    source = inputs["tokens"] if is_bpe else inputs["bytes"]
    state = model.init_state(source.shape[0], source.shape[1], device,
                             torch.bfloat16 if device.type == "cuda" else torch.float32)
    if is_bpe:
        logits = model.prefill(state, source, lengths=inputs["lengths"])
    else:
        logits = model.prefill(state, inputs, lengths=inputs["lengths"], patching="preprocessed")
    return state, logits


def _software_environment(device: torch.device) -> dict[str, Any]:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "numpy": np.__version__, "tokenizers": version("tokenizers"),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cpu_threads": {"torch": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(),
                            "environment": {name: os.environ.get(name) for name in
                                            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                             "TOKENIZERS_PARALLELISM", "CUDA_VISIBLE_DEVICES")}}}


def _gpu_telemetry(device: torch.device) -> dict[str, Any] | None:
    if device.type != "cuda":
        return None
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible_devices = visible.split(",") if visible else []
        smi_id = visible_devices[index].strip() if index < len(visible_devices) else str(index)
        result = subprocess.run(["nvidia-smi", f"--id={smi_id}",
                                 "--query-gpu=index,clocks.sm,clocks.mem,power.draw,temperature.gpu,driver_version",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                check=True, timeout=5)
        values = [value.strip() for value in result.stdout.strip().split(",")]
        if len(values) != 6:
            raise ValueError("unexpected nvidia-smi output")
        return dict(zip(("index", "sm_clock_mhz", "memory_clock_mhz", "power_watts", "temperature_c",
                         "driver_version"), values))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"unavailable": str(error)}


def benchmark(bundles: list[EvaluationModel], prompts: list[bytes], **kwargs: Any) -> dict[str, Any]:
    """Run under one outer autocast so its bf16 weight-cast cache lives for the whole benchmark.

    Each per-trial autocast block is then nested and does not clear the cache. Without this, a CUDA graph captured
    with cached bf16 weights read them after the cache was cleared at the end of a trial (illegal memory access on
    the 4070). Holding the casts matches a server with resident bf16 weights, for both families.
    """
    device = bundles[0].device if bundles else torch.device("cpu")
    with autocast_for(device):
        result = _benchmark(bundles, prompts, **kwargs)
    result["weight_cast"] = "bf16 autocast cache held for the whole benchmark (weights cast once)"
    return result


class MemoryBudgetExceeded(RuntimeError):
    def __init__(self, peak: int, budget: int, phase: str) -> None:
        super().__init__(f"reserved memory {peak} exceeds the {budget}-byte budget during {phase}")
        self.peak, self.budget, self.phase = peak, budget, phase


def _check_budget(device: torch.device, budget: int | None, phase: str) -> int:
    if device.type != "cuda":
        return 0
    peak = torch.cuda.max_memory_reserved(device)
    if budget is not None and peak > budget:
        raise MemoryBudgetExceeded(peak, budget, phase)
    return peak


@torch.inference_mode()
def _benchmark(bundles: list[EvaluationModel], prompts: list[bytes], *, mode: str, batch: int = 1,
               output_bytes: int = 256, warmup: int = 10, repetitions: int = 30, seed: int = 0,
               compiled: bool = False, execution: str | None = None, patch_capacity: str = "bucket64",
               graph_cache_size: int = 8, memory_budget_bytes: int | None = 10_500_000_000,
               calibrate_only: bool = False, byte_end_to_end_patching: str = "host") -> dict[str, Any]:
    if (len(bundles) not in (1, 2) or mode not in {"prefill", "decode", "e2e"} or warmup < 0 or
            repetitions <= 0 or batch <= 0 or graph_cache_size <= 0 or byte_end_to_end_patching not in {"host", "device"}):
        raise ValueError("invalid benchmark settings")
    needed = (warmup + repetitions) * batch
    if len(prompts) < needed:
        raise ValueError("provide batch distinct prompts for each warm-up and repetition")
    prompts = prompts[:needed]
    groups = [prompts[i:i + batch] for i in range(0, needed, batch)]
    if any(len(set(group)) != batch or any(not prompt for prompt in group) for group in groups):
        raise ValueError("each batch must contain different nonempty prompts")
    if set(prompts[:warmup * batch]) & set(prompts[warmup * batch:]):
        raise ValueError("calibration prompts must differ from timed prompts")
    if any(bundle.device != bundles[0].device for bundle in bundles):
        raise ValueError("comparison models must use the same device")
    device = bundles[0].device
    execution = ("compile" if compiled else "eager") if execution is None else execution
    if execution not in {"eager", "compile", "graph"} or (compiled and execution != "compile"):
        raise ValueError("execution must be eager, compile or graph")
    if patch_capacity not in {"exact", "bucket64", "upper_bound"}:
        raise ValueError("patch_capacity must be exact, bucket64 or upper_bound")
    if execution == "graph" and device.type != "cuda":
        raise ValueError("CUDA graph prefill requires CUDA")
    compiled = execution == "compile"
    variants = ("end_to_end", "gpu_resident_preprocessed", "gpu_resident") if mode == "prefill" else (mode,)
    token_vocabularies = [TokenBytes(bundle.tokenizer, bundle.config.vocab_size) if isinstance(bundle.config, BPEConfig) else None
                          for bundle in bundles]
    output: dict[str, Any] = {
        "mode": mode, "batch": batch, "warmup": warmup, "repetitions": repetitions,
        "requested_output_bytes_per_row": output_bytes,
        "generation_policy": "greedy over raw-byte outputs; EOS and special tokens suppressed",
        "compiled": compiled, "execution": execution, "capacity": patch_capacity, "patch_capacity": patch_capacity,
        "byte_end_to_end_patching": byte_end_to_end_patching,
        "byte_end_to_end_patching_definition": ("host: numpy patching and hashes on the CPU, then transfer; device: "
                                                "transfer the bytes, then device patching and hashes, one timed "
                                                "counts.max() read and the trunk, as in gpu_resident."),
        "capacity_policy_definitions": {
            "bucket64": ("BPE token positions round up to 64; byte patch counts round up to 64 and are capped "
                         "at max_patch_count. Byte input widths round up to 64."),
            "exact": "Use the batch maximum real token length or completed-patch count.",
            "upper_bound": ("Use max_patch_count for bytes; for BPE use at most one token per input byte plus BOS, "
                            "bounded by the configured position limit."),
        },
        "environment": _software_environment(device),
        "effective_execution": {variant: ("eager" if execution == "graph" and mode != "prefill" else execution)
                                for variant in variants},
        "prefill_definition_names": {"gpu_resident": "GPU-resident (device patching)",
                                     "gpu_resident_preprocessed": "GPU-resident (preprocessed)",
                                     "end_to_end": "End-to-end warm service"},
        "prefill_definitions": {
            "end_to_end": ("Host UTF-8 bytes through per-row tokenization or patching and hashes, host-to-device "
                           "transfers, graph lookup, static-buffer copies, replay or eager miss fallback, cache writes "
                           "and last-position native logits; final synchronization included. Capture, conversion, "
                           "later host decode routing and cache accounting are excluded."),
            "gpu_resident_preprocessed": ("Resident tokens or bytes and host-prepared patch/hash metadata through graph "
                                         "lookup, static-buffer copies, replay or eager miss fallback, cache writes and "
                                         "last-position native logits; final synchronization included. Host preprocessing, "
                                         "transfers, capture, conversion, later host decode routing and cache accounting excluded."),
            "gpu_resident": ("Resident tokens or bytes; for bytes, device patching and hashes followed by one timed "
                             "device-to-host counts.max() read and bucket selection; then graph lookup, static-buffer "
                             "copies, replay or eager miss fallback, cache writes and native logits. Host preprocessing, "
                             "transfers, capture, conversion, later host decode routing and cache accounting excluded."),
        },
        "prefill_cache_allocation": ("Captured graph caches are allocated during setup; eager prefill and timed "
                                     "graph-miss fallback allocate their caches within the timed interval."),
        "conversion_definition": ("Separately synchronized conversion of native logits to the common byte distribution: "
                                  "BPE first_byte_probabilities or byte softmax; excluded from prefill seconds."),
        "timing_attribution": ("preprocess_seconds ends immediately after host preprocessing and before transfers; zero "
                               "for resident variants. Within it, encode_seconds is tokenization (with UTF-8 decoding) for "
                               "BPE, or patching plus hashes for bytes; the rest is host tensor construction. CUDA events "
                               "bracket device work and may include idle time between submissions; total seconds include "
                               "the final synchronization."),
        "graph_capture_in_timing": False,
        "graph_replay_definition": ("The calibration prompts populate a bounded per-model, per-variant LRU, whose "
                                    "membership is then frozen: a timed miss uses eager fallback inside the timed "
                                    "interval and is never captured. Each capture includes an untimed replay. Patcher "
                                    "and trunk graphs share the LRU bound."),
        "memory_budget_bytes": memory_budget_bytes,
        "memory_budget_definition": ("Whole-process torch reserved memory (both models, retained graphs, capture and "
                                     "eager-fallback peaks) must stay within the budget through calibration and every "
                                     "timed trial, otherwise the cell raises MemoryBudgetExceeded."),
        "graph_capture_times_definition": ("Capture seconds cover CUDA graph capture and its completion; "
                                           "total setup seconds also include warm-up and one replay."),
        "prompt_accounting": ("prompt_sha256 identifies the ordered batch (the ordinary prompt digest for batch 1); "
                              "prompt_bytes is the mean per-row length. The *_per_row fields retain every input."),
        "parameter_bytes": [sum(p.numel() * p.element_size() for p in getattr(bundle.model, "parameters", list)())
                            for bundle in bundles],
        "persistent_footprint_definition": ("fp32 parameters (parameter_bytes) plus the autocast cache of bf16 weight "
                                            "copies held for the whole benchmark plus retained graph pools; "
                                            "baseline_allocated_bytes is measured before each timed trial."),
        "host_python_object_overhead_in_cache_bytes": False, "variants": {}}

    for variant in variants:
        graph_caches = [GraphCache(graph_cache_size) for _ in bundles]
        rows = [{"identity": bundle.identity, "seconds": [], "total_seconds": [], "preprocess_seconds": [],
                 "encode_seconds": [],
                 "gpu_event_seconds": [], "conversion_seconds": [], "bytes": [], "peak_allocated_bytes": 0,
                 "peak_reserved_bytes": 0, "baseline_allocated_bytes": [], "baseline_reserved_bytes": [],
                 "compile_new_graphs": [], "cache": [], "realised_bytes_per_unit": [], "overshoot_bytes": [],
                 "prompt_sha256": [], "prompt_bytes": [], "prompt_sha256_per_row": [], "prompt_bytes_per_row": [],
                 "capacities": [], "graph_hits": [], "graph_misses": [],
                 "graph_stage_hits": [], "graph_cache_sizes": [], "graph_captures": [], "graph_stage_captures": []}
                for bundle in bundles]
        gpu_before = _gpu_telemetry(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        variant_peak = 0
        for trial, group in enumerate(groups):
            if calibrate_only and trial >= warmup:
                break
            prepare_resident = not (mode == "e2e" or (mode == "prefill" and variant == "end_to_end"))
            cases = [InferenceCase(bundle, group, batch, output_bytes, vocabulary, prepare_resident,
                                   prepare_preprocessed=variant == "gpu_resident_preprocessed",
                                   patch_capacity=patch_capacity)
                     for bundle, vocabulary in zip(bundles, token_vocabularies)]
            order = list(range(len(bundles)))
            if trial % 2:
                order.reverse()
            for i in order:
                case, record, cache = cases[i], rows[i], graph_caches[i]
                is_byte = isinstance(case.bundle.config, ByteConfig)
                # End-to-end byte patching on the device: host bytes are transferred, then patched as in gpu_resident.
                device_e2e = mode == "prefill" and variant == "end_to_end" and is_byte and byte_end_to_end_patching == "device"
                device_stage = mode == "prefill" and (variant == "gpu_resident" or device_e2e) and is_byte
                with autocast_for(device):
                    if mode == "prefill" and execution == "graph" and trial < warmup:
                        inputs = (case.to_device(case.prepare_host(preprocessed=not device_e2e)) if variant == "end_to_end"
                                  else case.resident_inputs(preprocessed=variant == "gpu_resident_preprocessed"))
                        if device_stage:
                            pkey = _patcher_key(inputs)
                            _capture_missing(cache, pkey, lambda: CUDAGraphPatcher(case.bundle.model, inputs),
                                             device, phase="calibration", trial=trial)
                            metadata = cache.get(pkey).replay(inputs)
                            actual = int(metadata["counts"].max().item())
                            capacity = choose_patch_capacity(actual, metadata["ends"].shape[1], patch_capacity)
                            inputs = {**inputs, "patches": trim_patch_metadata(metadata, capacity)}
                        key = _trunk_key(inputs)
                        source = inputs["bytes"] if is_byte else inputs["tokens"]
                        _capture_missing(cache, key, lambda: CUDAGraphPrefill(
                            case.bundle.model, inputs, max_len=source.shape[1],
                            patching="preprocessed" if is_byte else "device"),
                            device, phase="calibration", trial=trial)
                        calibration_state, logits = cache.get(key).replay(inputs)
                        case.distribution(logits)
                        _sync(device)
                        calibration_state = logits = None
                        variant_peak = max(variant_peak, _check_budget(device, memory_budget_bytes, "calibration"))
                        continue

                    state = logits = generated = None
                    if mode == "decode":
                        state, logits, _ = case.prefill(resident=True, headroom=output_bytes)
                    _sync(device)
                    if device.type == "cuda" and trial >= warmup:
                        record["baseline_allocated_bytes"].append(torch.cuda.memory_allocated(device))
                        record["baseline_reserved_bytes"].append(torch.cuda.memory_reserved(device))
                        variant_peak = max(variant_peak, torch.cuda.max_memory_reserved(device))
                        torch.cuda.reset_peak_memory_stats(device)
                    if compiled:
                        from torch._dynamo.utils import counters
                        graphs_before = counters["stats"].get("unique_graphs", 0)
                    start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" and mode == "prefill" else None
                    end_event = torch.cuda.Event(enable_timing=True) if start_event is not None else None
                    preprocess_seconds = 0.0
                    stage_hits: dict[str, bool | None] = {"patcher": None, "trunk": None}
                    pkey = key = None
                    patcher_inputs = inputs = None
                    started = time.perf_counter()
                    if mode == "prefill":
                        if variant == "end_to_end":
                            host = case.prepare_host(preprocessed=not device_e2e)
                            preprocess_seconds = time.perf_counter() - started
                            if start_event is not None:
                                start_event.record(torch.cuda.current_stream(device))
                            inputs = case.to_device(host)
                        else:
                            if start_event is not None:
                                start_event.record(torch.cuda.current_stream(device))
                            inputs = case.resident_inputs(preprocessed=variant == "gpu_resident_preprocessed")
                        if device_stage:
                            patcher_inputs = inputs
                            pkey = _patcher_key(inputs)
                            if execution == "graph":
                                metadata, stage_hits["patcher"] = _run_graph_or_eager(
                                    cache, pkey, inputs, lambda: case.bundle.model.device_patch_metadata(
                                        inputs["bytes"], lengths=inputs["lengths"]))
                            else:
                                metadata = case.bundle.model.device_patch_metadata(inputs["bytes"], lengths=inputs["lengths"])
                            actual = int(metadata["counts"].max().item())
                            capacity = choose_patch_capacity(actual, metadata["ends"].shape[1], patch_capacity)
                            case.actual_capacity, case.padded_capacity = actual, capacity
                            inputs = {**inputs, "patches": trim_patch_metadata(metadata, capacity)}
                        if execution == "graph":
                            key = _trunk_key(inputs)
                            eligible = not device_stage or bool(stage_hits["patcher"])
                            (state, logits), stage_hits["trunk"] = _run_graph_or_eager(
                                cache, key, inputs, lambda: _eager_preprocessed(case, inputs), eligible=eligible)
                        else:
                            state, logits = _eager_preprocessed(case, inputs)
                        if end_event is not None:
                            end_event.record(torch.cuda.current_stream(device))
                    elif mode == "e2e":
                        state, logits, _ = case.prefill(resident=False, headroom=output_bytes)
                        generated = case.decode(state, logits)
                    else:
                        generated = case.decode(state, logits)
                    _sync(device)
                    elapsed = time.perf_counter() - started
                    gpu_seconds = start_event.elapsed_time(end_event) / 1000.0 if start_event is not None else None
                    if compiled:
                        new_graphs = counters["stats"].get("unique_graphs", 0) - graphs_before
                    if device.type == "cuda" and trial >= warmup:
                        record["peak_allocated_bytes"] = max(record["peak_allocated_bytes"], torch.cuda.max_memory_allocated(device))
                        record["peak_reserved_bytes"] = max(record["peak_reserved_bytes"], torch.cuda.max_memory_reserved(device))
                    variant_peak = max(variant_peak, _check_budget(device, memory_budget_bytes,
                                                                   "calibration" if trial < warmup else "timing"))
                    conversion_seconds = None
                    if mode == "prefill":
                        conversion_started = time.perf_counter()
                        case.distribution(logits)
                        _sync(device)
                        conversion_seconds = time.perf_counter() - conversion_started

                    if trial < warmup:
                        continue
                    count = sum(map(len, group)) if generated is None else generated["output_bytes"]
                    record["seconds"].append(elapsed)
                    record["total_seconds"].append(elapsed)
                    record["preprocess_seconds"].append(preprocess_seconds if mode == "prefill" else None)
                    record["encode_seconds"].append(getattr(case, "encode_seconds", None)
                                                    if mode == "prefill" and variant == "end_to_end" else None)
                    record["encoder"] = getattr(case, "encoder_name", None) if variant == "end_to_end" else None
                    record["gpu_event_seconds"].append(gpu_seconds)
                    record["conversion_seconds"].append(conversion_seconds)
                    digests = [hashlib.sha256(prompt).digest() for prompt in group]
                    record["prompt_sha256"].append((digests[0] if batch == 1 else
                                                    hashlib.sha256(b"".join(digests)).digest()).hex())
                    record["prompt_sha256_per_row"].append([digest.hex() for digest in digests])
                    record["prompt_bytes"].append(len(group[0]) if batch == 1 else sum(map(len, group)) / batch)
                    record["prompt_bytes_per_row"].append([len(prompt) for prompt in group])
                    record["bytes"].append(count)
                    cached_bytes = sum(map(len, group)) + (generated["cached_output_bytes"] if generated else 0)
                    record["cache"].append(cache_statistics(state, cached_bytes, logits))
                    if isinstance(case.bundle.config, BPEConfig):
                        record["realised_bytes_per_unit"].append(sum(map(len, group)) / case.total_prompt_tokens)
                    else:
                        positions = sum(p.count + int(p.length > 0) for p in state.patchers)
                        record["realised_bytes_per_unit"].append(sum(p.position for p in state.patchers) / max(positions, 1))
                    if generated is not None:
                        record["overshoot_bytes"].append(generated["overshoot_bytes"])
                    if compiled:
                        record["compile_new_graphs"].append(new_graphs)

                    if mode == "prefill":
                        source = inputs["bytes"] if is_byte else inputs["tokens"]
                        record["capacities"].append({"actual": case.actual_capacity, "padded": case.padded_capacity,
                                                     "input_positions": source.shape[1],
                                                     "row_lengths": list(case.real_lengths)})
                        if execution == "graph":
                            hit = bool(stage_hits["trunk"] and (not device_stage or stage_hits["patcher"]))
                            cache.record_trial(hit)
                            record["graph_hits"].append(hit)
                            record["graph_misses"].append(not hit)
                            record["graph_stage_hits"].append(stage_hits.copy())
                            # Membership is frozen after calibration (Sol round 9): a timed miss runs eager and is
                            # not captured afterwards, so no setup work is hidden from later requests.
                            record["graph_captures"].append(0)
                            record["graph_stage_captures"].append({"device_patching": 0, "trunk": 0})
                            record["graph_cache_sizes"].append(len(cache.entries))

        variant_peak = max(variant_peak, _check_budget(device, memory_budget_bytes,
                                                       "calibration" if calibrate_only else "timing"))
        if calibrate_only:
            output["variants"][variant] = {"peak_reserved_bytes": variant_peak,
                                           "graph_cache_sizes": [len(cache.entries) for cache in graph_caches],
                                           "captures": [cache.captures for cache in graph_caches]}
            cases = case = cache = inputs = patcher_inputs = metadata = None
            del graph_caches
            gc.collect()
            continue
        for bundle, record, cache in zip(bundles, rows, graph_caches):
            times = np.asarray(record["seconds"])
            record["median_seconds"] = float(np.median(times))
            record["p95_seconds"] = float(np.quantile(times, 0.95))
            record["bytes_per_second"] = float(np.median(np.asarray(record["bytes"]) / times))
            unit = "bytes_per_token" if isinstance(bundle.config, BPEConfig) else "mean_patch_length"
            record[unit] = record["realised_bytes_per_unit"]
            hits, misses = cache.hits, cache.misses
            record["graph_cache"] = {"max_size": graph_cache_size, "size": len(cache.entries), "hits": hits,
                                     "misses": misses, "miss_rate": misses / (hits + misses) if hits + misses else None,
                                     "captures": cache.captures,
                                     "keys_lru": [list(key) for key in cache.entries]}
        value: dict[str, Any] = {"models": rows, "gpu_before": gpu_before, "gpu_after": _gpu_telemetry(device),
                                 "peak_reserved_bytes_whole_variant": variant_peak}
        if len(rows) == 2:
            value["paired_bootstrap"] = paired_speed_interval(rows[0]["seconds"], rows[1]["seconds"],
                                                              a_bytes=rows[0]["bytes"], b_bytes=rows[1]["bytes"], seed=seed)
        output["variants"][variant] = value
        # Variants have separate graph caches; don't retain their static GPU state
        # while measuring the next variant.
        runner = state = logits = inputs = patcher_inputs = metadata = cases = case = cache = source = None
        del graph_caches
        gc.collect()
    return output
