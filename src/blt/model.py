from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .config import BPEConfig, ByteConfig, TransformerConfig
from .hashes import NGRAMS, RollingHashes, hash_parameters, hash_tables, hashes_at_ends
from .patcher import (IncrementalPatcher, max_patch_count, patch_batch, patch_batch_torch,
                      patch_capacity as choose_patch_capacity, trim_patch_metadata)


def _initialize(module: nn.Module) -> None:
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype=x.dtype)


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    dim = x.shape[-1]
    frequencies = theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
    positions = positions.to(device=x.device, dtype=torch.float32)
    angles = (positions[None, None, :, None] if positions.ndim == 1 else positions[:, None, :, None]) * frequencies
    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class RotaryTables(nn.Module):
    """Non-checkpointed tables shared by all layers of a transformer."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.head_dim, self.theta, self.max_positions = config.head_dim, config.rope_theta, config.max_positions
        for name in ("cos_float", "sin_float", "cos_bfloat16", "sin_bfloat16",
                     "cos_float16", "sin_float16", "cos_float64", "sin_float64"):
            self.register_buffer(name, torch.empty(0), persistent=False)
        self._build(self.cos_float.device)

    def _build(self, device: torch.device) -> None:
        # Use the destination device and FP32 formula from rope(), then cast once.
        frequencies = self.theta ** (-torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        angles = torch.arange(self.max_positions, device=device, dtype=torch.float32)[:, None] * frequencies
        cos, sin = angles.cos(), angles.sin()
        self.cos_float, self.sin_float = cos, sin
        self.cos_bfloat16, self.sin_bfloat16 = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
        self.cos_float16, self.sin_float16 = cos.to(torch.float16), sin.to(torch.float16)
        self.cos_float64, self.sin_float64 = cos.to(torch.float64), sin.to(torch.float64)

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse=recurse)
        # Module.to()/cuda()/dtype changes must not lower the FP32 master table.
        self._build(self.cos_float.device)
        return self

    def tables(self, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if dtype == torch.float32:
            return self.cos_float, self.sin_float
        if dtype == torch.bfloat16:
            return self.cos_bfloat16, self.sin_bfloat16
        if dtype == torch.float16:
            return self.cos_float16, self.sin_float16
        if dtype == torch.float64:
            return self.cos_float64, self.sin_float64
        raise ValueError(f"unsupported RoPE compute dtype: {dtype}")


@dataclass
class KVCache:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    length: int = 0

    def reset(self) -> None:
        self.length = 0


@dataclass(init=False)
class BatchedKVCache:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    _lengths: list[int]
    _lengths_loader: Callable[[], None] | None = None

    def __init__(self, keys: list[torch.Tensor], values: list[torch.Tensor], lengths: list[int]) -> None:
        self.keys, self.values, self._lengths = keys, values, lengths
        self._lengths_loader = None

    @property
    def lengths(self) -> list[int]:
        if self._lengths_loader is not None:
            self._lengths_loader()
        return self._lengths

    @lengths.setter
    def lengths(self, value: list[int]) -> None:
        self._lengths = value
        self._lengths_loader = None

    def __len__(self) -> int:
        return self.keys[0].shape[0]

    def __getitem__(self, row: int) -> KVCache:
        # Compatibility view; cache updates always operate on the batched tensors.
        return KVCache([key[row:row + 1] for key in self.keys],
                       [value[row:row + 1] for value in self.values], self.lengths[row])


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig, rotary: RotaryTables | None = None) -> None:
        super().__init__()
        self.config = config
        if rotary is None:
            self.rotary = RotaryTables(config)
            rotary = self.rotary
        # Transformer owns and moves the shared buffers; this is not another child module.
        object.__setattr__(self, "_rotary", rotary)
        self.q = nn.Linear(config.width, config.q_heads * config.head_dim, bias=False)
        self.k = nn.Linear(config.width, config.kv_heads * config.head_dim, bias=False)
        self.v = nn.Linear(config.width, config.kv_heads * config.head_dim, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)

    def forward(self, x: torch.Tensor, start: int = 0, cache: KVCache | None = None,
                layer: int = 0) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        b, t, _ = x.shape
        c = self.config
        q = self.q(x).view(b, t, c.q_heads, c.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
        if start < 0 or start + t > c.max_positions:
            raise ValueError("position exceeds configured limit")
        cos, sin = self._rotary.tables(q.dtype)
        cos, sin = cos[None, None, start:start + t], sin[None, None, start:start + t]
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        if cache is not None:
            if t != 1 or start >= cache.keys[layer].shape[2]:
                raise ValueError("invalid static-cache step or capacity")
            cache.keys[layer][:, :, start:start + 1].copy_(k)
            cache.values[layer][:, :, start:start + 1].copy_(v)
            keys = cache.keys[layer][:, :, :start + 1]
            values = cache.values[layer][:, :, :start + 1]
        else:
            keys, values = k, v
        # Explicit KV repetition supports the fused CUDA SDPA kernels on GQA models;
        # using enable_gqa on some backends instead selects the math implementation.
        if c.q_heads != c.kv_heads:
            repeats = c.q_heads // c.kv_heads
            keys = keys.repeat_interleave(repeats, dim=1)
            values = values.repeat_interleave(repeats, dim=1)
        y = F.scaled_dot_product_attention(q, keys, values, dropout_p=0.0, is_causal=cache is None and t > 1)
        return self.out(y.transpose(1, 2).contiguous().view(b, t, c.width)), (k, v)

    def step_batched(self, x: torch.Tensor, cache: BatchedKVCache, layer: int,
                     rows: torch.Tensor, positions: torch.Tensor, max_context: int) -> torch.Tensor:
        b, _, _ = x.shape
        c = self.config
        q = self.q(x).view(b, 1, c.q_heads, c.head_dim).transpose(1, 2)
        k = self.k(x).view(b, 1, c.kv_heads, c.head_dim).transpose(1, 2)
        v = self.v(x).view(b, 1, c.kv_heads, c.head_dim).transpose(1, 2)
        cos, sin = self._rotary.tables(q.dtype)
        cos, sin = cos[positions][:, None, None, :], sin[positions][:, None, None, :]
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        cache.keys[layer][rows, :, positions, :] = k.squeeze(2)
        cache.values[layer][rows, :, positions, :] = v.squeeze(2)
        keys = cache.keys[layer][:, :, :max_context].index_select(0, rows)
        values = cache.values[layer][:, :, :max_context].index_select(0, rows)
        if c.q_heads != c.kv_heads:
            repeats = c.q_heads // c.kv_heads
            keys = keys.repeat_interleave(repeats, dim=1)
            values = values.repeat_interleave(repeats, dim=1)
        mask = torch.arange(max_context, device=x.device)[None, None, None, :] <= positions[:, None, None, None]
        y = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask, dropout_p=0.0, is_causal=False)
        return self.out(y.transpose(1, 2).contiguous().view(b, 1, c.width))


class Block(nn.Module):
    def __init__(self, config: TransformerConfig, rotary: RotaryTables | None = None) -> None:
        super().__init__()
        if rotary is None:
            self.rotary = RotaryTables(config)
            rotary = self.rotary
        self.attn_norm = RMSNorm(config.width, config.norm_eps)
        self.attn = Attention(config, rotary)
        self.ffn_norm = RMSNorm(config.width, config.norm_eps)
        self.gate = nn.Linear(config.width, config.ffn, bias=False)
        self.up = nn.Linear(config.width, config.ffn, bias=False)
        self.down = nn.Linear(config.ffn, config.width, bias=False)

    def forward(self, x: torch.Tensor, start: int = 0, cache: KVCache | None = None,
                layer: int = 0) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        a, kv = self.attn(self.attn_norm(x), start, cache, layer)
        x = x + a
        z = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(z)) * self.up(z)), kv

    def step_batched(self, x: torch.Tensor, cache: BatchedKVCache, layer: int,
                     rows: torch.Tensor, positions: torch.Tensor, max_context: int) -> torch.Tensor:
        x = x + self.attn.step_batched(self.attn_norm(x), cache, layer, rows, positions, max_context)
        z = self.ffn_norm(x)
        return x + self.down(F.silu(self.gate(z)) * self.up(z))


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.rotary = RotaryTables(config)
        self.blocks = nn.ModuleList(Block(config, self.rotary) for _ in range(config.layers))
        self.norm = RMSNorm(config.width, config.norm_eps)

    def forward(self, x: torch.Tensor, capture: bool = False) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        if x.shape[1] > self.config.max_positions:
            raise ValueError("position exceeds configured limit")
        saved = []
        for block in self.blocks:
            x, kv = block(x)
            if capture:
                saved.append(kv)
        return self.norm(x), saved

    def cache(self, batch_size: int, capacity: int, device: torch.device | str, dtype: torch.dtype) -> KVCache:
        c = self.config
        if capacity <= 0 or capacity > c.max_positions:
            raise ValueError("invalid cache capacity")
        shape = (batch_size, c.kv_heads, capacity, c.head_dim)
        return KVCache([torch.empty(shape, device=device, dtype=dtype) for _ in self.blocks],
                       [torch.empty(shape, device=device, dtype=dtype) for _ in self.blocks])

    def step(self, x: torch.Tensor, cache: KVCache) -> torch.Tensor:
        if x.shape[1] != 1:
            raise ValueError("a cache step requires one position")
        for i, block in enumerate(self.blocks):
            x, _ = block(x, cache.length, cache, i)
        cache.length += 1
        return self.norm(x)

    def batched_cache(self, batch_size: int, capacity: int, device: torch.device | str, dtype: torch.dtype) -> BatchedKVCache:
        cache = self.cache(batch_size, capacity, device, dtype)
        return BatchedKVCache(cache.keys, cache.values, [0] * batch_size)

    def step_batched(self, x: torch.Tensor, cache: BatchedKVCache, row_ids: list[int], positions: list[int]) -> torch.Tensor:
        if not row_ids or len(row_ids) != x.shape[0] or len(positions) != len(row_ids):
            raise ValueError("invalid batched cache update")
        if min(positions) < 0 or max(positions) >= cache.keys[0].shape[2]:
            raise ValueError("static cache capacity exceeded")
        rows = torch.tensor(row_ids, device=x.device, dtype=torch.long)
        offsets = torch.tensor(positions, device=x.device, dtype=torch.long)
        for layer, block in enumerate(self.blocks):
            x = block.step_batched(x, cache, layer, rows, offsets, max(positions) + 1)
        for row, position in zip(row_ids, positions):
            cache.lengths[row] = position + 1
        return self.norm(x)

    def step_batched_tensor(self, x: torch.Tensor, cache: BatchedKVCache, rows: torch.Tensor,
                            positions: torch.Tensor, max_context: int) -> torch.Tensor:
        """A fixed-shape cache write with device-resident row and position indices."""
        for layer, block in enumerate(self.blocks):
            x = block.step_batched(x, cache, layer, rows, positions, max_context)
        return self.norm(x)


def _copy_cache(cache: KVCache, saved: list[tuple[torch.Tensor, torch.Tensor]], length: int) -> None:
    if length > cache.keys[0].shape[2]:
        raise ValueError("static cache capacity exceeded")
    for i, (key, value) in enumerate(saved):
        cache.keys[i][:, :, :length].copy_(key[:, :, :length])
        cache.values[i][:, :, :length].copy_(value[:, :, :length])
    cache.length = length


def _inputs(batch: torch.Tensor | dict[str, Any], key: str) -> torch.Tensor:
    x = batch[key] if isinstance(batch, dict) else batch
    if x.ndim != 2:
        raise ValueError("inputs must have shape [batch, time]")
    return x.long()


@torch.compiler.disable
def _byte_patches(raw: torch.Tensor, min_len: int, max_len: int, block_size: int):
    # Delimiter control flow runs on the observed bytes, outside compiled tensor graphs.
    return patch_batch(raw.detach().cpu().numpy(), min_len, max_len, block_size)


@dataclass
class BPEState:
    cache: KVCache | BatchedKVCache
    max_len: int


class BPELM(nn.Module):
    """Input tokens include BOS; each output predicts the following token."""
    def __init__(self, config: BPEConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.trunk.width)
        self.trunk = Transformer(config.trunk)
        self.apply(_initialize)

    @property
    def output_weight(self) -> torch.Tensor:
        return self.embedding.weight

    def hidden(self, batch: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        tokens = _inputs(batch, "tokens")
        hidden, _ = self.trunk(self.embedding(tokens))
        return hidden

    def forward(self, batch: torch.Tensor | dict[str, Any], *, return_hidden: bool = False) -> torch.Tensor:
        hidden = self.hidden(batch)
        return hidden if return_hidden else F.linear(hidden, self.output_weight)

    @torch.no_grad()
    def init_state(self, batch_size: int, max_len: int, device: torch.device | str, dtype: torch.dtype) -> BPEState:
        return BPEState(self.trunk.cache(batch_size, max_len, device, dtype), max_len)

    @torch.no_grad()
    def prefill(self, state: BPEState, inputs: torch.Tensor, *, last_only: bool = True,
                lengths: list[int] | torch.Tensor | None = None) -> torch.Tensor:
        """Prefill right-padded rows; device length tensors are trusted, avoiding a host read."""
        tokens = _inputs(inputs, "tokens")
        b, t = tokens.shape
        cache = state.cache
        fresh = not cache.length if isinstance(cache, KVCache) else not any(cache.lengths)
        if not fresh or not t or b != cache.keys[0].shape[0]:
            raise ValueError("prefill requires a nonempty prefix and a fresh, sufficiently large state")
        if lengths is None:
            if t > state.max_len or not isinstance(cache, KVCache):
                raise ValueError("prefill requires a nonempty prefix and a fresh, sufficiently large state")
            hidden, saved = self.trunk(self.embedding(tokens), capture=True)
            _copy_cache(cache, saved, t)
            return F.linear(hidden[:, -1:] if last_only else hidden, self.embedding.weight)
        if isinstance(lengths, torch.Tensor):
            if lengths.shape != (b,) or lengths.dtype != torch.long or lengths.device != tokens.device:
                raise ValueError("invalid token lengths")
            size_tensor = lengths
            sizes = None  # The trusted device tensor is read only when host decode needs it.
        else:
            sizes = list(lengths)
            if len(sizes) != b or any(size <= 0 or size > min(t, state.max_len) for size in sizes):
                raise ValueError("invalid token lengths")
            size_tensor = torch.tensor(sizes, device=tokens.device, dtype=torch.long)
        hidden, saved = self.trunk(self.embedding(tokens), capture=True)
        if isinstance(cache, KVCache):
            cache = BatchedKVCache(cache.keys, cache.values, [0] * b)
            state.cache = cache
        copy_length = min(t, state.max_len)
        # As in ByteLatentLM._clear_cache_storage: ragged masked attention reads the slots beyond a shorter row's
        # length, and init_state allocates with empty(), whose contents can be NaN (a masked NaN still poisons the
        # attention output). Found by the CUDA decode gate on 2026-11-11. A single row never attends past its own
        # length, so batch 1 (the frozen prefill timing path) is left exactly as it was.
        if b > 1:
            for key, value in zip(cache.keys, cache.values):
                key.zero_()
                value.zero_()
        for layer, (key, value) in enumerate(saved):
            cache.keys[layer][:, :, :copy_length].copy_(key[:, :, :copy_length])
            cache.values[layer][:, :, :copy_length].copy_(value[:, :, :copy_length])
        if sizes is None:
            def load_lengths() -> None:
                cache.lengths = size_tensor.detach().cpu().tolist()
            cache._lengths_loader = load_lengths
        else:
            cache.lengths = sizes
        if last_only:
            rows = torch.arange(b, device=tokens.device)
            hidden = hidden[rows, size_tensor - 1].unsqueeze(1)
        return F.linear(hidden, self.embedding.weight)

    @torch.no_grad()
    def step(self, state: BPEState, next_input: torch.Tensor) -> torch.Tensor:
        tokens = next_input.long().reshape(-1, 1)
        if isinstance(state.cache, BatchedKVCache):
            rows = list(range(len(state.cache)))
            hidden = self.trunk.step_batched(self.embedding(tokens), state.cache, rows, list(state.cache.lengths))
        else:
            hidden = self.trunk.step(self.embedding(tokens), state.cache)
        return F.linear(hidden, self.embedding.weight)


@dataclass
class ByteState:
    encoder: BatchedKVCache
    decoder: BatchedKVCache
    globals: BatchedKVCache
    _patchers: list[IncrementalPatcher] | None
    _hashes: list[RollingHashes] | None
    _conditioning: torch.Tensor | None
    _logits: torch.Tensor | None
    max_len: int
    position: int = 0
    _initializer: Callable[[ByteState], None] | None = None
    _routing_config: tuple[int, int, int, int, int] | None = None
    _routing: tuple[torch.Tensor, list[int] | None, int, bool] | None = None

    def _ensure_host_routing(self) -> None:
        if self._patchers is None:
            assert self._routing_config is not None
            minimum, maximum, block, rows, seed = self._routing_config
            self._patchers = [IncrementalPatcher(minimum, maximum, block) for _ in range(len(self.globals))]
            self._hashes = [RollingHashes(rows, seed) for _ in range(len(self.globals))]

    def _materialize_routing(self) -> None:
        if self._routing is None:
            return
        self._ensure_host_routing()
        assert self._patchers is not None and self._hashes is not None
        summary, sizes, block_size, have_hashes = self._routing
        self._routing = None
        rows = summary.cpu().tolist()  # One transfer, only when host decode/state inspection needs it.
        if sizes is None:
            sizes = [row.pop() for row in rows]
            self.position = max(sizes, default=0)
        counts = []
        for patcher, hashes, row, size in zip(self._patchers, self._hashes, rows, sizes):
            count, last = row[:2]
            counts.append(count)
            patcher.position, patcher.count, patcher.length = size, count, size - last - 1
            if have_hashes:
                hashes.history.clear()
                hashes.history.extend(row[2 + max(0, 8 - size):])
        self.globals.lengths = [count + 1 for count in counts]
        self.encoder.lengths = [((size - 1) % block_size + 1) if size else 0 for size in sizes]
        self.decoder.lengths = [size % block_size + 1 for size in sizes]

    @property
    def patchers(self) -> list[IncrementalPatcher]:
        self._materialize_routing()
        self._ensure_host_routing()
        assert self._patchers is not None
        return self._patchers

    @property
    def hashes(self) -> list[RollingHashes]:
        self._materialize_routing()
        self._ensure_host_routing()
        assert self._hashes is not None
        return self._hashes

    def _ensure_initialized(self) -> None:
        if self._logits is None:
            assert self._initializer is not None
            self._initializer(self)

    @property
    def conditioning(self) -> torch.Tensor:
        self._ensure_initialized()
        assert self._conditioning is not None
        return self._conditioning

    @conditioning.setter
    def conditioning(self, value: torch.Tensor) -> None:
        self._conditioning = value

    @property
    def logits(self) -> torch.Tensor:
        self._ensure_initialized()
        assert self._logits is not None
        return self._logits

    @logits.setter
    def logits(self, value: torch.Tensor) -> None:
        self._logits = value


class ByteLatentLM(nn.Module):
    """Inputs are raw bytes; output at t predicts byte t from bytes strictly before t.

    A state consumes raw bytes. Empty prefill returns the first-byte prediction;
    nonempty prefill and step return the prediction after the consumed prefix.
    """
    def __init__(self, config: ByteConfig) -> None:
        super().__init__()
        if config.min_len is None:
            raise ValueError("calibrate min_len before constructing a byte model")
        self.config = config
        c = config
        local_base = dict(width=c.local_width, q_heads=c.local_heads, kv_heads=c.local_heads,
                          head_dim=c.local_width // c.local_heads, ffn=c.local_ffn,
                          rope_theta=c.trunk.rope_theta, norm_eps=c.trunk.norm_eps, max_positions=c.block_size)
        self.byte_embedding = nn.Embedding(c.byte_vocab_size, c.local_width)
        self.local_bos = nn.Parameter(torch.zeros(c.local_width))
        self.global_bos = nn.Parameter(torch.zeros(c.trunk.width))
        self.encoder = Transformer(TransformerConfig(layers=c.enc_layers, **local_base))
        self.decoder = Transformer(TransformerConfig(layers=c.dec_layers, **local_base))
        self.pool = nn.Linear(c.local_width, c.trunk.width, bias=False)
        self.global_trunk = Transformer(c.trunk)
        self.condition = nn.Linear(c.trunk.width, c.local_width, bias=False)
        self.head = nn.Linear(c.local_width, c.output_classes, bias=False)
        self.hash_embedding = nn.Embedding(len(NGRAMS) * c.hash_rows, c.trunk.width) if c.hash_enabled else None
        self.hash_parameters = hash_parameters(c.hash_seed)
        prefixes, powers, ngrams, offsets, coefficients = hash_tables(self.hash_parameters)
        self.register_buffer("hash_prefixes", prefixes, persistent=False)
        self.register_buffer("hash_powers", powers, persistent=False)
        self.register_buffer("hash_ngrams", ngrams, persistent=False)
        self.register_buffer("hash_window_offsets", offsets, persistent=False)
        self.register_buffer("hash_coefficients", coefficients, persistent=False)
        self.register_buffer("hash_embedding_offsets", torch.arange(len(NGRAMS)) * c.hash_rows, persistent=False)
        self.apply(_initialize)
        nn.init.normal_(self.local_bos, std=0.02)
        nn.init.normal_(self.global_bos, std=0.02)
        if c.pool_init == "balanced":
            # Rescale rather than redraw, so every other weight matches the "normal" model with the same seed.
            with torch.no_grad():
                self.pool.weight.div_(math.sqrt(c.local_width))
        # Zeros draw no random numbers, so every other weight and the initial function match the model without it.
        self.global_readout = (nn.Parameter(torch.zeros(c.output_classes, c.trunk.width))
                               if c.global_readout else None)

    def _local(self, x: torch.Tensor, module: Transformer, capture: bool = False) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        b, t, d = x.shape
        block = self.config.block_size
        padded = ((t + block - 1) // block) * block
        x = F.pad(x, (0, 0, 0, padded - t)).reshape(b * (padded // block), block, d)
        y, saved = module(x, capture=capture)
        return y.reshape(b, padded, d)[:, :t], saved

    def _hash_vectors(self, ids: torch.Tensor) -> torch.Tensor:
        assert self.hash_embedding is not None
        return self.hash_embedding(ids + self.hash_embedding_offsets).sum(dim=-2) / math.sqrt(len(NGRAMS))

    def _encode_global(self, raw: torch.Tensor, ends: torch.Tensor, capture: bool = False,
                       hash_ids: torch.Tensor | None = None):
        b, _ = raw.shape
        c = self.config
        enc, enc_kv = self._local(self.byte_embedding(raw), self.encoder, capture)
        selected = enc.gather(1, ends.clamp_min(0).unsqueeze(-1).expand(-1, -1, c.local_width))
        x = self.pool(selected)
        if self.hash_embedding is not None:
            if hash_ids is None:
                hash_ids = hashes_at_ends(raw, ends, c.hash_rows, c.hash_seed,
                                          tables=(self.hash_prefixes, self.hash_powers, self.hash_ngrams,
                                                  self.hash_window_offsets, self.hash_coefficients))
            x = x + self._hash_vectors(hash_ids)
        x = x.masked_fill((ends < 0).unsqueeze(-1), 0)
        x = torch.cat((self.global_bos.expand(b, 1, -1), x), dim=1)
        global_states, global_kv = self.global_trunk(x, capture=capture)
        return enc, enc_kv, global_states, global_kv

    def _full_hidden(self, raw: torch.Tensor, capture: bool = False,
                     metadata: dict[str, torch.Tensor] | None = None, *,
                     hash_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
        b, t = raw.shape
        if t == 0:
            raise ValueError("use empty prefill for the initial byte prediction")
        c = self.config
        if metadata is None:
            if bool(((raw < 0) | (raw > 255)).any()):
                raise ValueError("byte inputs must be in [0, 255]")
            patches = _byte_patches(raw, c.min_len, c.max_len, c.block_size)
            ends = torch.as_tensor(patches.ends, device=raw.device, dtype=torch.long)
            ids = torch.as_tensor(patches.patch_ids, device=raw.device, dtype=torch.long)
        else:
            # The input pipeline validates bytes and supplies fixed-size device tensors.
            patches = metadata
            ends, ids = metadata["ends"], metadata["patch_ids"]
        enc, enc_kv, global_states, global_kv = self._encode_global(raw, ends, capture, hash_ids)
        # patch_ids[t] counts boundaries strictly before t; the last end at t is excluded.
        cond = global_states.gather(1, ids.unsqueeze(-1).expand(-1, -1, c.trunk.width))
        previous = torch.cat((self.local_bos.expand(b, 1, -1), enc[:, :-1]), dim=1)
        dec, dec_kv = self._local(previous + self.condition(cond), self.decoder, capture)
        info = {"enc": enc, "enc_kv": enc_kv, "dec_kv": dec_kv, "global_kv": global_kv,
                "global_states": global_states, "patches": patches, "cond": cond}
        return dec, info

    @property
    def output_weight(self) -> torch.Tensor:
        if self.global_readout is None:
            return self.head.weight
        # Concatenated so head(dec) + readout(cond) is one linear map of hidden() and chunked losses need no change.
        return torch.cat((self.head.weight, self.global_readout), dim=1)

    def hidden(self, batch: torch.Tensor | dict[str, Any]) -> torch.Tensor:
        raw = _inputs(batch, "bytes")
        metadata = batch.get("patches") if isinstance(batch, dict) else None
        hidden, info = self._full_hidden(raw, metadata=metadata)
        return hidden if self.global_readout is None else torch.cat((hidden, info["cond"]), dim=-1)

    def forward(self, batch: torch.Tensor | dict[str, Any], *, return_hidden: bool = False) -> torch.Tensor:
        hidden = self.hidden(batch)
        if return_hidden:
            return hidden
        return self.head(hidden) if self.global_readout is None else F.linear(hidden, self.output_weight)

    @torch.no_grad()
    def init_state(self, batch_size: int, max_len: int, device: torch.device | str, dtype: torch.dtype) -> ByteState:
        c = self.config
        if batch_size <= 0 or max_len <= 0:
            raise ValueError("batch size and maximum byte length must be positive")
        if self.global_readout is not None:
            # Every cached path (prefill, step, static decode, benchmarks) starts here.
            raise NotImplementedError("cached inference does not yet apply the global readout")
        encoder = self.encoder.batched_cache(batch_size, c.block_size, device, dtype)
        decoder = self.decoder.batched_cache(batch_size, c.block_size, device, dtype)
        assert c.min_len is not None
        capacity = 1 + max_patch_count(max_len, c.min_len, c.block_size)
        globals_ = self.global_trunk.batched_cache(batch_size, capacity, device, dtype)
        return ByteState(encoder, decoder, globals_, None, None, None, None, max_len,
                         _initializer=self._initialize_state,
                         _routing_config=(c.min_len, c.max_len, c.block_size, c.hash_rows, c.hash_seed))

    @torch.no_grad()
    def _initialize_state(self, state: ByteState) -> None:
        """Lazily produce the empty-prefix distribution for direct step users."""
        self._clear_cache_storage(state)
        batch_size = len(state.globals)
        rows = list(range(batch_size))
        device = state.globals.keys[0].device
        bos = self.global_bos.to(device=device).view(1, 1, -1).expand(batch_size, -1, -1)
        cond = self.condition(self.global_trunk.step_batched(bos, state.globals, rows, [0] * batch_size))
        first = self.decoder.step_batched(self.local_bos.to(device=device).view(1, 1, -1).expand(batch_size, -1, -1) + cond,
                                          state.decoder, rows, [0] * batch_size)
        state.conditioning, state.logits = cond, self.head(first)

    @staticmethod
    def _clear_cache_storage(state: ByteState) -> None:
        # Ragged masked attention may inspect slots beyond a shorter row's
        # logical length; initialise them before attention, not in init_state.
        for cache in (state.encoder, state.decoder, state.globals):
            for key, value in zip(cache.keys, cache.values):
                key.zero_()
                value.zero_()

    def _device_patch_metadata(self, raw: torch.Tensor, sizes: torch.Tensor) -> dict[str, torch.Tensor]:
        """Stage A: fixed-shape device patching and hashes, without host reads."""
        c = self.config
        patched = patch_batch_torch(raw, c.min_len, c.max_len, c.block_size, lengths=sizes)
        metadata = {"ends": patched.ends, "patch_ids": patched.patch_ids, "counts": patched.counts}
        if self.hash_embedding is not None:
            metadata["hash_ids"] = hashes_at_ends(
                raw, patched.ends, c.hash_rows, c.hash_seed,
                tables=(self.hash_prefixes, self.hash_powers, self.hash_ngrams,
                        self.hash_window_offsets, self.hash_coefficients))
        return metadata

    @torch.no_grad()
    def device_patch_metadata(self, source: torch.Tensor, *, lengths: list[int] | torch.Tensor | None = None
                              ) -> dict[str, torch.Tensor]:
        """Prepare upper-bound metadata on the input device for a later prefill.

        This stage is graph-capturable and never reads tensor values on the host.
        The returned metadata may be trimmed after reading ``counts.max()``.
        Device length tensors must already contain valid lengths.
        """
        if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
            raise ValueError("device patching requires nonempty [batch, time] inputs")
        if source.device != self.global_bos.device:
            raise ValueError("device patching inputs must be on the model device")
        b, t = source.shape
        raw = source.to(dtype=torch.long)
        if isinstance(lengths, torch.Tensor):
            if lengths.shape != (b,) or lengths.dtype != torch.long or lengths.device != raw.device:
                raise ValueError("invalid byte lengths")
            size_tensor = lengths
            padded = True
        else:
            sizes = [t] * b if lengths is None else list(lengths)
            if len(sizes) != b or any(size < 0 or size > t for size in sizes):
                raise ValueError("invalid byte lengths")
            size_tensor = (torch.full((b,), t, device=raw.device, dtype=torch.long) if lengths is None else
                           torch.tensor(sizes, device=raw.device, dtype=torch.long))
            padded = any(size != t for size in sizes)
        if padded:
            raw = torch.where(torch.arange(t, device=raw.device)[None, :] < size_tensor[:, None], raw, 0)
        if source.dtype != torch.uint8:
            invalid = ((raw < 0) | (raw > 255)).any()
            if raw.device.type == "cuda":
                torch._assert_async(~invalid, "byte inputs must be in [0, 255]")
            elif bool(invalid):
                raise ValueError("byte inputs must be in [0, 255]")
        return self._device_patch_metadata(raw, size_tensor)

    @torch.no_grad()
    def prefill(self, state: ByteState, inputs: torch.Tensor | dict[str, Any], *, last_only: bool = True,
                lengths: list[int] | torch.Tensor | None = None, patching: str = "host",
                patch_capacity: str = "bucket64") -> torch.Tensor:
        """Fill caches from byte rows; device length tensors and preprocessed metadata are trusted."""
        source = inputs["bytes"] if isinstance(inputs, dict) else inputs
        if source.ndim != 2:
            raise ValueError("inputs must have shape [batch, time]")
        if patching not in {"host", "device", "preprocessed"}:
            raise ValueError("patching must be host, device or preprocessed")
        if patch_capacity not in {"exact", "bucket64", "upper_bound"}:
            raise ValueError("patch_capacity must be exact, bucket64 or upper_bound")
        host = source.detach().cpu() if patching == "host" else None
        raw = source.to(device=state.globals.keys[0].device, dtype=torch.long)
        b, t = raw.shape
        if state.position != 0 or b != len(state.globals):
            raise ValueError("prefill requires a fresh, sufficiently large state")
        if t == 0:
            if lengths is not None:
                sizes = lengths.detach().cpu().tolist() if isinstance(lengths, torch.Tensor) else list(lengths)
                if len(sizes) != b or any(size != 0 for size in sizes):
                    raise ValueError("prefill requires a fresh, sufficiently large state")
            return state.logits
        c = self.config
        if isinstance(lengths, torch.Tensor):
            if (patching == "host" or lengths.shape != (b,) or lengths.dtype != torch.long or
                    lengths.device != raw.device):
                raise ValueError("invalid byte lengths")
            size_tensor, sizes, size_array = lengths, None, None
            padded = True
        else:
            sizes = [t] * b if lengths is None else list(lengths)
            size_array = np.asarray(sizes, dtype=np.int64)
            if (len(sizes) != b or np.any(size_array < 0) or
                    np.any(size_array > min(t, state.max_len))):
                raise ValueError("prefill requires a fresh, sufficiently large state")
            size_tensor = (torch.full((b,), t, device=raw.device, dtype=torch.long) if lengths is None else
                           torch.tensor(sizes, device=raw.device, dtype=torch.long))
            padded = bool(np.any(size_array != t))
        if padded:
            valid = torch.arange(t, device=raw.device)[None, :] < size_tensor[:, None]
            raw = torch.where(valid, raw, 0)
        if patching == "host":
            assert host is not None and size_array is not None and sizes is not None
            host_array = np.where(np.arange(t)[None, :] < size_array[:, None], host.numpy(), 0)
            patched = patch_batch(host_array, c.min_len, c.max_len, c.block_size, lengths=size_array)
            counts_list = patched.counts.tolist()
            counts = torch.as_tensor(patched.counts, device=raw.device, dtype=torch.long)
            ends = torch.as_tensor(patched.ends, device=raw.device, dtype=torch.long)
            ids = torch.as_tensor(patched.patch_ids, device=raw.device, dtype=torch.long)
        elif patching == "device":
            if source.dtype != torch.uint8:
                invalid = ((raw < 0) | (raw > 255)).any()
                if raw.device.type == "cuda":
                    torch._assert_async(~invalid, "byte inputs must be in [0, 255]")
                elif bool(invalid):
                    raise ValueError("byte inputs must be in [0, 255]")
            patched = self._device_patch_metadata(raw, size_tensor)
            if patch_capacity != "upper_bound":
                count = int(patched["counts"].max().item())
                capacity = choose_patch_capacity(count, patched["ends"].shape[1], patch_capacity)
                patched = trim_patch_metadata(patched, capacity)
            ends, ids, counts = patched["ends"], patched["patch_ids"], patched["counts"]
            host_array = None
        else:
            if not isinstance(inputs, dict) or "patches" not in inputs:
                raise ValueError("preprocessed prefill requires patch metadata")
            patched = inputs["patches"]
            ends, ids, counts = patched["ends"], patched["patch_ids"], patched["counts"]
            capacity = ends.shape[1] if ends.ndim == 2 else -1
            if (ends.ndim != 2 or ends.shape[0] != b or capacity > max_patch_count(t, c.min_len, c.block_size) or
                    ids.shape != (b, t) or counts.shape != (b,) or
                    any(value.device != raw.device or value.dtype != torch.long for value in (ends, ids, counts))):
                raise ValueError("invalid preprocessed patch metadata")
            hash_ids = patched.get("hash_ids")
            if ((self.hash_embedding is not None and hash_ids is None) or
                    (hash_ids is not None and (hash_ids.shape != (b, capacity, len(NGRAMS)) or
                                               hash_ids.device != raw.device or hash_ids.dtype != torch.long))):
                raise ValueError("invalid preprocessed hash metadata")
            host_array = None
        metadata = {"ends": ends, "patch_ids": ids}
        if patching in {"device", "preprocessed"} and self.hash_embedding is not None:
            metadata["hash_ids"] = patched["hash_ids"]
        # A zero-length ragged prefill can leave deferred routing on a fresh
        # state. A subsequent nonempty prefill replaces that routing entirely.
        state._routing = None
        for cache in (state.encoder, state.decoder, state.globals):
            cache._lengths_loader = None
        self._clear_cache_storage(state)
        if last_only:
            hidden = None
            enc, enc_kv, global_states, global_kv = self._encode_global(
                raw, ends, capture=True, hash_ids=metadata.get("hash_ids"))
        else:
            hidden, info = self._full_hidden(raw, capture=True, metadata=metadata, hash_ids=metadata.get("hash_ids"))
            enc, enc_kv = info["enc"], info["enc_kv"]
            global_states, global_kv = info["global_states"], info["global_kv"]
        for layer, (key, value) in enumerate(global_kv):
            # A padded input can be wider than max_len even when every valid
            # row fits. Only padded, causally irrelevant slots can exceed it.
            copy_length = min(key.shape[2], state.globals.keys[layer].shape[2])
            state.globals.keys[layer][:, :, :copy_length].copy_(key[:, :, :copy_length])
            state.globals.values[layer][:, :, :copy_length].copy_(value[:, :, :copy_length])
        # The host path already has the routing metadata. The device path defers
        # all per-row host work until a caller actually requests host decode state.
        if host_array is not None:
            state.globals.lengths = [count + 1 for count in counts_list]
            routing = [[int(patched.ends[row, count - 1]) if count else -1,
                        *([0] * max(0, 8 - size)), *host_array[row, max(0, size - 8):size].tolist()]
                       for row, (size, count) in enumerate(zip(sizes, counts_list))]
            for row, (size, count) in enumerate(zip(sizes, counts_list)):
                patcher = state.patchers[row]
                patcher.position, patcher.count = size, count
                patcher.length = size - routing[row][0] - 1
                if self.hash_embedding is not None:
                    state.hashes[row].history.extend(routing[row][1 + max(0, 8 - size):])
        # Only the final encoder block can be read by a future local step.
        blocks = (t + c.block_size - 1) // c.block_size
        block_indices = ((size_tensor - 1).clamp_min(0) // c.block_size).clamp_max(blocks - 1)
        rows_tensor = torch.arange(b, device=raw.device)
        for layer, (key, value) in enumerate(enc_kv):
            state.encoder.keys[layer].copy_(key.reshape(b, blocks, *key.shape[1:])[rows_tensor, block_indices])
            state.encoder.values[layer].copy_(value.reshape(b, blocks, *value.shape[1:])[rows_tensor, block_indices])
        tails = None if size_array is None else size_array % c.block_size
        tail_tensor = size_tensor.remainder(c.block_size)
        # A captured graph must handle every tail length in this byte bucket.
        max_tail = c.block_size - 1 if tails is None else int(tails.max(initial=0))
        if max_tail:
            absolute = (size_tensor - size_tensor.remainder(c.block_size))[:, None] + torch.arange(max_tail, device=raw.device)
            previous_index = (absolute - 1).clamp(min=0, max=t - 1)
            previous = enc.gather(1, previous_index.unsqueeze(-1).expand(-1, -1, c.local_width))
            previous = torch.where((absolute == 0).unsqueeze(-1), self.local_bos.view(1, 1, -1), previous)
            patch_ids = ids.gather(1, absolute.clamp_max(t - 1))
            cond = global_states.gather(1, patch_ids.unsqueeze(-1).expand(-1, -1, c.trunk.width))
            _, dec_kv = self.decoder(previous + self.condition(cond), capture=True)
            for layer, (key, value) in enumerate(dec_kv):
                state.decoder.keys[layer][:, :, :max_tail].copy_(key)
                state.decoder.values[layer][:, :, :max_tail].copy_(value)
        final_global = global_states[rows_tensor, counts].unsqueeze(1)
        conditioning = self.condition(final_global)
        final_enc = enc[rows_tensor, (size_tensor - 1).clamp_min(0)].unsqueeze(1)
        previous = torch.where((size_tensor == 0)[:, None, None], self.local_bos.view(1, 1, -1), final_enc)
        following = self.decoder.step_batched_tensor(previous + conditioning, state.decoder,
                                                     rows_tensor, tail_tensor, max_tail + 1)
        state.position = t if sizes is None else max(sizes, default=0)
        state.conditioning, state.logits = conditioning, self.head(following)
        if host_array is None:
            offsets = size_tensor[:, None] - 8 + torch.arange(8, device=raw.device)
            tail = raw.gather(1, offsets.clamp(min=0, max=t - 1))
            tail = torch.where(offsets >= 0, tail, 0)
            if ends.shape[1]:
                last = ends.gather(1, (counts - 1).clamp_min(0)[:, None])
                last = torch.where(counts[:, None] > 0, last, -1)
            else:
                last = torch.full((b, 1), -1, device=raw.device, dtype=torch.long)
            summary = torch.cat((counts[:, None], last, tail), dim=1)
            if sizes is None:
                summary = torch.cat((summary, size_tensor[:, None]), dim=1)
            state._routing = (summary, sizes, c.block_size, self.hash_embedding is not None)
            for cache in (state.encoder, state.decoder, state.globals):
                cache._lengths_loader = state._materialize_routing
        else:
            assert sizes is not None and tails is not None
            state.encoder.lengths = [((size - 1) % c.block_size + 1) if size else 0 for size in sizes]
            state.decoder.lengths = (tails + 1).tolist()
        if last_only:
            return state.logits
        assert hidden is not None
        return torch.cat((self.head(hidden), state.logits), dim=1)

    @torch.no_grad()
    def step(self, state: ByteState, next_input: torch.Tensor) -> torch.Tensor:
        state._ensure_initialized()
        raw = next_input.long().reshape(-1)
        c = self.config
        if raw.numel() != len(state.globals) or any(p.position >= state.max_len for p in state.patchers):
            raise ValueError("batch size mismatch or byte cache exhausted")
        values = raw.tolist()
        if any(not 0 <= value < 256 for value in values):
            raise ValueError("expected raw bytes")
        rows = list(range(len(state.globals)))
        enc_positions = [patcher.position % c.block_size for patcher in state.patchers]
        enc = self.encoder.step_batched(self.byte_embedding(raw[:, None]), state.encoder, rows, enc_positions)
        closed_rows, closed_ids = [], []
        for i, value in enumerate(values):
            ids = state.hashes[i].step(value) if self.hash_embedding is not None else None
            if state.patchers[i].step(value):
                closed_rows.append(i)
                if ids is not None:
                    closed_ids.append(ids)
        if closed_rows:
            selected = torch.tensor(closed_rows, dtype=torch.long, device=raw.device)
            x = self.pool(enc.index_select(0, selected))
            if self.hash_embedding is not None:
                x = x + self._hash_vectors(torch.tensor(closed_ids, dtype=torch.long, device=raw.device).unsqueeze(1))
            positions = [state.globals.lengths[i] for i in closed_rows]
            conditioning = self.condition(self.global_trunk.step_batched(x, state.globals, closed_rows, positions))
            state.conditioning.index_copy_(0, selected, conditioning)
        state.position = max(patcher.position for patcher in state.patchers)
        dec_positions = [patcher.position % c.block_size for patcher in state.patchers]
        dec = self.decoder.step_batched(enc + state.conditioning, state.decoder, rows, dec_positions)
        state.logits = self.head(dec)
        return state.logits
