"""Fixed-address decode state, CUDA graphs, and greedy speculation.

The existing model state remains the reference and is used for prefill.  The
classes here copy that state into fixed-size tensors before incremental decode.
Graph objects are deliberately process-local and are never checkpointed.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
import torch.nn.functional as F

from .hashes import NGRAMS, hash_parameters
from .model import BPELM, BPEState, BatchedKVCache, ByteLatentLM, ByteState, Transformer, _rotate


_PRIME = 2**31 - 1


def _delimiter(byte: torch.Tensor) -> torch.Tensor:
    return (((byte >= 9) & (byte <= 13)) | ((byte >= 32) & (byte <= 47)) |
            ((byte >= 58) & (byte <= 64)) | ((byte >= 91) & (byte <= 96)) |
            ((byte >= 123) & (byte <= 126)))


@dataclass
class DeviceRouting:
    """Per-row patch and suffix state, updated without reading device values."""

    positions: torch.Tensor
    patch_lengths: torch.Tensor
    patch_counts: torch.Tensor
    history: torch.Tensor  # right-aligned, oldest to newest
    history_lengths: torch.Tensor
    min_len: int
    max_len: int
    block_size: int
    hash_rows: int
    parameters: tuple[tuple[int, int], ...]

    @classmethod
    def from_reference(cls, state: ByteState, model: ByteLatentLM) -> DeviceRouting:
        device = state.conditioning.device
        c = model.config
        history = torch.zeros((len(state.patchers), 8), dtype=torch.long, device=device)
        lengths = []
        for row, hashes in enumerate(state.hashes):
            values = list(hashes.history)
            lengths.append(len(values))
            if values:
                history[row, 8 - len(values):].copy_(torch.tensor(values, dtype=torch.long, device=device))
        assert c.min_len is not None
        return cls(torch.tensor([p.position for p in state.patchers], dtype=torch.long, device=device),
                   torch.tensor([p.length for p in state.patchers], dtype=torch.long, device=device),
                   torch.tensor([p.count for p in state.patchers], dtype=torch.long, device=device),
                   history, torch.tensor(lengths, dtype=torch.long, device=device),
                   c.min_len, c.max_len, c.block_size, c.hash_rows, hash_parameters(c.hash_seed))

    def tensors(self) -> list[torch.Tensor]:
        return [self.positions, self.patch_lengths, self.patch_counts, self.history, self.history_lengths]

    def step(self, byte: torch.Tensor, *, need_hashes: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Consume one already validated byte per row and return closure and hashes."""
        length = self.patch_lengths + 1
        closed = ((_delimiter(byte) & (length >= self.min_len)) | (length == self.max_len) |
                  ((self.positions + 1).remainder(self.block_size) == 0))
        self.patch_lengths.copy_(torch.where(closed, 0, length))
        self.patch_counts.add_(closed.long())
        self.positions.add_(1)
        if not need_hashes:
            return closed, None
        self.history.copy_(torch.cat((self.history[:, 1:], byte[:, None]), dim=1))
        self.history_lengths.add_(1).clamp_(max=8)
        result = []
        for n, (base, prefix) in zip(NGRAMS, self.parameters):
            value = (prefix + self.history_lengths.clamp(max=n)).remainder(_PRIME)
            for offset in range(n - 1, -1, -1):
                candidate = (value * base + self.history[:, 7 - offset] + 1).remainder(_PRIME)
                value = torch.where(self.history_lengths > offset, candidate, value)
            result.append(value.remainder(self.hash_rows))
        return closed, torch.stack(result, dim=-1)


@dataclass
class StaticCache:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]

    @classmethod
    def from_reference(cls, cache: Any) -> StaticCache:
        return cls([key.clone() for key in cache.keys], [value.clone() for value in cache.values])

    def tensors(self) -> list[torch.Tensor]:
        return [*self.keys, *self.values]

    def clear_unused(self, lengths: list[int]) -> None:
        # Masked attention can still propagate NaNs from uninitialised KV
        # values, so initialise each row's unwritten tail before replay.
        for key, value in zip(self.keys, self.values):
            for row, length in enumerate(lengths):
                key[row, :, length:].zero_()
                value[row, :, length:].zero_()


def _transformer_step(module: Transformer, x: torch.Tensor, cache: StaticCache,
                      positions: torch.Tensor, commit: torch.Tensor | None = None) -> torch.Tensor:
    """One static-shape step with device positions and optionally masked KV writes."""
    b = x.shape[0]
    capacity = cache.keys[0].shape[2]
    # Non-closing rows can have a full global cache; they must still be safe to evaluate.
    address = positions.clamp(max=capacity - 1)
    rotary_positions = positions.clamp(max=module.config.max_positions - 1)
    cos = sin = None
    for layer, block in enumerate(module.blocks):
        c = block.attn.config
        normed = block.attn_norm(x)
        q = block.attn.q(normed).view(b, 1, c.q_heads, c.head_dim).transpose(1, 2)
        k = block.attn.k(normed).view(b, 1, c.kv_heads, c.head_dim).transpose(1, 2)
        v = block.attn.v(normed).view(b, 1, c.kv_heads, c.head_dim).transpose(1, 2)
        if cos is None:
            cos_table, sin_table = module.rotary.tables(q.dtype)
            cos = cos_table[rotary_positions][:, None, None, :]
            sin = sin_table[rotary_positions][:, None, None, :]
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        index = address[:, None, None, None].expand(b, c.kv_heads, 1, c.head_dim)
        keys, values = cache.keys[layer], cache.values[layer]
        if commit is not None:
            mask = commit[:, None, None, None]
            k = torch.where(mask, k, keys.gather(2, index))
            v = torch.where(mask, v, values.gather(2, index))
        keys.scatter_(2, index, k)
        values.scatter_(2, index, v)
        attention_keys, attention_values = keys, values
        if c.q_heads != c.kv_heads:
            repeats = c.q_heads // c.kv_heads
            attention_keys = keys.repeat_interleave(repeats, dim=1)
            attention_values = values.repeat_interleave(repeats, dim=1)
        mask = torch.arange(capacity, device=x.device)[None, None, None, :] <= positions[:, None, None, None]
        y = F.scaled_dot_product_attention(q, attention_keys, attention_values, attn_mask=mask,
                                           dropout_p=0.0, is_causal=False)
        x = x + block.attn.out(y.transpose(1, 2).contiguous().view(b, 1, c.width))
        z = block.ffn_norm(x)
        x = x + block.down(F.silu(block.gate(z)) * block.up(z))
    return module.norm(x)


def _transformer_chunk(module: Transformer, x: torch.Tensor, cache: StaticCache,
                       starts: torch.Tensor, valid_counts: list[int] | None = None) -> torch.Tensor:
    """Teacher-force a causal chunk, optionally committing a ragged prefix per row."""
    batch, length, _ = x.shape
    capacity = cache.keys[0].shape[2]
    positions = starts[:, None] + torch.arange(length, device=x.device)[None, :]
    # Unused ragged suffixes may extend beyond the configured table.
    rotary_positions = positions.clamp(max=module.config.max_positions - 1)
    cos = sin = None
    if valid_counts is not None:
        if len(valid_counts) != batch or any(count < 0 or count > length for count in valid_counts):
            raise ValueError("invalid ragged verification lengths")
        row_list = [row for row, count in enumerate(valid_counts) for _ in range(count)]
        offset_list = [offset for count in valid_counts for offset in range(count)]
        rows = torch.tensor(row_list, device=x.device, dtype=torch.long)
        offsets = torch.tensor(offset_list, device=x.device, dtype=torch.long)
        destination = positions[rows, offsets]
    for layer, block in enumerate(module.blocks):
        c = block.attn.config
        normed = block.attn_norm(x)
        q = block.attn.q(normed).view(batch, length, c.q_heads, c.head_dim).transpose(1, 2)
        k = block.attn.k(normed).view(batch, length, c.kv_heads, c.head_dim).transpose(1, 2)
        v = block.attn.v(normed).view(batch, length, c.kv_heads, c.head_dim).transpose(1, 2)
        if cos is None:
            cos_table, sin_table = module.rotary.tables(q.dtype)
            cos = cos_table[rotary_positions][:, None, :, :]
            sin = sin_table[rotary_positions][:, None, :, :]
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        keys, values = cache.keys[layer], cache.values[layer]
        if valid_counts is None:
            index = positions[:, None, :, None].expand(batch, c.kv_heads, length, c.head_dim)
            keys.scatter_(2, index, k)
            values.scatter_(2, index, v)
        else:
            keys[rows, :, destination, :] = k[rows, :, offsets, :]
            values[rows, :, destination, :] = v[rows, :, offsets, :]
        attention_keys, attention_values = keys, values
        if c.q_heads != c.kv_heads:
            repeats = c.q_heads // c.kv_heads
            attention_keys = keys.repeat_interleave(repeats, dim=1)
            attention_values = values.repeat_interleave(repeats, dim=1)
        mask = torch.arange(capacity, device=x.device)[None, None, None, :] <= positions[:, None, :, None]
        y = F.scaled_dot_product_attention(q, attention_keys, attention_values, attn_mask=mask,
                                           dropout_p=0.0, is_causal=False)
        x = x + block.attn.out(y.transpose(1, 2).contiguous().view(batch, length, c.width))
        z = block.ffn_norm(x)
        x = x + block.down(F.silu(block.gate(z)) * block.up(z))
    return module.norm(x)


class _StaticState:
    """Snapshot and restore copy into the same addresses used by captured graphs."""

    max_len: int
    consumed: int

    def tensors(self) -> list[torch.Tensor]:
        raise NotImplementedError

    def snapshot(self) -> tuple[list[torch.Tensor], int]:
        return [tensor.clone() for tensor in self.tensors()], self.consumed

    def restore(self, snapshot: tuple[list[torch.Tensor], int]) -> None:
        values, consumed = snapshot
        tensors = self.tensors()
        if len(values) != len(tensors) or any(a.shape != b.shape for a, b in zip(tensors, values)):
            raise ValueError("snapshot does not match this static state")
        for target, source in zip(tensors, values):
            target.copy_(source)
        self.consumed = consumed

    def _advance(self) -> None:
        if self.consumed >= self.max_len:
            raise ValueError("decode cache capacity exceeded")
        self.consumed += 1

    def _check_token(self, token: torch.Tensor | None) -> None:
        if token is not None and (token.numel() != self.current.numel() or token.device != self.current.device or
                                  token.dtype not in (torch.int32, torch.int64)):
            raise ValueError("decode input must contain one integer per row on the state device")

    @torch.no_grad()
    def step(self, token: torch.Tensor | None = None) -> torch.Tensor:
        self._check_token(token)
        self._advance()
        self._step_tensor(self.current if token is None else token)
        return self.logits

    def _step_tensor(self, token: torch.Tensor) -> None:
        raise NotImplementedError


class StaticBPEState(_StaticState):
    def __init__(self, model: BPELM, state: BPEState, logits: torch.Tensor,
                 allowed: torch.Tensor | None = None) -> None:
        lengths = (list(state.cache.lengths) if isinstance(state.cache, BatchedKVCache)
                   else [state.cache.length] * logits.shape[0])
        self.model, self.max_len, self.consumed = model, state.max_len, max(lengths)
        self.cache = StaticCache.from_reference(state.cache)
        # Prefill may be right-padded and caches are allocated with empty().
        # Clear the tail of every row, not just the tail of the longest row.
        self.cache.clear_unused(lengths)
        self.positions = torch.tensor(lengths, dtype=torch.long, device=logits.device)
        self.allowed = allowed
        if allowed is not None and (allowed.shape != (model.config.vocab_size,) or allowed.device != logits.device or
                                    allowed.dtype != torch.bool):
            raise ValueError("allowed mask must be boolean and match the vocabulary and device")
        self.logits = logits.clone()
        self.current = self._select(self.logits)

    def _select(self, logits: torch.Tensor) -> torch.Tensor:
        scores = logits[:, -1, :]
        if self.allowed is not None:
            scores = scores.masked_fill(~self.allowed, -torch.inf)
        return scores.argmax(dim=-1)

    def tensors(self) -> list[torch.Tensor]:
        return [*self.cache.tensors(), self.positions, self.logits, self.current]

    def _step_tensor(self, token: torch.Tensor) -> None:
        hidden = _transformer_step(self.model.trunk, self.model.embedding(token.reshape(-1, 1)), self.cache, self.positions)
        self.positions.add_(1)
        self.logits.copy_(F.linear(hidden, self.model.embedding.weight))
        self.current.copy_(self._select(self.logits))

    @torch.no_grad()
    def step_chunk(self, tokens: torch.Tensor) -> torch.Tensor:
        if (tokens.ndim != 2 or tokens.shape[0] != self.current.shape[0] or tokens.shape[1] < 1 or
                self.consumed + tokens.shape[1] > self.max_len):
            raise ValueError("invalid BPE verification chunk or insufficient cache capacity")
        hidden = _transformer_chunk(self.model.trunk, self.model.embedding(tokens), self.cache, self.positions)
        logits = F.linear(hidden, self.model.embedding.weight)
        self.positions.add_(tokens.shape[1])
        self.consumed += tokens.shape[1]
        self.logits.copy_(logits[:, -1:])
        self.current.copy_(self._select(self.logits))
        return logits


class StaticByteState(_StaticState):
    def __init__(self, model: ByteLatentLM, state: ByteState) -> None:
        self.model, self.max_len = model, state.max_len
        self.consumed = max(p.position for p in state.patchers)
        self.encoder = StaticCache.from_reference(state.encoder)
        self.decoder = StaticCache.from_reference(state.decoder)
        self.globals = StaticCache.from_reference(state.globals)
        self.encoder.clear_unused(list(state.encoder.lengths))
        self.decoder.clear_unused(list(state.decoder.lengths))
        self.globals.clear_unused(list(state.globals.lengths))
        self.global_positions = torch.tensor(state.globals.lengths, dtype=torch.long, device=state.conditioning.device)
        self.routing = DeviceRouting.from_reference(state, model)
        self.conditioning = state.conditioning.clone()
        self.logits = state.logits.clone()
        self.current = self.logits[:, -1, :256].argmax(dim=-1)

    def tensors(self) -> list[torch.Tensor]:
        return [*self.encoder.tensors(), *self.decoder.tensors(), *self.globals.tensors(),
                self.global_positions, *self.routing.tensors(), self.conditioning, self.logits, self.current]

    def _speculation_tensors(self) -> list[torch.Tensor]:
        # Global KV entries are append-only; stale rejected suffixes are masked
        # by global_positions and overwritten on the next commit.
        return [*self.encoder.tensors(), *self.decoder.tensors(), self.global_positions,
                *self.routing.tensors(), self.conditioning, self.logits, self.current]

    def speculation_snapshot(self) -> tuple[list[torch.Tensor], int]:
        return [tensor.clone() for tensor in self._speculation_tensors()], self.consumed

    def restore_speculation(self, snapshot: tuple[list[torch.Tensor], int]) -> None:
        values, consumed = snapshot
        tensors = self._speculation_tensors()
        if len(values) != len(tensors) or any(a.shape != b.shape for a, b in zip(tensors, values)):
            raise ValueError("snapshot does not match this byte state")
        for target, source in zip(tensors, values):
            target.copy_(source)
        self.consumed = consumed

    def _encode_tensor(self, token: torch.Tensor):
        model, c = self.model, self.model.config
        byte = token.reshape(-1)
        enc = _transformer_step(model.encoder, model.byte_embedding(byte[:, None]), self.encoder,
                                self.routing.positions.remainder(c.block_size))
        closed, ids = self.routing.step(byte, need_hashes=model.hash_embedding is not None)
        return enc, closed, ids

    def _global_tensor(self, enc: torch.Tensor, closed: torch.Tensor, ids: torch.Tensor | None) -> None:
        model = self.model
        pooled = model.pool(enc)
        if ids is not None:
            pooled = pooled + model._hash_vectors(ids[:, None, :])
        global_hidden = _transformer_step(model.global_trunk, pooled, self.globals, self.global_positions, closed)
        proposed = model.condition(global_hidden)
        self.conditioning.copy_(torch.where(closed[:, None, None], proposed, self.conditioning))
        self.global_positions.add_(closed.long())

    def _decoder_tensor(self, enc: torch.Tensor) -> None:
        model, c = self.model, self.model.config
        dec = _transformer_step(model.decoder, enc + self.conditioning, self.decoder,
                                self.routing.positions.remainder(c.block_size))
        self.logits.copy_(model.head(dec))
        self.current.copy_(self.logits[:, -1, :256].argmax(dim=-1))

    def _step_tensor(self, token: torch.Tensor, *, stale_global: bool = False):
        enc, closed, ids = self._encode_tensor(token)
        if not stale_global:
            self._global_tensor(enc, closed, ids)
        self._decoder_tensor(enc)
        return enc, closed, ids

    @torch.no_grad()
    def conditional_step(self, token: torch.Tensor | None = None) -> torch.Tensor:
        """Run the global trunk only when at least one row closes a patch."""
        self._check_token(token)
        self._advance()
        enc, closed, ids = self._encode_tensor(self.current if token is None else token)
        if bool(closed.any().item()):
            self._global_tensor(enc, closed, ids)
        self._decoder_tensor(enc)
        return self.logits

    @torch.no_grad()
    def draft_step(self, token: torch.Tensor | None = None) -> torch.Tensor:
        self._check_token(token)
        self._advance()
        self._draft_trace = self._step_tensor(self.current if token is None else token, stale_global=True)
        return self.logits


class CUDAGraphStepper:
    """Capture one greedy step; recreate after process/checkpoint resume."""

    @torch.no_grad()
    def __init__(self, state: _StaticState, warmup: int = 3, *, byte_draft: bool = False) -> None:
        if state.current.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("CUDA graph decoding requires CUDA state")
        if warmup < 1 or state.consumed >= state.max_len:
            raise ValueError("graph capture needs warm-up and at least one free cache position")
        if byte_draft and not isinstance(state, StaticByteState):
            raise ValueError("a byte draft graph requires a byte state")
        self.state = state
        self.byte_draft = byte_draft
        initial = state.snapshot()
        device = state.current.device
        torch.cuda.synchronize(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            # Restore on each warm-up so even a nearly full cache can be captured.
            for _ in range(warmup):
                state.restore(initial)
                if byte_draft:
                    trace = state._step_tensor(state.current, stale_global=True)
                else:
                    state._step_tensor(state.current)
        stream.synchronize()
        if byte_draft:
            self.trace = tuple(torch.empty_like(value) if value is not None else None for value in trace)
        state.restore(initial)
        torch.cuda.synchronize(device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            if byte_draft:
                result = state._step_tensor(state.current, stale_global=True)
                for target, source in zip(self.trace, result):
                    if target is not None:
                        target.copy_(source)
            else:
                state._step_tensor(state.current)
        torch.cuda.synchronize(device)
        state.restore(initial)

    def step(self) -> torch.Tensor:
        self.state._advance()
        self.graph.replay()
        if self.byte_draft:
            self.state._draft_trace = self.trace
        return self.state.logits

    def reset(self, snapshot: tuple[list[torch.Tensor], int]) -> None:
        self.state.restore(snapshot)

    def __getstate__(self):
        raise TypeError("CUDA graphs are process-local; recreate and recapture after resume")


class CUDASplitGraphStepper:
    """Graph encoder/routing separately from the conditional global/decoder graphs.

    The branch reads one boolean from the device for each consumed byte. All
    three graphs and their shared intermediate buffers are process-local.
    """

    @torch.no_grad()
    def __init__(self, state: StaticByteState, warmup: int = 3) -> None:
        if state.current.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("split graph decoding requires CUDA state")
        if warmup < 1 or state.consumed >= state.max_len:
            raise ValueError("graph capture needs warm-up and at least one free cache position")
        self.state = state
        initial = state.snapshot()
        device = state.current.device
        torch.cuda.synchronize(device)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                state.restore(initial)
                enc, closed, ids = state._encode_tensor(state.current)
                state._global_tensor(enc, closed, ids)
                state._decoder_tensor(enc)
                state.restore(initial)
                enc, closed, ids = state._encode_tensor(state.current)
                state._decoder_tensor(enc)
        stream.synchronize()
        self.enc = torch.empty_like(enc)
        self.closed = torch.empty_like(closed)
        self.ids = torch.empty_like(ids) if ids is not None else None
        state.restore(initial)
        torch.cuda.synchronize(device)
        self.encoder_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.encoder_graph):
            enc, closed, ids = state._encode_tensor(state.current)
            self.enc.copy_(enc)
            self.closed.copy_(closed)
            if self.ids is not None:
                self.ids.copy_(ids)
        state.restore(initial)
        self.global_decoder_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.global_decoder_graph):
            state._global_tensor(self.enc, self.closed, self.ids)
            state._decoder_tensor(self.enc)
        state.restore(initial)
        self.decoder_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.decoder_graph):
            state._decoder_tensor(self.enc)
        torch.cuda.synchronize(device)
        state.restore(initial)

    def step(self) -> torch.Tensor:
        self.state._advance()
        self.encoder_graph.replay()
        if bool(self.closed.any().item()):
            self.global_decoder_graph.replay()
        else:
            self.decoder_graph.replay()
        return self.state.logits

    def reset(self, snapshot: tuple[list[torch.Tensor], int]) -> None:
        self.state.restore(snapshot)

    def __getstate__(self):
        raise TypeError("CUDA graphs are process-local; recreate and recapture after resume")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _common_prefix(matches: torch.Tensor) -> int:
    """Accept a common prefix across rows so a fixed-size batch advances together."""
    return int(matches.all(dim=0).long().cumprod(dim=0).sum().item())


@dataclass
class SpeculationResult:
    tokens: torch.Tensor  # [batch, emitted positions], including a correction if needed
    accepted_draft_positions: int
    draft_seconds: float
    verify_seconds: float
    rollback_seconds: float


@torch.inference_mode()
def speculate_byte(state: StaticByteState, k: int, *,
                   draft_graph: CUDAGraphStepper | None = None) -> SpeculationResult:
    """Draft with frozen conditioning; verify packed patches and corrected decoder."""
    if (k <= 0 or state.consumed + k > state.max_len or
            (draft_graph is not None and (draft_graph.state is not state or not draft_graph.byte_draft))):
        raise ValueError("invalid draft length or insufficient byte cache")
    device = state.current.device
    _synchronize(device)
    started = time.perf_counter()
    before = state.speculation_snapshot()
    start_positions = state.routing.positions.clone()
    initial_prediction = state.current.clone()
    initial_conditioning = state.conditioning.clone()
    decoder_offset = len(state.encoder.tensors())
    decoder_before = before[0][decoder_offset:decoder_offset + len(state.decoder.tensors())]
    draft = []
    encoded = []
    closures = []
    hashes = []
    for _ in range(k):
        draft.append(state.current.clone())
        if draft_graph is None:
            state.draft_step()
        else:
            draft_graph.step()
        enc, closed, ids = state._draft_trace
        encoded.append(enc.clone() if draft_graph is not None else enc)
        closures.append(closed.clone() if draft_graph is not None else closed)
        if ids is not None:
            hashes.append(ids.clone() if draft_graph is not None else ids)
    drafted = torch.stack(draft, dim=1)
    _synchronize(device)
    draft_seconds = time.perf_counter() - started

    started = time.perf_counter()
    model = state.model
    close_tensor = torch.stack(closures, dim=1)
    flags = close_tensor.cpu().tolist()
    counts = [sum(row) for row in flags]
    max_count = max(counts, default=0)
    if max_count:
        patch_indices = [[index for index, closed in enumerate(row) if closed] + [0] * (max_count - count)
                         for row, count in zip(flags, counts)]
        indices = torch.tensor(patch_indices, device=device, dtype=torch.long)
        encodings = torch.cat(encoded, dim=1)
        selected = encodings.gather(1, indices[:, :, None].expand(-1, -1, encodings.shape[-1]))
        global_input = model.pool(selected)
        if hashes:
            hash_ids = torch.stack(hashes, dim=1)
            selected_hashes = hash_ids.gather(1, indices[:, :, None].expand(-1, -1, hash_ids.shape[-1]))
            global_input = global_input + model._hash_vectors(selected_hashes)
        hidden = _transformer_chunk(model.global_trunk, global_input, state.globals,
                                    state.global_positions, valid_counts=counts)
        conditions = model.condition(hidden)
        state.global_positions.add_(torch.tensor(counts, device=device, dtype=torch.long))
        lookup = torch.cat((initial_conditioning, conditions), dim=1)
        condition_indices = close_tensor.long().cumsum(dim=1)
        conditioning = lookup.gather(1, condition_indices[:, :, None].expand(-1, -1, lookup.shape[-1]))
    else:
        conditioning = initial_conditioning.expand(-1, k, -1)
    for target, saved in zip(state.decoder.tensors(), decoder_before):
        target.copy_(saved)
    expected = []
    prediction = initial_prediction
    for index in range(k):
        expected.append(prediction)
        position = (start_positions + index + 1).remainder(model.config.block_size)
        dec = _transformer_step(model.decoder, encoded[index] + conditioning[:, index:index + 1],
                                state.decoder, position)
        logits = model.head(dec)
        prediction = logits[:, -1, :256].argmax(dim=-1)
    expected = torch.stack(expected, dim=1)
    state.conditioning.copy_(conditioning[:, -1:])
    state.logits.copy_(logits)
    state.current.copy_(prediction)
    accepted = _common_prefix(drafted == expected)
    _synchronize(device)
    verify_seconds = time.perf_counter() - started

    started = time.perf_counter()
    emitted = drafted[:, :accepted]
    if accepted < k:
        emitted = torch.cat((emitted, expected[:, accepted:accepted + 1]), dim=1)
        state.restore_speculation(before)
        for index in range(emitted.shape[1]):
            state.step(emitted[:, index])
    state._draft_trace = None
    _synchronize(device)
    rollback_seconds = time.perf_counter() - started
    return SpeculationResult(emitted, accepted, draft_seconds, verify_seconds, rollback_seconds)


@torch.inference_mode()
def speculate_bpe(target: StaticBPEState, drafter: StaticBPEState, k: int, *,
                  draft_graph: CUDAGraphStepper | None = None) -> SpeculationResult:
    """Verify greedy proposals; caller must ensure the two tokenizer mappings match."""
    if (k <= 0 or target is drafter or target.consumed != drafter.consumed or
            target.consumed + k > target.max_len or drafter.consumed + k > drafter.max_len or
            target.current.shape != drafter.current.shape or target.current.device != drafter.current.device or
            target.model.config.vocab_size != drafter.model.config.vocab_size or
            target.model.config.bos_id != drafter.model.config.bos_id or
            (draft_graph is not None and draft_graph.state is not drafter)):
        raise ValueError("incompatible BPE states, draft length, or cache capacity")
    device = target.current.device
    _synchronize(device)
    started = time.perf_counter()
    target_start = target.positions.clone()
    draft_start = drafter.positions.clone()
    target_consumed, draft_consumed = target.consumed, drafter.consumed
    initial_prediction = target.current.clone()
    draft = []
    for _ in range(k):
        draft.append(drafter.current.clone())
        if draft_graph is None:
            drafter.step()
        else:
            draft_graph.step()
    drafted = torch.stack(draft, dim=1)
    _synchronize(device)
    draft_seconds = time.perf_counter() - started

    started = time.perf_counter()
    verified_logits = target.step_chunk(drafted)
    scores = verified_logits if target.allowed is None else verified_logits.masked_fill(~target.allowed, -torch.inf)
    subsequent = scores.argmax(dim=-1)
    expected = torch.cat((initial_prediction[:, None], subsequent[:, :-1]), dim=1)
    accepted = _common_prefix(drafted == expected)
    _synchronize(device)
    verify_seconds = time.perf_counter() - started

    started = time.perf_counter()
    emitted = drafted[:, :accepted]
    if accepted < k:
        correction = expected[:, accepted]
        emitted = torch.cat((emitted, correction[:, None]), dim=1)
        # KV entries beyond the logical lengths are masked and overwritten by
        # future steps.  Accepted entries are already the right prefix.
        target.positions.copy_(target_start + accepted)
        drafter.positions.copy_(draft_start + accepted)
        target.consumed = target_consumed + accepted
        drafter.consumed = draft_consumed + accepted
        target.step(correction)
        drafter.step(correction)
    _synchronize(device)
    rollback_seconds = time.perf_counter() - started
    return SpeculationResult(emitted, accepted, draft_seconds, verify_seconds, rollback_seconds)
