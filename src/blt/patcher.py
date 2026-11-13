from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import string
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F


_DELIMITERS = np.zeros(256, dtype=np.bool_)
_DELIMITERS[list((string.whitespace + string.punctuation).encode("ascii"))] = True


@dataclass(frozen=True)
class PatchBatch:
    ends: np.ndarray  # [batch, max_patches], -1 padding; only completed patches
    patch_ids: np.ndarray  # [batch, time], including the current incomplete patch
    counts: np.ndarray
    boundaries: np.ndarray  # [batch, time], true after this observed byte


@dataclass(frozen=True)
class TorchPatchBatch:
    ends: torch.Tensor
    patch_ids: torch.Tensor
    counts: torch.Tensor
    boundaries: torch.Tensor


def patch_capacity(count: int, upper_bound: int, policy: str) -> int:
    """Choose a sufficient patch capacity without inspecting a tensor."""
    if not 0 <= count <= upper_bound:
        raise ValueError("patch count exceeds upper bound")
    if policy == "exact":
        return count
    if policy == "bucket64":
        return min(((count + 63) // 64) * 64, upper_bound)
    if policy == "upper_bound":
        return upper_bound
    raise ValueError("patch capacity must be exact, bucket64 or upper_bound")


def trim_patch_metadata(metadata: dict[str, torch.Tensor], capacity: int) -> dict[str, torch.Tensor]:
    """Return views of the patch-indexed metadata at a known sufficient capacity.

    The caller is responsible for ensuring that capacity covers every count.
    No tensor values are read on the host by this function.
    """
    ends = metadata["ends"]
    if ends.ndim != 2 or not 0 <= capacity <= ends.shape[1]:
        raise ValueError("invalid patch capacity")
    result = dict(metadata)
    result["ends"] = ends[:, :capacity]
    if "hash_ids" in result:
        hashes = result["hash_ids"]
        if hashes.ndim != 3 or hashes.shape[:2] != ends.shape:
            raise ValueError("hash metadata does not match patch capacity")
        result["hash_ids"] = hashes[:, :capacity]
    return result


def max_patch_count(time: int, min_len: int, block_size: int = 64) -> int:
    if time < 0 or not 1 <= min_len <= block_size:
        raise ValueError("invalid patch capacity")
    full, tail = divmod(time, block_size)
    return full * ((block_size + min_len - 1) // min_len) + (tail + min_len - 1) // min_len


class IncrementalPatcher:
    def __init__(self, min_len: int, max_len: int = 16, block_size: int = 64) -> None:
        if not 1 <= min_len <= max_len <= block_size:
            raise ValueError("invalid patch lengths")
        self.min_len, self.max_len, self.block_size = min_len, max_len, block_size
        self.length = 0
        self.position = 0
        self.count = 0

    def step(self, byte: int) -> bool:
        if not 0 <= byte <= 255:
            raise ValueError("expected a byte")
        self.length += 1
        close = bool((_DELIMITERS[byte] and self.length >= self.min_len) or self.length == self.max_len
                     or (self.position + 1) % self.block_size == 0)
        self.position += 1
        if close:
            self.length = 0
            self.count += 1
        return close


def patch_batch(data: np.ndarray, min_len: int, max_len: int = 16, block_size: int = 64, *,
                lengths: np.ndarray | None = None, max_patches: int | None = None) -> PatchBatch:
    """Scan all independent local blocks in parallel; incomplete tails stay open."""
    array = np.asarray(data)
    if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer) or np.any(array < 0) or np.any(array > 255):
        raise ValueError("data must be a [batch, time] byte array")
    if not 1 <= min_len <= max_len <= block_size:
        raise ValueError("invalid patch lengths")
    b, t = array.shape
    sizes = np.full(b, t, dtype=np.int64) if lengths is None else np.asarray(lengths, dtype=np.int64)
    if sizes.shape != (b,) or np.any(sizes < 0) or np.any(sizes > t):
        raise ValueError("invalid byte lengths")
    padded = ((t + block_size - 1) // block_size) * block_size
    values = np.pad(array, ((0, 0), (0, padded - t))).reshape(-1, block_size)
    valid = (np.arange(padded)[None, :] < sizes[:, None]).reshape(-1, block_size)
    block_boundaries = np.zeros_like(valid)
    current = np.zeros(values.shape[0], dtype=np.int32)
    for i in range(block_size):
        current += valid[:, i]
        closed = valid[:, i] & ((_DELIMITERS[values[:, i]] & (current >= min_len)) | (current == max_len))
        if i == block_size - 1:
            closed |= valid[:, i]
        block_boundaries[:, i] = closed
        current[closed] = 0
    boundaries = block_boundaries.reshape(b, padded)[:, :t]
    counts = boundaries.sum(axis=1, dtype=np.int64)
    ids = np.cumsum(boundaries, axis=1, dtype=np.int64) - boundaries
    capacity = int(counts.max(initial=0)) if max_patches is None else max_patches
    if capacity < int(counts.max(initial=0)):
        raise ValueError("patch capacity is too small")
    ends = np.full((b, capacity), -1, dtype=np.int64)
    rows, cols = np.nonzero(boundaries)
    if len(rows):
        ends[rows, ids[rows, cols]] = cols
    return PatchBatch(ends, ids, counts, boundaries)


def patch_batch_torch(data: torch.Tensor, min_len: int, max_len: int = 16, block_size: int = 64, *,
                      lengths: torch.Tensor | None = None, max_patches: int | None = None) -> TorchPatchBatch:
    """Device-only scan; ends use a static upper bound, padded with -1."""
    if data.ndim != 2 or data.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("data must be a [batch, time] integer tensor")
    if not 1 <= min_len <= max_len <= block_size:
        raise ValueError("invalid patch lengths")
    b, t = data.shape
    capacity = max_patch_count(t, min_len, block_size) if max_patches is None else max_patches
    if capacity < max_patch_count(t, min_len, block_size):
        raise ValueError("static patch capacity is too small")
    if lengths is not None and lengths.device != data.device:
        raise ValueError("byte lengths must already be on the input device")
    sizes = torch.full((b,), t, dtype=torch.long, device=data.device) if lengths is None else lengths.to(dtype=torch.long)
    if sizes.shape != (b,):
        raise ValueError("invalid byte length shape")
    padded = ((t + block_size - 1) // block_size) * block_size
    if t == 0:
        return TorchPatchBatch(torch.full((b, capacity), -1, dtype=torch.long, device=data.device),
                               torch.empty((b, 0), dtype=torch.long, device=data.device),
                               torch.zeros(b, dtype=torch.long, device=data.device),
                               torch.empty((b, 0), dtype=torch.bool, device=data.device))
    values = F.pad(data, (0, padded - t)).reshape(-1, block_size)
    positions = torch.arange(padded, device=data.device)
    valid = ((positions[None, :] < sizes[:, None]) & (positions[None, :] < t)).reshape(-1, block_size)
    byte = values.long()
    delimiter = (((byte >= 9) & (byte <= 13)) | ((byte >= 32) & (byte <= 47)) |
                 ((byte >= 58) & (byte <= 64)) | ((byte >= 91) & (byte <= 96)) |
                 ((byte >= 123) & (byte <= 126)))
    # For each byte, map every possible incoming patch length to its outgoing
    # length. Padding is the identity; each full local block forces a closure.
    states = torch.arange(max_len, device=data.device, dtype=torch.long)
    advanced = states + 1
    block_end = torch.arange(block_size, device=data.device) == block_size - 1
    closed = ((delimiter.unsqueeze(-1) & (advanced >= min_len)) | (advanced == max_len) |
              block_end[None, :, None])
    transitions = torch.where(valid.unsqueeze(-1), torch.where(closed, 0, advanced), states)
    # Inclusive parallel prefix of function composition, six gathers for a
    # 64-byte block. Each local block starts in state zero.
    stride = 1
    while stride < block_size:
        composed = transitions[:, stride:, :].gather(2, transitions[:, :-stride, :])
        transitions = torch.cat((transitions[:, :stride, :], composed), dim=1)
        stride *= 2
    boundaries = (valid & (transitions[:, :, 0] == 0)).reshape(b, padded)[:, :t]
    boundary_ids = boundaries.long()
    counts = boundary_ids.sum(dim=1)
    ids = boundary_ids.cumsum(dim=1) - boundary_ids
    indices = torch.where(boundaries, ids, capacity)
    values = torch.where(boundaries, torch.arange(t, device=data.device).expand(b, -1), -1)
    ends = torch.full((b, capacity + 1), -1, dtype=torch.long, device=data.device)
    ends.scatter_(1, indices, values)
    return TorchPatchBatch(ends[:, :capacity], ids, counts, boundaries)


@dataclass(frozen=True)
class Calibration:
    min_len: int
    mean: float
    histogram: dict[int, int]


def calibrate_min_len(byte_samples: Iterable[bytes], target_mean: float, max_len: int = 16, block_size: int = 64,
                      stopped: Callable[[], bool] | None = None) -> Calibration:
    samples = list(byte_samples)
    if target_mean <= 0 or not samples or not any(samples):
        raise ValueError("calibration requires nonempty samples and a positive target")
    sizes = np.asarray([len(sample) for sample in samples], dtype=np.int64)
    values = np.zeros((len(samples), int(sizes.max())), dtype=np.uint8)
    for row, sample in enumerate(samples):
        values[row, :len(sample)] = np.frombuffer(sample, dtype=np.uint8)
    candidates = []
    for minimum in range(1, max_len + 1):
        if stopped is not None and stopped():
            raise InterruptedError("patch calibration interrupted")
        histogram: Counter[int] = Counter()
        patches = patch_batch(values, minimum, max_len, block_size, lengths=sizes)
        for row, size in enumerate(sizes):
            ends = patches.ends[row, :patches.counts[row]]
            previous = -1
            for end in ends:
                histogram[int(end) - previous] += 1
                previous = int(end)
            if previous + 1 < size:
                histogram[int(size) - previous - 1] += 1
        mean = sum(length * count for length, count in histogram.items()) / sum(histogram.values())
        candidates.append(Calibration(minimum, mean, dict(sorted(histogram.items()))))
    return min(candidates, key=lambda item: (abs(item.mean - target_mean), item.min_len))
