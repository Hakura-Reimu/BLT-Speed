from __future__ import annotations

from collections import deque
from functools import lru_cache
import hashlib
from typing import Iterable

import torch


NGRAMS = tuple(range(3, 9))
_PRIME = 2**31 - 1


@lru_cache(maxsize=256)
def _parameters(n: int, seed: int) -> tuple[int, int]:
    def derive(label: str) -> int:
        digest = hashlib.blake2s(f"{seed}:{n}:{label}".encode("ascii")).digest()
        return int.from_bytes(digest[:4], "little")
    return 257 + derive("base") % (_PRIME - 258), derive("salt") % _PRIME


def salt(n: int, seed: int) -> int:
    return _parameters(n, seed)[1]


def hash_parameters(seed: int) -> tuple[tuple[int, int], ...]:
    return tuple(_parameters(n, seed) for n in NGRAMS)


@lru_cache(maxsize=256)
def _hash_table_values(parameters: tuple[tuple[int, int], ...]) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    return (tuple(prefix for _, prefix in parameters),
            tuple(tuple(pow(base, power, _PRIME) for power in range(9)) for base, _ in parameters))


def hash_tables(parameters: tuple[tuple[int, int], ...], *, device: torch.device | str | None = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Constants for evaluating all six independent hashes in parallel."""
    prefixes, powers = _hash_table_values(parameters)
    return (torch.tensor(prefixes, dtype=torch.long, device=device),
            torch.tensor(powers, dtype=torch.long, device=device),
            torch.tensor(NGRAMS, dtype=torch.long, device=device),
            torch.tensor(tuple(range(7, -1, -1)), dtype=torch.long, device=device),
            torch.tensor(tuple(tuple(reversed(row[:8])) for row in powers), dtype=torch.long, device=device))


def hash_history(history: Iterable[int], n: int, rows: int = 8192, seed: int = 0) -> int:
    values = list(history)[-n:]
    if not 3 <= n <= 8 or rows <= 0 or any(not 0 <= value < 256 for value in values):
        raise ValueError("invalid hash input")
    # Prefix the actual history length so a short context is deterministic and distinct.
    base, prefix = _parameters(n, seed)
    result = (prefix + len(values)) % _PRIME
    for value in values:
        result = (result * base + value + 1) % _PRIME
    return result % rows


class RollingHashes:
    def __init__(self, rows: int = 8192, seed: int = 0) -> None:
        if rows <= 0:
            raise ValueError("hash table size must be positive")
        self.rows, self.seed = rows, seed
        self.history: deque[int] = deque(maxlen=8)

    def step(self, byte: int) -> list[int]:
        if not 0 <= byte < 256:
            raise ValueError("expected a byte")
        self.history.append(byte)
        return [hash_history(self.history, n, self.rows, self.seed) for n in NGRAMS]


def hashes_at_ends(data: torch.Tensor, ends: torch.Tensor, rows: int = 8192, seed: int = 0, *,
                   parameters: tuple[tuple[int, int], ...] | None = None,
                   tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
                   ) -> torch.Tensor:
    """Hash each selected suffix without reading before its context start."""
    if data.ndim != 2 or ends.ndim != 2 or data.shape[0] != ends.shape[0] or rows <= 0:
        raise ValueError("invalid hash tensor shapes")
    if tables is None:
        tables = hash_tables(hash_parameters(seed) if parameters is None else parameters, device=data.device)
    prefixes, powers, ngrams, offsets, coefficients = tables
    # Every selected byte is read once. The final axis is oldest to newest.
    positions = ends.unsqueeze(-1) - offsets
    indices = positions.clamp(min=0, max=max(data.shape[1] - 1, 0))
    window = data.gather(1, indices.flatten(1)).reshape(*ends.shape, 8).long()
    lengths = (ends.unsqueeze(-1) + 1).clamp_min(0).minimum(ngrams)
    start_power = powers.expand(*ends.shape, -1, -1).gather(-1, lengths.unsqueeze(-1)).squeeze(-1)
    start = ((prefixes + lengths) * start_power).remainder(_PRIME)
    valid = (positions.unsqueeze(-2) >= 0) & (offsets < ngrams.unsqueeze(-1))
    contribution = torch.where(valid, (window.unsqueeze(-2) + 1) * coefficients, 0).sum(dim=-1)
    return (start + contribution).remainder(_PRIME).remainder(rows)
