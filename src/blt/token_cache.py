from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from threading import Event
from typing import Any

import numpy as np

from .data import WindowIndex, sample_window_index, sha256_file, tokenize_window, verify_supply


def build_token_cache(supply: str | Path, tokenizer_path: str | Path, tokenizer: Any, *, window_bytes: int = 8192,
                      batch_size: int = 128, sample_size: int = 256, sample_seed: int = 0) -> Path:
    if min(window_bytes, batch_size, sample_size) <= 0:
        raise ValueError("window, batch and verification sample sizes must be positive")
    supply, tokenizer_path = Path(supply), Path(tokenizer_path)
    manifest = verify_supply(supply)
    manifest_hash, tokenizer_hash = sha256_file(supply / "manifest.json"), sha256_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    if not 1 <= vocab_size <= 65536:
        raise ValueError("uint16 cache requires a vocabulary of at most 65536 tokens")
    root = supply / "tokenized"
    root.mkdir(parents=True, exist_ok=True)
    output = root / tokenizer_hash
    if output.exists():
        raise FileExistsError(f"token cache already exists: {output}")
    with WindowIndex(supply / "train.bin", window_bytes) as windows:
        sample_count = min(sample_size, len(windows))
        sample = {sample_window_index(sample_seed, i, 0, 1, len(windows)) for i in range(sample_count)}
        with tempfile.TemporaryDirectory(prefix=".token-cache-", dir=root) as temporary:
            work = Path(temporary)
            offsets = np.lib.format.open_memmap(work / "offsets.npy", mode="w+", dtype="<i8", shape=(len(windows) + 1,))
            offsets[0] = 0
            total_tokens, total_bytes, max_tokens = 0, 0, 0
            with open(work / "tokens.bin", "wb") as binary:
                for start in range(0, len(windows), batch_size):
                    group = [windows[i].data for i in range(start, min(start + batch_size, len(windows)))]
                    encodings = tokenizer.encode_batch([raw.decode("utf-8") for raw in group], add_special_tokens=False)
                    if len(encodings) != len(group):
                        raise ValueError("tokenizer returned the wrong batch size")
                    for position, (raw, encoding) in enumerate(zip(group, encodings), start):
                        ids = encoding.ids
                        if not ids or min(ids) < 0 or max(ids) >= vocab_size or max(ids) > 65535:
                            raise ValueError("tokenizer returned invalid token IDs")
                        if position in sample and tokenizer.decode(ids, skip_special_tokens=False).encode("utf-8") != raw:
                            raise ValueError(f"tokenizer failed the sampled byte round-trip at window {position}")
                        binary.write(np.asarray(ids, dtype="<u2").tobytes())
                        total_tokens += len(ids)
                        total_bytes += len(raw)
                        max_tokens = max(max_tokens, len(ids))
                        offsets[position + 1] = total_tokens
                binary.flush()
                os.fsync(binary.fileno())
            offsets.flush()
            del offsets
            with open(work / "offsets.npy", "rb") as handle:
                os.fsync(handle.fileno())
            meta = {"version": 1, "tokenizer_sha256": tokenizer_hash, "manifest_sha256": manifest_hash,
                    "train_sha256": manifest["files"]["train.bin"]["sha256"], "window_bytes": window_bytes,
                    "drop_short_tail": windows.drop_short_tail, "window_index_sha256": windows.index_sha256,
                    "window_count": len(windows), "total_tokens": total_tokens, "total_bytes": total_bytes,
                    "bytes_per_token": total_bytes / total_tokens, "max_tokens": max_tokens, "vocab_size": vocab_size,
                    "sample_seed": sample_seed, "verified_windows": sample_count,
                    "tokens_sha256": sha256_file(work / "tokens.bin"), "offsets_sha256": sha256_file(work / "offsets.npy")}
            (work / "meta.json").write_text(json.dumps(meta, sort_keys=True, indent=2) + "\n", encoding="utf-8")
            os.rename(work, output)
    return output


class TokenCache:
    def __init__(self, path: str | Path, windows: WindowIndex, manifest_hash: str, tokenizer_hash: str,
                 vocab_size: int, *, verify_hashes: bool = True, stop: Event | None = None) -> None:
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text(encoding="utf-8"))
        expected = {"version": 1, "manifest_sha256": manifest_hash, "tokenizer_sha256": tokenizer_hash,
                    "window_bytes": windows.window_bytes, "drop_short_tail": windows.drop_short_tail,
                    "window_index_sha256": windows.index_sha256, "window_count": len(windows), "vocab_size": vocab_size}
        if any(self.meta.get(key) != value for key, value in expected.items()):
            raise ValueError("token cache does not match the tokenizer, supply, or window layout")
        for name, key in (("tokens.bin", "tokens_sha256"), ("offsets.npy", "offsets_sha256")):
            if verify_hashes and sha256_file(self.path / name, stop) != self.meta.get(key):
                raise ValueError(f"token cache checksum mismatch: {name}")
        self.offsets = np.load(self.path / "offsets.npy", mmap_mode="r", allow_pickle=False)
        if self.offsets.shape != (len(windows) + 1,) or self.offsets.dtype != np.dtype("<i8"):
            raise ValueError("invalid token offsets")
        diffs = np.diff(self.offsets)
        if (self.offsets[0] != 0 or np.any(diffs <= 0) or self.offsets[-1] != self.meta["total_tokens"]
                or int(diffs.max()) != self.meta["max_tokens"]
                or (self.path / "tokens.bin").stat().st_size != 2 * self.meta["total_tokens"]):
            raise ValueError("invalid token cache offsets or size")
        self.tokens = np.memmap(self.path / "tokens.bin", mode="r", dtype="<u2")
        self.max_tokens = int(self.meta["max_tokens"])

    def __getitem__(self, index: int) -> np.ndarray:
        return self.tokens[int(self.offsets[index]):int(self.offsets[index + 1])]

    def __len__(self) -> int:
        return len(self.offsets) - 1


class MemoryTokenCache:
    """Compatibility for direct, in-memory callers; the training CLI requires a disk cache."""
    def __init__(self, windows: WindowIndex, tokenizer: Any) -> None:
        self.tokens = [np.asarray(tokenize_window(tokenizer, windows[i].data), dtype=np.int64) for i in range(len(windows))]
        self.max_tokens = max(map(len, self.tokens))

    def __getitem__(self, index: int) -> np.ndarray:
        return self.tokens[index]

    def __len__(self) -> int:
        return len(self.tokens)
