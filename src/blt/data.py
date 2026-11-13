from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import mmap
import os
from pathlib import Path
import sqlite3
import struct
import tempfile
from threading import Event
from typing import Iterable, Iterator


SEPARATOR = b"\n"
WINDOW_BYTES = 8192


def sha256_file(path: str | Path, stop: Event | None = None) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            if stop is not None and stop.is_set():
                raise InterruptedError("data verification interrupted")
            digest.update(block)
    return digest.hexdigest()


def document_rank(content_hash: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{content_hash}".encode("ascii")).hexdigest()


def split_documents(documents: Iterable[bytes], seed: int, val_bytes: int, test_bytes: int,
                    train_bytes: int) -> dict[str, list[bytes]]:
    """Group identical documents and allocate ranked groups to disjoint splits."""
    unique = {hashlib.sha256(doc).hexdigest(): doc for doc in documents if doc}
    result: dict[str, list[bytes]] = {"train": [], "val": [], "test": []}
    ordered = sorted(unique.items(), key=lambda item: (document_rank(item[0], seed), item[0]))
    totals = {key: 0 for key in result}
    for _, doc in ordered:
        split = "val" if totals["val"] < val_bytes else "test" if totals["test"] < test_bytes else "train"
        if split == "train" and totals["train"] >= train_bytes:
            continue
        result[split].append(doc)
        totals[split] += len(doc)
    return result


def _parquet_texts(path: Path) -> Iterator[str]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if "text" not in parquet.schema_arrow.names:
        raise ValueError(f"{path} has no text column")
    for batch in parquet.iter_batches(columns=["text"], batch_size=1024):
        for text in batch.column(0).to_pylist():
            if text:
                if not isinstance(text, str):
                    raise ValueError(f"non-text document in {path}")
                yield text


def prepare_supply(source_files: Iterable[str | Path], output: str | Path, *, seed: int = 0,
                   train_bytes: int = 10_000_000_000, val_bytes: int = 16 * 1024 * 1024,
                   test_bytes: int = 16 * 1024 * 1024) -> dict:
    """Write a new supply from local parquet files; never overwrite a prior supply."""
    sources = sorted({Path(path).resolve() for path in source_files})
    output = Path(output)
    if not sources or any(not path.is_file() for path in sources):
        raise ValueError("supply requires existing parquet files")
    if min(train_bytes, val_bytes, test_bytes) <= 0:
        raise ValueError("split byte targets must be positive")
    if output.exists():
        raise FileExistsError(f"immutable supply already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_records = [{"path": str(path), "sha256": sha256_file(path)} for path in sources]
    with tempfile.TemporaryDirectory(prefix=".blt-supply-", dir=output.parent) as temporary:
        work = Path(temporary)
        database = sqlite3.connect(work / "catalog.sqlite")
        database.execute("CREATE TABLE docs (digest TEXT PRIMARY KEY, rank TEXT NOT NULL, size INTEGER NOT NULL, split TEXT, written INTEGER NOT NULL DEFAULT 0)")
        total_documents = 0
        for source in sources:
            for text in _parquet_texts(source):
                raw = text.encode("utf-8")
                digest = hashlib.sha256(raw).hexdigest()
                database.execute("INSERT OR IGNORE INTO docs(digest,rank,size) VALUES(?,?,?)",
                                 (digest, document_rank(digest, seed), len(raw)))
                total_documents += 1
            database.commit()
        unique_documents = database.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        totals = {"val": 0, "test": 0, "train": 0}
        targets = {"val": val_bytes, "test": test_bytes, "train": train_bytes}
        for digest, size in database.execute("SELECT digest,size FROM docs ORDER BY rank,digest"):
            split = next((name for name in ("val", "test", "train") if totals[name] < targets[name]), None)
            if split is None:
                break
            database.execute("UPDATE docs SET split=? WHERE digest=?", (split, digest))
            totals[split] += size
        if any(totals[name] < targets[name] for name in targets):
            raise ValueError(f"not enough unique source bytes for targets: {totals}")
        database.commit()
        counts = {name: 0 for name in totals}
        files = {name: (open(work / f"{name}.bin", "wb"), open(work / f"{name}.idx", "w", encoding="utf-8", newline="\n")) for name in totals}
        try:
            for source in sources:
                for text in _parquet_texts(source):
                    raw = text.encode("utf-8")
                    digest = hashlib.sha256(raw).hexdigest()
                    row = database.execute("SELECT split,written FROM docs WHERE digest=?", (digest,)).fetchone()
                    if row is None or row[0] is None or row[1]:
                        continue
                    split = row[0]
                    binary, index = files[split]
                    if counts[split]:
                        binary.write(SEPARATOR)
                    start = binary.tell()
                    binary.write(raw)
                    record = {"start": start, "end": binary.tell(), "sha256": digest,
                              "separator_before": len(SEPARATOR) if counts[split] else 0}
                    index.write(json.dumps(record, sort_keys=True) + "\n")
                    counts[split] += 1
                    database.execute("UPDATE docs SET written=1 WHERE digest=?", (digest,))
            database.commit()
            missing = database.execute("SELECT COUNT(*) FROM docs WHERE split IS NOT NULL AND written=0").fetchone()[0]
            if missing:
                raise ValueError(f"{missing} selected documents disappeared while reading the sources")
        finally:
            for binary, index in files.values():
                binary.close()
                index.close()
            database.close()
        if any(sha256_file(path) != record["sha256"] for path, record in zip(sources, source_records)):
            raise ValueError("a source parquet file changed while preparing the supply")
        manifest = {"version": 1, "seed": seed, "separator_hex": SEPARATOR.hex(),
                    "source_files": source_records,
                    "source_documents": total_documents, "unique_documents": unique_documents,
                    "selected_documents": sum(counts.values()), "duplicate_documents": total_documents - unique_documents,
                    "targets": targets, "document_bytes": totals, "document_counts": counts,
                    "files": {f"{name}.{ext}": {"sha256": sha256_file(work / f"{name}.{ext}"),
                                                     "size": (work / f"{name}.{ext}").stat().st_size}
                              for name in totals for ext in ("bin", "idx")}}
        (work / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (work / "catalog.sqlite").unlink()
        # Publish only a completed directory. Refuse an existing destination above.
        os.rename(work, output)
    return manifest


def verify_supply(supply: str | Path, stop: Event | None = None) -> dict:
    root = Path(supply)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = {f"{name}.{ext}" for name in ("train", "val", "test") for ext in ("bin", "idx")}
    if manifest.get("version") != 1 or set(manifest.get("files", {})) != expected:
        raise ValueError("invalid data manifest")
    for name, recorded in manifest["files"].items():
        path = root / name
        if path.stat().st_size != recorded["size"] or sha256_file(path, stop) != recorded["sha256"]:
            raise ValueError(f"immutable supply file changed: {path}")
    return manifest


def utf8_end(data: bytes | mmap.mmap, start: int, maximum: int, end: int | None = None) -> int:
    """Find the largest UTF-8 boundary at or before start + maximum."""
    limit = len(data) if end is None else end
    if not 0 <= start <= limit <= len(data) or maximum < 1:
        raise ValueError("invalid UTF-8 window bounds")
    stop = min(start + maximum, limit)
    while stop < limit and stop > start and data[stop] & 0xC0 == 0x80:
        stop -= 1
    if stop == start and stop < limit:
        raise ValueError("window is too small for one UTF-8 code point")
    return stop


@dataclass(frozen=True)
class ByteWindow:
    data: bytes
    start: int
    end: int
    document: str | None = None


class WindowIndex:
    def __init__(self, path: str | Path, window_bytes: int = WINDOW_BYTES, *, drop_short_tail: bool = True) -> None:
        self.path = Path(path)
        self.window_bytes, self.drop_short_tail = window_bytes, drop_short_tail
        self.handle = open(self.path, "rb")
        if not os.fstat(self.handle.fileno()).st_size:
            self.handle.close()
            raise ValueError("empty training stream")
        self.data = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        self.windows: list[tuple[int, int]] = []
        digest = hashlib.sha256()
        start = 0
        while start < len(self.data):
            end = utf8_end(self.data, start, window_bytes)
            if not (drop_short_tail and end == len(self.data) and end - start < window_bytes - 3):
                self.windows.append((start, end))
                digest.update(struct.pack("<QQ", start, end))
            start = end
        self.index_sha256 = digest.hexdigest()
        if not self.windows:
            self.close()
            raise ValueError("no complete training windows")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> ByteWindow:
        start, end = self.windows[index]
        return ByteWindow(self.data[start:end], start, end)

    def close(self) -> None:
        self.data.close()
        self.handle.close()

    def __enter__(self) -> WindowIndex:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def iter_eval_chunks(supply: str | Path, split: str, window_bytes: int = WINDOW_BYTES) -> Iterator[ByteWindow]:
    """Yield fresh document contexts; byte models use implicit BOS, BPE callers prepend BOS."""
    if split not in {"val", "test", "train"}:
        raise ValueError("unknown split")
    root = Path(supply)
    with open(root / f"{split}.bin", "rb") as binary, open(root / f"{split}.idx", encoding="utf-8") as index:
        data = mmap.mmap(binary.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for line in index:
                doc = json.loads(line)
                start, limit = doc["start"], doc["end"]
                while start < limit:
                    end = utf8_end(data, start, window_bytes, limit)
                    yield ByteWindow(data[start:end], start, end, doc["sha256"])
                    start = end
        finally:
            data.close()


def _permutation(index: int, size: int, seed: int, epoch: int) -> int:
    if size == 1:
        return 0
    bits = max(2, (size - 1).bit_length())
    bits += bits % 2
    half, mask = bits // 2, (1 << (bits // 2)) - 1
    value = index
    while True:
        left, right = value >> half, value & mask
        for round_number in range(6):
            key = f"{seed}:{epoch}:{round_number}:{right}".encode("ascii")
            mixed = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") & mask
            left, right = right, left ^ mixed
        value = (left << half) | right
        if value < size:
            return value


def sample_window_index(seed: int, global_step: int, micro_index: int, windows_per_step: int, total_windows: int) -> int:
    if min(global_step, micro_index) < 0 or windows_per_step <= 0 or total_windows <= 0 or micro_index >= windows_per_step:
        raise ValueError("invalid deterministic sampling coordinates")
    epoch, offset = divmod(global_step * windows_per_step + micro_index, total_windows)
    return _permutation(offset, total_windows, seed, epoch)


def tokenize_window(tokenizer: object, data: bytes) -> list[int]:
    text = data.decode("utf-8")
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if tokenizer.decode(ids, skip_special_tokens=False).encode("utf-8") != data:
        raise ValueError("tokenizer did not round-trip the UTF-8 window")
    if not ids and data:
        raise ValueError("tokenizer produced no tokens")
    return ids
