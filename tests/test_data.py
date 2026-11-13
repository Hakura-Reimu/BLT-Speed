import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from blt.data import (WindowIndex, iter_eval_chunks, prepare_supply, sample_window_index, split_documents,
                      tokenize_window, utf8_end, verify_supply)


def test_split_duplicates_and_sampling_permutation_are_deterministic():
    documents = [(f"document {i}: " + "x" * 30).encode() for i in range(24)]
    documents.extend([documents[0], documents[5]])
    first = split_documents(documents, 17, 80, 80, 300)
    second = split_documents(reversed(documents), 17, 80, 80, 300)
    assert first == second
    sets = [set(first[key]) for key in ("train", "val", "test")]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert sum(map(len, sets)) == sum(map(len, first.values()))
    for epoch in range(3):
        indices = [sample_window_index(9, epoch * 23 + i, 0, 1, 23) for i in range(23)]
        assert sorted(indices) == list(range(23))
    assert sample_window_index(8, 14, 2, 4, 23) == sample_window_index(8, 14, 2, 4, 23)


def test_utf8_windows_and_tokenizer_round_trip(tmp_path):
    text = ("Aé中🙂 \nalpha—β, punctuation! " * 12).encode("utf-8")
    chunks = []
    offset = 0
    while offset < len(text):
        end = utf8_end(text, offset, 13)
        chunk = text[offset:end]
        chunk.decode("utf-8")
        chunks.append(chunk)
        offset = end
    assert b"".join(chunks) == text
    path = tmp_path / "stream.bin"
    path.write_bytes(text)
    with WindowIndex(path, 13, drop_short_tail=False) as index:
        assert b"".join(index[i].data for i in range(len(index))) == text
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["<|bos|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator([text.decode()], trainer)
    for chunk in chunks:
        assert tokenizer.decode(tokenize_window(tokenizer, chunk), skip_special_tokens=False).encode() == chunk


def test_prepare_supply_is_disjoint_immutable_and_self_verifying(tmp_path):
    texts = [f"document {i}: English é🙂 and some more words {i} " * 2 for i in range(40)]
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"text": texts + texts[:5]}), source)
    root = tmp_path / "supply"
    manifest = prepare_supply([source], root, seed=41, val_bytes=100, test_bytes=100, train_bytes=500)
    verify_supply(root)
    assert manifest["source_documents"] == 45
    assert manifest["unique_documents"] == 40
    assert manifest["duplicate_documents"] == 5
    ids = []
    for split in ("train", "val", "test"):
        records = [json.loads(line) for line in (root / f"{split}.idx").read_text().splitlines()]
        ids.append({record["sha256"] for record in records})
        raw = (root / f"{split}.bin").read_bytes()
        for i, record in enumerate(records):
            doc = raw[record["start"]:record["end"]]
            assert hashlib.sha256(doc).hexdigest() == record["sha256"]
            assert record["separator_before"] == (0 if i == 0 else 1)
            if i:
                assert raw[record["start"] - 1:record["start"]] == b"\n"
        chunks = list(iter_eval_chunks(root, split, 17))
        for record in records:
            assert b"".join(chunk.data for chunk in chunks if chunk.document == record["sha256"]) == raw[record["start"]:record["end"]]
            assert all(len(chunk.data) <= 17 for chunk in chunks)
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    with pytest.raises(FileExistsError):
        prepare_supply([source], root, train_bytes=1, val_bytes=1, test_bytes=1)
    changed = root / "train.bin"
    changed.write_bytes(changed.read_bytes() + b"x")
    with pytest.raises(ValueError, match="immutable"):
        verify_supply(root)
