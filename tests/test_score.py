import math
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from blt.data import ByteWindow
from blt.config import ByteConfig, TransformerConfig
from blt.data import WindowIndex
from blt.eval_io import load_evaluation_model
from blt.score import score_chunks, score_supply
from blt.train import Trainer, TrainingConfig, make_model


class Uniform(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.classes = classes

    def forward(self, inputs):
        return torch.zeros((*inputs.shape, self.classes), device=inputs.device)


class BytesTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=list(text.encode("utf-8")))

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode("utf-8")


@pytest.mark.parametrize("classes,expected", [(256, 8.0), (257, math.log2(257))])
def test_uniform_byte_logits_and_exact_accounting(classes, expected):
    chunks = [ByteWindow("aé".encode(), 0, 3, "doc-a"), ByteWindow(b" xyz", 10, 14, "doc-b")]
    result = score_chunks(Uniform(classes), chunks, family="byte")
    assert result["bytes"] == 7 and result["chunk_count"] == 2
    assert result["bpb"] == pytest.approx(expected, abs=1e-6)
    assert len(result["per_chunk_nll"]) == 2


def test_bpe_and_byte_score_the_same_document_bytes():
    chunks = [ByteWindow("aé".encode(), 0, 3, "a"), ByteWindow(b" xyz", 10, 14, "b")]
    byte = score_chunks(Uniform(257), chunks, family="byte")
    bpe = score_chunks(Uniform(257), chunks, family="bpe", tokenizer=BytesTokenizer(), bos_id=256)
    assert byte["bytes"] == bpe["bytes"] == 7
    assert [r["bytes"] for r in byte["chunks"]] == [r["bytes"] for r in bpe["chunks"]]
    assert byte["bpb"] == pytest.approx(bpe["bpb"], abs=1e-7)


def test_supply_scoring_resets_each_document_and_excludes_separators(tmp_path):
    docs = ["first é🙂 document".encode("utf-8"), "second 中 text".encode("utf-8")]
    stream = b"\n".join(docs)
    (tmp_path / "val.bin").write_bytes(stream)
    offset = 0
    lines = []
    for doc in docs:
        lines.append(json.dumps({"start": offset, "end": offset + len(doc), "sha256": hashlib.sha256(doc).hexdigest()}))
        offset += len(doc) + 1
    (tmp_path / "val.idx").write_text("\n".join(lines) + "\n", encoding="utf-8")
    byte = score_supply(Uniform(257), str(tmp_path), split="val", window_bytes=7, family="byte")
    bpe = score_supply(Uniform(257), str(tmp_path), split="val", window_bytes=7, family="bpe", tokenizer=BytesTokenizer(), bos_id=256)
    assert byte["bytes"] == bpe["bytes"] == sum(map(len, docs))
    assert [(row["document"], row["start"], row["end"]) for row in byte["chunks"]] == [
        (row["document"], row["start"], row["end"]) for row in bpe["chunks"]]


def test_loads_frozen_run_checkpoint_without_an_optimizer(tmp_path):
    data = tmp_path / "train.bin"
    data.write_bytes(b"abcdefgh" * 4)
    trunk = TransformerConfig(1, 16, 2, 1, 8, 32, max_positions=128)
    config = ByteConfig("tiny", trunk, 16, 2, 32, 3, 6, max_len=7, hash_rows=32)
    training = TrainingConfig(config, 0, 0, 1, 1, 8, 0.001, 20)
    with WindowIndex(data, 8) as windows:
        model = make_model(config)
        trainer = Trainer(training, model, windows, tmp_path / "run", "cpu", "manifest")
        trainer.save()
        loaded = load_evaluation_model(run=tmp_path / "run", device="cpu")
        raw = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            torch.testing.assert_close(loaded.model(raw), model(raw))
        assert loaded.identity["kind"] == "checkpoint" and loaded.identity["step"] == 0
