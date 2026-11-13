from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from blt.config import BPEConfig, ByteConfig, TransformerConfig
from blt.data import WindowIndex
from blt.token_cache import MemoryTokenCache
from blt.train import LongContextTrainer, Trainer, TrainingConfig, bucket_length, make_model, seed_everything


class ByteTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=list(text.encode("utf-8")))

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode("utf-8")


def _config():
    trunk = TransformerConfig(1, 8, 2, 1, 4, 16, max_positions=64)
    model = ByteConfig("tiny", trunk, 8, 2, 16, 2, 6, max_len=4, block_size=8, hash_rows=8)
    return TrainingConfig(model, 3, 9, 8, 2, 8, 0.001, 20, warmup_steps=2)


def _buckets(stack, data):
    return {size: stack.enter_context(WindowIndex(data, size)) for size in (8, 16, 32)}


def test_long_context_initialization_and_resume_are_deterministic(tmp_path):
    path = tmp_path / "train.bin"
    path.write_bytes(b"one small context, two small contexts! " * 80)
    config = _config()
    with ExitStack() as stack:
        buckets = _buckets(stack, path)
        seed_everything(config.init_seed)
        source = Trainer(config, make_model(config.model), buckets[8], tmp_path / "source", "cpu", "manifest")
        source.train_step()
        checkpoint = source.save()
        incompatible = LongContextTrainer(config, make_model(config.model), buckets, tmp_path / "wrong-data", "cpu", "another-manifest")
        with pytest.raises(ValueError, match="incompatible"):
            incompatible.initialize_from(checkpoint)

        def continuation(name):
            trainer = LongContextTrainer(config, make_model(config.model), buckets, tmp_path / name, "cpu", "manifest")
            trainer.initialize_from(checkpoint)
            return trainer

        uninterrupted = continuation("full")
        assert uninterrupted.sample_indices(0) == source.sample_indices(1)
        assert uninterrupted.scheduler.state_dict() == source.scheduler.state_dict()
        for size, expected_batch in ((8, 2), (16, 2), (32, 1)):
            with uninterrupted._using_bucket(size):
                batch, target, _, _ = uninterrupted._prepare_microbatch((0,))
                assert batch["bytes"].shape == target.shape == (expected_batch, size)

        expected = [uninterrupted.train_step() for _ in range(4)]
        first = continuation("resumed")
        actual = [first.train_step() for _ in range(2)]
        first.save()
        resumed = LongContextTrainer(config, make_model(config.model), buckets, tmp_path / "resumed", "cpu", "manifest")
        resumed.load()
        actual.extend(resumed.train_step() for _ in range(2))
        assert [row["bucket_bytes"] for row in actual] == [8, 16, 8, 32]
        assert [row["sample_indices"] for row in actual] == [row["sample_indices"] for row in expected]
        assert [row["loss_nats_per_byte"] for row in actual] == pytest.approx(
            [row["loss_nats_per_byte"] for row in expected], abs=1e-6, rel=0)
        assert resumed.bucket_bytes == {8: 128, 16: 64, 32: 64}
        assert resumed.initialization["source_consumed_bytes"] == source.consumed_bytes
        for left, right in zip(resumed.model.parameters(), uninterrupted.model.parameters()):
            torch.testing.assert_close(left, right, atol=1e-6, rtol=0)


def test_long_context_bpe_uses_a_token_cache_and_padding_bucket_for_each_window(tmp_path):
    data = tmp_path / "train.bin"
    data.write_bytes(b"abc def ghi jkl " * 80)
    trunk = TransformerConfig(1, 8, 2, 1, 4, 16, max_positions=64)
    model = BPEConfig("tiny-bpe", trunk, 257, 256)
    config = TrainingConfig(model, 1, 9, 8, 2, 8, 0.001, 20, warmup_steps=2)
    tokenizer = ByteTokenizer()
    with ExitStack() as stack:
        buckets = _buckets(stack, data)
        caches = {size: MemoryTokenCache(window, tokenizer) for size, window in buckets.items()}
        trainer = LongContextTrainer(config, make_model(model), buckets, tmp_path / "run", "cpu",
                                     "manifest", "tokenizer", tokenizer, caches)
        for size in buckets:
            with trainer._using_bucket(size):
                inputs, targets, _, _ = trainer._prepare_microbatch((0,))
                assert inputs["tokens"].shape == targets.shape
                assert targets.shape[1] == bucket_length(len(caches[size][0]), 256, model.trunk.max_positions)
                assert targets[0, :size].tolist() == list(buckets[size][0].data)
        assert trainer.bucket_cycle == (8, 16, 8, 32)
        with pytest.raises(ValueError, match="missing pretokenized"):
            LongContextTrainer(config, make_model(model), buckets, tmp_path / "missing", "cpu",
                               "manifest", "tokenizer", tokenizer, {8: caches[8]})


def test_bucket_cache_preparation_creates_independent_verified_caches(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("longctx_train_cli", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "LONG_CONTEXT_LENGTHS", (8, 16, 32))
    supply = tmp_path / "supply"
    supply.mkdir()
    records = {}
    for split in ("train", "val", "test"):
        for ext in ("bin", "idx"):
            name = f"{split}.{ext}"
            data = (b"small words and spaces. " * 100) if ext == "bin" else b"{}\n"
            (supply / name).write_bytes(data)
            records[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    (supply / "manifest.json").write_text(json.dumps({"version": 1, "files": records}), encoding="utf-8")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(["small words and spaces. " * 10], trainers.BpeTrainer(
        vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    paths = module.prepare_long_context_caches(supply, tokenizer_path, tokenizer)
    assert [path.name for path in paths] == ["window-8", "window-16", "window-32"]
    assert [json.loads((path / "meta.json").read_text())["window_bytes"] for path in paths] == [8, 16, 32]
    assert module.prepare_long_context_caches(supply, tokenizer_path, tokenizer) == paths
