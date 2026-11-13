from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from blt.config import BPEConfig, ByteConfig, TransformerConfig
from blt.data import ByteWindow
from blt.patcher import max_patch_count
from blt.train import Trainer, TrainingConfig, _loss, bucket_length, bucketed_patches, make_model


class ByteTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=list(text.encode("utf-8")))

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode("utf-8")


class Windows:
    def __init__(self, data):
        self.data = data
        self.windows = [(0, len(row)) for row in data]
        self.index_sha256 = "fixture"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return ByteWindow(self.data[index], 0, len(self.data[index]))


class TokenCache:
    def __init__(self, windows):
        self.tokens = [np.frombuffer(row, dtype=np.uint8) for row in windows.data]
        self.max_tokens = max(map(len, self.tokens))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index]


def trainer_fixture(tmp_path, family, *, hash_enabled=True):
    if family == "bpe":
        windows = Windows([b"a" * 60, b"b" * 100, b"c" * 430])
        trunk = TransformerConfig(1, 8, 2, 1, 4, 16, max_positions=768)
        model_config = BPEConfig("tiny-bpe", trunk, 257, 256)
        tokenizer, cache = ByteTokenizer(), TokenCache(windows)
    else:
        windows = Windows([b"a" * 512, b"b" * 120, b" " * 512])
        trunk = TransformerConfig(1, 8, 2, 1, 4, 16, max_positions=513)
        model_config = ByteConfig("tiny-byte", trunk, 8, 2, 16, 3, 6, hash_rows=16, hash_enabled=hash_enabled)
        tokenizer, cache = None, None
    config = TrainingConfig(model_config, 1, 9, 2, 2, 512, 0.001, 20, warmup_steps=2, val_windows=2)
    torch.manual_seed(17)
    trainer = Trainer(config, make_model(model_config), windows, tmp_path / family, "cpu", "manifest",
                      "tokenizer" if tokenizer else None, tokenizer, cache)
    return trainer


def assert_loss_and_gradients_match(model, bucketed, target, wide, wide_target):
    other = copy.deepcopy(model)
    loss_a, loss_b = _loss(model, bucketed, target), _loss(other, wide, wide_target)
    torch.testing.assert_close(loss_a, loss_b, rtol=1e-5, atol=2e-5)
    loss_a.backward()
    loss_b.backward()
    for left, right in zip(model.parameters(), other.parameters()):
        if left.grad is None or right.grad is None:
            assert left.grad is None and right.grad is None
        else:
            torch.testing.assert_close(left.grad, right.grad, rtol=3e-5, atol=3e-5)


def test_bucket_selection_at_boundaries_and_limits():
    assert [bucket_length(n, 256, 1024) for n in (0, 1, 255, 256, 257, 1024)] == [0, 256, 256, 256, 512, 1024]
    assert bucket_length(65, 256, 128) == 128
    assert bucket_length(50, 128, 63) == 63
    with pytest.raises(ValueError, match="limit"):
        bucket_length(129, 256, 128)
    with pytest.raises(ValueError):
        bucket_length(-1, 256, 512)


def test_patch_capacity_at_exact_boundary_retains_open_tail():
    trunk = TransformerConfig(1, 8, 2, 1, 4, 16, max_positions=513)
    config = ByteConfig("tiny", trunk, 8, 2, 16, 3, 6, hash_enabled=False)
    values = np.full((1, 2050), ord("a"), dtype=np.uint8)
    patches = bucketed_patches(values, np.array([2050]), config)
    assert patches.counts.tolist() == [128]
    assert patches.ends.shape == (1, 128)
    assert patches.ends[0, -1] == 2047
    assert patches.patch_ids[0, -2:].tolist() == [128, 128]


def test_no_completed_byte_patches_use_only_global_bos(tmp_path):
    trunk = TransformerConfig(1, 8, 2, 1, 4, 16, max_positions=64)
    model_config = ByteConfig("tiny-empty", trunk, 8, 2, 16, 3, 6, hash_rows=16)
    config = TrainingConfig(model_config, 1, 9, 1, 1, 2, 0.001, 20, warmup_steps=2)
    trainer = Trainer(config, make_model(model_config), Windows([b"ab"]), tmp_path / "empty", "cpu", "manifest")
    batch, targets, units, _ = trainer._prepare_microbatch((0,))
    assert batch["patches"]["ends"].shape == (1, 0) and units == 0
    metadata = dict(batch["patches"])
    metadata["ends"] = F.pad(metadata["ends"], (0, 1), value=-1)
    wide = {"bytes": batch["bytes"], "patches": metadata}
    assert_loss_and_gradients_match(trainer.model, batch, targets, wide, targets)


def test_bpe_bucket_loss_gradients_and_padding_match_run_wide(tmp_path):
    trainer = trainer_fixture(tmp_path, "bpe")
    batch, targets, units, _ = trainer._prepare_microbatch((0, 1))
    repeated, repeated_targets, _, _ = trainer._prepare_microbatch((0, 1))
    assert batch["tokens"].shape == targets.shape == (2, 256)
    assert units == 160 and torch.equal(batch["tokens"], repeated["tokens"]) and torch.equal(targets, repeated_targets)
    assert torch.all(targets[0, 60:] == -100) and torch.all(targets[1, 100:] == -100)
    assert torch.all(batch["tokens"][0, 60:] == trainer.config.model.bos_id)
    wide_length = trainer.sequence_length
    assert wide_length == 448 and wide_length > targets.shape[1]
    wide = {"tokens": F.pad(batch["tokens"], (0, wide_length - 256), value=trainer.config.model.bos_id)}
    wide_targets = F.pad(targets, (0, wide_length - 256), value=-100)
    assert_loss_and_gradients_match(trainer.model, batch, targets, wide, wide_targets)
    large, _, _, _ = trainer._prepare_microbatch((2,))
    assert large["tokens"].shape == (2, 512)


@pytest.mark.parametrize("hash_enabled", [False, True])
def test_byte_patch_bucket_loss_gradients_and_masks_match_run_wide(tmp_path, hash_enabled):
    trainer = trainer_fixture(tmp_path, "byte", hash_enabled=hash_enabled)
    batch, targets, units, _ = trainer._prepare_microbatch((0, 1))
    metadata = batch["patches"]
    assert batch["bytes"].shape == targets.shape == (2, 512)
    assert metadata["ends"].shape == (2, 128)
    assert units == int(metadata["counts"].sum())
    for row, length in enumerate((512, 120)):
        count = int(metadata["counts"][row])
        assert torch.all(metadata["ends"][row, :count] >= 0)
        assert torch.all(metadata["ends"][row, count:] == -1)
        assert not metadata["boundaries"][row, length:].any()
        assert torch.all(targets[row, length:] == -100)
        assert int(metadata["patch_ids"][row].max()) <= count
    run_wide = max_patch_count(512, trainer.config.model.min_len, trainer.config.model.block_size)
    assert run_wide == 176
    wide_metadata = dict(metadata)
    wide_metadata["ends"] = F.pad(metadata["ends"], (0, run_wide - 128), value=-1)
    wide = {"bytes": batch["bytes"], "patches": wide_metadata}
    assert_loss_and_gradients_match(trainer.model, batch, targets, wide, targets)
    dense, dense_target, _, _ = trainer._prepare_microbatch((2,))
    assert dense["patches"]["ends"].shape == (2, 256)
    assert torch.all(dense_target[1] == -100) and torch.all(dense["patches"]["ends"][1] == -1)


@pytest.mark.parametrize("family", ["bpe", "byte"])
def test_validation_uses_bucketed_shapes_and_masks(tmp_path, monkeypatch, family):
    import blt.train as train
    trainer = trainer_fixture(tmp_path, family)
    docs = [b"a" * 60, b"b" * 100]
    supply = tmp_path / "supply"
    supply.mkdir()
    (supply / "val.bin").write_bytes(b"".join(docs))
    cursor = 0
    records = []
    for doc in docs:
        records.append({"start": cursor, "end": cursor + len(doc), "sha256": hashlib.sha256(doc).hexdigest()})
        cursor += len(doc)
    (supply / "val.idx").write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    captured = []
    original_loss = train._loss
    def record(model, inputs, targets):
        captured.append((inputs, targets))
        return original_loss(model, inputs, targets)
    monkeypatch.setattr(train, "_loss", record)
    assert np.isfinite(trainer.validate(supply))
    assert len(captured) == 1
    inputs, targets = captured[0]
    assert torch.all(targets[0, 60:] == -100) and torch.all(targets[1, 100:] == -100)
    if family == "bpe":
        assert inputs.shape == (2, 256) and targets.shape == (2, 256)
    else:
        assert inputs["bytes"].shape == targets.shape == (2, 512)
        assert inputs["patches"]["ends"].shape == (2, 128)


@pytest.mark.parametrize("family", ["bpe", "byte"])
def test_padding_metrics_and_bucket_history_resume(tmp_path, family):
    trainer = trainer_fixture(tmp_path, family)
    trainer.sample_indices = lambda step: [0, 1]
    metrics = trainer.train_step()
    expected_capacity = 256 if family == "bpe" else 128
    assert metrics["padding_padded_positions"] == 2 * expected_capacity
    assert metrics["padding_efficiency"] == pytest.approx(metrics["padding_real_positions"] / (2 * expected_capacity))
    assert metrics["padding_distinct_buckets"] == metrics["padding_new_buckets"] == 1
    checkpoint = trainer.save()
    resumed = trainer_fixture(tmp_path / "other", family)
    resumed.sample_indices = lambda step: [0, 1]
    # Resume must use the same frozen run directory.
    resumed.run_dir = trainer.run_dir
    resumed.load(checkpoint)
    assert resumed._padding_buckets_seen == trainer._padding_buckets_seen
    assert resumed._padding_history_complete
    assert resumed.train_step()["padding_new_buckets"] == 0
    legacy = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for name in ("padding_bucket_shapes_seen", "padding_history_complete", "padding_policy"):
        legacy.pop(name)
    legacy_path = tmp_path / "legacy.pt"
    torch.save(legacy, legacy_path)
    old_resume = trainer_fixture(tmp_path / "legacy-run", family)
    old_resume.sample_indices = lambda step: [0, 1]
    old_resume.run_dir = trainer.run_dir
    old_resume.load(legacy_path)
    assert not old_resume._padding_history_complete
    assert old_resume.train_step()["padding_history_complete"] is False


def test_benchmark_excludes_compile_steps_and_reports_efficiency():
    path = Path(__file__).resolve().parents[1] / "scripts" / "bench_train.py"
    spec = importlib.util.spec_from_file_location("bucket_bench_train", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class FakeTrainer:
        config = SimpleNamespace(windows_per_step=2)
        consumed_bytes = 0
        calls = 0
        def train_step(self):
            self.consumed_bytes += 100
            new_graphs = [1, 0, 1, 0][self.calls]
            self.calls += 1
            return {"compile_new_graphs": new_graphs, "compile_recompiles": int(self.calls > 1 and new_graphs > 0),
                    "padding_bucket_shapes": [[2, 0, 256]], "padding_real_positions": 10,
                    "padding_padded_positions": 12, "units_with_tail_per_window": 5,
                    "data_wait_seconds": 0.0}
    result = module.measure_training(FakeTrainer(), torch.device("cpu"), steps=2, warmup_steps=1, max_attempts=3)
    assert result["status"] == "ok" and result["steady_steps"] == 2
    assert result["measurement_attempts"] == 3 and result["excluded_compile_steps"] == 1
    assert result["warmup_compile_new_graphs"] == result["compile_new_graphs"] == 1
    assert result["padding_efficiency"] == pytest.approx(10 / 12)
    assert result["distinct_padding_buckets"] == 1 and result["bytes_per_second"] > 0

    class AlwaysCompiles(FakeTrainer):
        def train_step(self):
            row = super().train_step()
            row["compile_new_graphs"] = 1
            return row
    incomplete = module.measure_training(AlwaysCompiles(), torch.device("cpu"), steps=2, warmup_steps=0, max_attempts=2)
    assert incomplete["status"] == "insufficient_steady_steps" and incomplete["steady_steps"] == 0
    assert "bytes_per_second" not in incomplete


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled BF16 bucket parity requires CUDA")
def test_cuda_compiled_training_reuses_a_graph_for_each_bucket(tmp_path):
    from torch._dynamo import reset
    reset()
    trunk = TransformerConfig(1, 8, 2, 1, 4, 16, max_positions=768)
    model_config = BPEConfig("tiny-cuda", trunk, 257, 256)
    config = TrainingConfig(model_config, 3, 9, 1, 1, 512, 0.001, 20, warmup_steps=2, compile_model=True)
    windows = Windows([b"a" * 80, b"b" * 300])
    cache, tokenizer = TokenCache(windows), ByteTokenizer()
    torch.manual_seed(33)
    eager_model = make_model(model_config)
    compiled_model = copy.deepcopy(eager_model)
    eager = Trainer(replace(config, compile_model=False), eager_model, windows, tmp_path / "eager", "cuda",
                    "manifest", "tokenizer", tokenizer, cache)
    compiled = Trainer(config, compiled_model, windows, tmp_path / "compiled", "cuda",
                       "manifest", "tokenizer", tokenizer, cache)
    eager.sample_indices = compiled.sample_indices = lambda step: [step % 2]
    eager_metrics = [eager.train_step() for _ in range(3)]
    compiled_metrics = [compiled.train_step() for _ in range(3)]
    assert [row["padding_bucket_shapes"][0][-1] for row in compiled_metrics] == [256, 512, 256]
    assert compiled_metrics[0]["compile_new_graphs"] >= 1 and compiled_metrics[1]["compile_new_graphs"] >= 1
    assert compiled_metrics[2]["compile_new_graphs"] == 0
    for actual, expected in zip(compiled_metrics, eager_metrics):
        assert actual["loss_nats_per_byte"] == pytest.approx(expected["loss_nats_per_byte"], rel=0.02, abs=0.02)
