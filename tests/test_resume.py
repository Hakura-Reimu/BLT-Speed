from types import SimpleNamespace
import hashlib
import json
import random

import numpy as np
import pytest
import torch

from blt.config import BPEConfig, TransformerConfig
from blt.data import WindowIndex
from blt.train import Trainer, TrainingConfig, make_model, seed_everything


class ByteTokenizer:
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=list(text.encode("utf-8")))

    def decode(self, ids, skip_special_tokens=False):
        return bytes(ids).decode("utf-8")


def test_resume_matches_twenty_continuous_updates(tmp_path):
    data = tmp_path / "train.bin"
    data.write_bytes((b"some short simple words for deterministic model training. " * 8))
    transformer = TransformerConfig(layers=1, width=16, q_heads=2, kv_heads=1, head_dim=8, ffn=24, max_positions=64)
    model_config = BPEConfig("tiny", transformer, vocab_size=257, bos_id=256)
    config = TrainingConfig(model_config, init_seed=4, data_seed=99, windows_per_step=2, micro_batch_size=1,
                            window_bytes=8, peak_lr=0.003, schedule_steps=20, warmup_steps=2)
    with WindowIndex(data, 8) as windows:
        seed_everything(config.init_seed)
        continuous = Trainer(config, make_model(model_config), windows, tmp_path / "continuous", "cpu", "manifest", "tokenizer", ByteTokenizer())
        continuous.start_prefetch()
        try:
            expected = [continuous.train_step() for _ in range(20)]
        finally:
            continuous.stop_prefetch()
        seed_everything(config.init_seed)
        first = Trainer(config, make_model(model_config), windows, tmp_path / "resumed", "cpu", "manifest", "tokenizer", ByteTokenizer())
        first.start_prefetch()
        try:
            actual = [first.train_step() for _ in range(10)]
            checkpoint = first.save()
        finally:
            first.stop_prefetch()
        assert torch.load(checkpoint, map_location="cpu", weights_only=False)["sample_audit"] == continuous.sample_audit(9)
        (tmp_path / "resumed/log.jsonl").write_bytes(b'{"step": 10}\n{"step": 11}\n{"step":')
        expected_rng = (random.random(), np.random.random(), torch.rand(1))
        seed_everything(1234)
        second = Trainer(config, make_model(model_config), windows, tmp_path / "resumed", "cpu", "manifest", "tokenizer", ByteTokenizer())
        second.load()
        assert (tmp_path / "resumed/log.jsonl").read_text() == '{"step": 10}\n'
        assert random.random() == expected_rng[0]
        assert np.random.random() == expected_rng[1]
        torch.testing.assert_close(torch.rand(1), expected_rng[2])
        second.start_prefetch()
        try:
            actual.extend(second.train_step() for _ in range(10))
        finally:
            second.stop_prefetch()
        assert [entry["sample_indices"] for entry in actual] == [entry["sample_indices"] for entry in expected]
        assert [entry["consumed_bytes"] for entry in actual] == [entry["consumed_bytes"] for entry in expected]
        assert [entry["loss_nats_per_byte"] for entry in actual] == pytest.approx(
            [entry["loss_nats_per_byte"] for entry in expected], abs=1e-6, rel=0)
        assert second.step == continuous.step == 20
        assert second.consumed_bytes == continuous.consumed_bytes
        assert second.sample_audit(19) == continuous.sample_audit(19)
        for a, b in zip(continuous.model.parameters(), second.model.parameters()):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=0)


def test_optimizer_excludes_embedding_and_norm_weights_from_decay(tmp_path):
    data = tmp_path / "bytes.bin"
    data.write_bytes(b"abcdefgh" * 3)
    transformer = TransformerConfig(1, 16, 2, 1, 8, 24, max_positions=64)
    model_config = BPEConfig("tiny", transformer, 257, 256)
    config = TrainingConfig(model_config, 1, 1, 1, 1, 8, 0.001, 20)
    with WindowIndex(data, 8) as windows:
        model = make_model(model_config)
        trainer = Trainer(config, model, windows, tmp_path / "run", "cpu", "manifest", "tokenizer", ByteTokenizer())
        decay = {id(p) for group in trainer.optimizer.param_groups if group["weight_decay"] for p in group["params"]}
        assert id(model.embedding.weight) not in decay
        assert id(model.trunk.norm.weight) not in decay
        assert id(model.trunk.blocks[0].attn.q.weight) in decay


def test_checkpoint_rotation_preserves_milestones(tmp_path):
    data = tmp_path / "bytes.bin"
    data.write_bytes(b"abcdefgh" * 4)
    transformer = TransformerConfig(1, 16, 2, 1, 8, 24, max_positions=64)
    model_config = BPEConfig("tiny", transformer, 257, 256)
    config = TrainingConfig(model_config, 1, 1, 1, 1, 8, 0.001, 20)
    with WindowIndex(data, 8) as windows:
        trainer = Trainer(config, make_model(model_config), windows, tmp_path / "run", "cpu", "manifest", "tokenizer", ByteTokenizer())
        trainer.train_step()
        trainer.save((8,))
        milestone = tmp_path / "run/ckpt/bytes-8.pt"
        original = hashlib.sha256(milestone.read_bytes()).hexdigest()
        trainer.train_step()
        trainer.save()
        trainer.train_step()
        trainer.save()
        directory = tmp_path / "run/ckpt"
        assert (directory / "rolling-0.pt").exists() and (directory / "rolling-1.pt").exists()
        assert hashlib.sha256(milestone.read_bytes()).hexdigest() == original
        assert json.loads((directory / "latest.json").read_text())["step"] == 3
        assert torch.load(milestone, map_location="cpu", weights_only=False)["step"] == 1


def test_active_time_milestone_and_numerics_policy_survive_resume(tmp_path, monkeypatch):
    import time
    import blt.train as train
    from blt.score import NUMERICS_POLICY
    monkeypatch.setattr(train, "TIME_MILESTONES", (0,))
    original_save = Trainer.save
    def slow_save(self, *args, **kwargs):
        time.sleep(0.2)
        return original_save(self, *args, **kwargs)
    monkeypatch.setattr(Trainer, "save", slow_save)
    data = tmp_path / "bytes.bin"
    data.write_bytes(b"abcdefgh" * 4)
    transformer = TransformerConfig(1, 16, 2, 1, 8, 24, max_positions=64)
    model_config = BPEConfig("tiny", transformer, 257, 256)
    config = TrainingConfig(model_config, 1, 1, 1, 1, 8, 0.001, 20)
    pause = tmp_path / "PAUSE"
    with WindowIndex(data, 8) as windows:
        first = Trainer(config, make_model(model_config), windows, tmp_path / "run", "cpu", "manifest", "tokenizer", ByteTokenizer())
        first.run(tmp_path, stop_bytes=16, pause_file=pause)
        milestone = tmp_path / "run/ckpt/active-0s.pt"
        assert torch.load(milestone, map_location="cpu", weights_only=False)["step"] == 1
        assert json.loads((tmp_path / "run/numerics_policy.json").read_text()) == NUMERICS_POLICY
        logged = [json.loads(line)["active_seconds"] for line in (tmp_path / "run/log.jsonl").read_text().splitlines()]
        assert logged == sorted(logged) and logged[0] > 0
        sidecar = json.loads((tmp_path / "run/ckpt/active-0s.json").read_text())
        assert sidecar["step"] == 1 and sidecar["overshoot_seconds"] == pytest.approx(sidecar["active_seconds"])
        # The last checkpoint was written after the last logged step; only the ledger holds that write.
        ledger = json.loads((tmp_path / "run/active_time.json").read_text())
        assert ledger["sessions"] == 1 and ledger["active_seconds"] >= logged[-1] + 0.2
        second = Trainer(config, make_model(model_config), windows, tmp_path / "run", "cpu", "manifest", "tokenizer", ByteTokenizer())
        second.load()
        assert second.active_seconds == pytest.approx(ledger["active_seconds"])
        assert second.saved_time_milestones == {0}
        second.run(tmp_path, stop_bytes=24, pause_file=pause)
        assert second.active_seconds > ledger["active_seconds"]
        assert json.loads((tmp_path / "run/active_time.json").read_text())["sessions"] == 2
        assert torch.load(milestone, map_location="cpu", weights_only=False)["step"] == 1
