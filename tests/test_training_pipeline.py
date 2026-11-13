import torch

from blt.config import ByteConfig, TransformerConfig
from blt.data import WindowIndex
from blt.train import Trainer, TrainingConfig, make_model


def test_byte_training_uses_cpu_patch_metadata_and_one_result_sync(tmp_path, monkeypatch):
    import blt.model as module

    path = tmp_path / "train.bin"
    path.write_bytes(b"words, punctuation! and spaces form patches. " * 20)
    trunk = TransformerConfig(1, 16, 2, 1, 8, 32, max_positions=128)
    model_config = ByteConfig("tiny", trunk, 16, 2, 32, 3, 6, max_len=7, hash_rows=32)
    config = TrainingConfig(model_config, 4, 8, 3, 2, 65, 0.001, 20, warmup_steps=2)
    with WindowIndex(path, 65) as windows:
        trainer = Trainer(config, make_model(model_config), windows, tmp_path / "run", "cpu", "manifest")
        def forbidden(*args, **kwargs):
            raise AssertionError("training fell back to host patch discovery")
        calls = []
        original = torch.Tensor.cpu
        def tracked(tensor, *args, **kwargs):
            calls.append(1)
            return original(tensor, *args, **kwargs)
        with monkeypatch.context() as scoped:
            scoped.setattr(module, "_byte_patches", forbidden)
            scoped.setattr(torch.Tensor, "cpu", tracked)
            result = trainer.train_step()
        assert len(calls) == 1
        assert result["consumed_bytes"] > 0
        assert result["units_with_tail_per_window"] >= result["units_per_window"]
        assert "window_sha256" not in result
