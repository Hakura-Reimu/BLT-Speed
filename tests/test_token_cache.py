import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from blt.config import BPEConfig, TransformerConfig
from blt.data import WindowIndex, prepare_supply, sha256_file
from blt.token_cache import TokenCache, build_token_cache
from blt.train import Trainer, TrainingConfig, make_model


def _fixture(tmp_path):
    texts = [f"Document {i}: byte level text with é🙂 and varied words. " * 2 for i in range(40)]
    source = tmp_path / "docs.parquet"
    pq.write_table(pa.table({"text": texts}), source)
    supply = tmp_path / "supply"
    prepare_supply([source], supply, seed=7, val_bytes=150, test_bytes=150, train_bytes=1500)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=320, special_tokens=["<|bos|>"],
                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    cache_path = build_token_cache(supply, path, tokenizer, window_bytes=16, batch_size=5, sample_size=7)
    return supply, tokenizer, path, cache_path


def test_token_cache_roundtrip_provenance_and_tampering(tmp_path):
    supply, tokenizer, path, cache_path = _fixture(tmp_path)
    manifest_hash = sha256_file(supply / "manifest.json")
    token_hash = sha256_file(path)
    assert cache_path == supply / "tokenized" / token_hash
    with WindowIndex(supply / "train.bin", 16) as windows:
        cache = TokenCache(cache_path, windows, manifest_hash, token_hash, tokenizer.get_vocab_size())
        assert len(cache) == len(windows)
        for index in range(len(windows)):
            assert tokenizer.decode(cache[index].tolist(), skip_special_tokens=False).encode() == windows[index].data
        assert cache.meta["verified_windows"] == 7
        assert cache.meta["bytes_per_token"] == pytest.approx(cache.meta["total_bytes"] / cache.meta["total_tokens"])
        with pytest.raises(ValueError, match="does not match"):
            TokenCache(cache_path, windows, "different", token_hash, tokenizer.get_vocab_size())
        with pytest.raises(ValueError, match="does not match"):
            TokenCache(cache_path, windows, manifest_hash, "different", tokenizer.get_vocab_size())
        with pytest.raises(FileExistsError):
            build_token_cache(supply, path, tokenizer, window_bytes=16)
        with open(cache_path / "tokens.bin", "ab") as binary:
            binary.write(b"x")
        with pytest.raises(ValueError, match="checksum mismatch"):
            TokenCache(cache_path, windows, manifest_hash, token_hash, tokenizer.get_vocab_size())


def test_training_uses_cached_tokens_and_fixed_padded_microbatches(tmp_path):
    supply, tokenizer, path, cache_path = _fixture(tmp_path)
    manifest_hash, token_hash = sha256_file(supply / "manifest.json"), sha256_file(path)
    transformer = TransformerConfig(1, 16, 2, 1, 8, 24, max_positions=128)
    model_config = BPEConfig("tiny", transformer, tokenizer.get_vocab_size(), tokenizer.token_to_id("<|bos|>"))
    config = TrainingConfig(model_config, 1, 9, 3, 2, 16, 0.001, 20, warmup_steps=2)
    class NeverTokenize:
        def encode(self, *args, **kwargs):
            raise AssertionError("training attempted to tokenize a cached window")
    with WindowIndex(supply / "train.bin", 16) as windows:
        cache = TokenCache(cache_path, windows, manifest_hash, token_hash, tokenizer.get_vocab_size())
        trainer = Trainer(config, make_model(model_config), windows, tmp_path / "run", "cpu", manifest_hash,
                          token_hash, NeverTokenize(), cache)
        batch, targets, _, _ = trainer._prepare_microbatch((0,))
        # The 256-token bucket is capped at this tiny test model's 128 positions.
        assert batch["tokens"].shape == targets.shape == (2, 128)
        assert torch.all(targets[1] == -100)
        valid = len(cache[0])
        assert torch.all(targets[0, valid:] == -100)
        with torch.no_grad():
            expected = trainer.model(batch["tokens"][:, :valid])
            actual = trainer.model(batch)
        torch.testing.assert_close(actual[:, :valid], expected, atol=2e-5, rtol=2e-5)
        trainer.start_prefetch()
        try:
            first = trainer.train_step()
            second = trainer.train_step()
        finally:
            trainer.stop_prefetch()
        assert first["sample_indices"] == trainer.sample_indices(0)
        assert second["sample_indices"] == trainer.sample_indices(1)
        assert "window_sha256" not in first
