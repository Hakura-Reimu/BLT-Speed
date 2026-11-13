from dataclasses import replace
import math

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from blt.config import BPEConfig, ByteConfig, TransformerConfig, parameter_counts, preset
from blt.model import BPELM, ByteLatentLM
from blt.patcher import max_patch_count, patch_batch


def trunk(layers=2):
    return TransformerConfig(layers=layers, width=16, q_heads=2, kv_heads=1, head_dim=8, ffn=32, max_positions=256)


def byte_config(hash_enabled=True):
    return ByteConfig("tiny-byte", trunk(), local_width=16, local_heads=2, local_ffn=32, min_len=3,
                      target_patch_mean=6, max_len=7, block_size=64, hash_enabled=hash_enabled, hash_rows=32)


def examples():
    a = (b"ab, cd!efgh i-jkl mn?op qr/st ") * 5
    b = (b"a bcdef gh!ij klm,nopq rstuv; ") * 5
    return torch.tensor([list(a[:133]), list(b[:133])], dtype=torch.long)


@pytest.mark.parametrize("hash_enabled", [True, False])
def test_byte_causality_across_patch_and_block_edges(hash_enabled):
    torch.manual_seed(4)
    model = ByteLatentLM(byte_config(hash_enabled)).eval()
    original = examples()
    with torch.no_grad():
        expected = model(original)
        for position in (1, 6, 7, 31, 63, 64, 65, 127, 128):
            changed = original.clone()
            changed[:, position] = ord("!") if original[0, position] != ord("!") else ord("a")
            actual = model(changed)
            torch.testing.assert_close(actual[:, :position + 1], expected[:, :position + 1], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("hash_enabled", [True, False])
def test_byte_full_forward_matches_incremental_and_prefill(hash_enabled):
    torch.manual_seed(5)
    model = ByteLatentLM(byte_config(hash_enabled)).eval()
    raw = examples()
    with torch.no_grad():
        full = model(raw)
        state = model.init_state(2, 150, "cpu", torch.float32)
        outputs = [model.prefill(state, raw[:, :0])]
        for i in range(raw.shape[1] - 1):
            outputs.append(model.step(state, raw[:, i]))
        torch.testing.assert_close(torch.cat(outputs, dim=1), full, atol=2e-5, rtol=2e-5)
        for cut in (1, 7, 63, 64, 65, 128):
            state = model.init_state(2, 150, "cpu", torch.float32)
            predicted = model.prefill(state, raw[:, :cut])
            torch.testing.assert_close(predicted, full[:, cut:cut + 1], atol=2e-5, rtol=2e-5)
            predicted = model.step(state, raw[:, cut])
            torch.testing.assert_close(predicted, full[:, cut + 1:cut + 2], atol=2e-5, rtol=2e-5)
        state = model.init_state(2, 150, "cpu", torch.float32)
        all_positions = model.prefill(state, raw[:, :65], last_only=False)
        torch.testing.assert_close(all_positions, full[:, :66], atol=2e-5, rtol=2e-5)


def test_bpe_causality_prefill_and_incremental(monkeypatch):
    torch.manual_seed(7)
    model = BPELM(BPEConfig("tiny", trunk(), vocab_size=33, bos_id=0)).eval()
    tokens = torch.randint(0, 33, (2, 73))
    with torch.no_grad():
        full = model(tokens)
        changed = tokens.clone()
        changed[:, 40:] = (changed[:, 40:] + 1) % 33
        torch.testing.assert_close(model(changed)[:, :40], full[:, :40], atol=1e-6, rtol=1e-6)
        calls = []
        linear = F.linear
        def capture(x, weight, bias=None):
            result = linear(x, weight, bias)
            if weight is model.embedding.weight:
                calls.append(x.shape)
            return result
        monkeypatch.setattr(F, "linear", capture)
        state = model.init_state(2, 80, "cpu", torch.float32)
        out = [model.prefill(state, tokens[:, :23])]
        assert calls[-1][1] == 1
        for i in range(23, tokens.shape[1]):
            out.append(model.step(state, tokens[:, i]))
        torch.testing.assert_close(torch.cat(out, dim=1), full[:, 22:], atol=2e-5, rtol=2e-5)
        state = model.init_state(2, 80, "cpu", torch.float32)
        torch.testing.assert_close(model.prefill(state, tokens, last_only=False), full, atol=1e-6, rtol=1e-6)


def test_static_global_cache_handles_maximal_block_boundary_count():
    model = ByteLatentLM(byte_config()).eval()
    raw = torch.full((1, 128), ord(" "), dtype=torch.long)
    state = model.init_state(1, 128, "cpu", torch.float32)
    with torch.no_grad():
        model.prefill(state, raw)
    assert state.globals[0].length == 45  # BOS plus 22 ends per 64-byte block.
    assert state.globals[0].keys[0].shape[2] == 45


@pytest.mark.parametrize("hash_enabled", [True, False])
def test_byte_fixed_patch_padding_preserves_logits_and_avoids_host_reads(monkeypatch, hash_enabled):
    import blt.model as module

    torch.manual_seed(13)
    model = ByteLatentLM(byte_config(hash_enabled)).eval()
    raw = examples()[:, :65]
    padded = torch.nn.functional.pad(raw, (0, 63))
    metadata = patch_batch(padded.numpy(), 3, 7, 64, lengths=np.array([65, 65]),
                           max_patches=max_patch_count(128, 3))
    patches = {"ends": torch.from_numpy(metadata.ends), "patch_ids": torch.from_numpy(metadata.patch_ids),
               "counts": torch.from_numpy(metadata.counts), "boundaries": torch.from_numpy(metadata.boundaries)}
    with torch.no_grad():
        expected = model(raw)
        with monkeypatch.context() as scoped:
            def forbidden(*args, **kwargs):
                raise AssertionError("metadata training path read tensor values on the host")
            scoped.setattr(module, "_byte_patches", forbidden)
            scoped.setattr(torch.Tensor, "cpu", forbidden)
            scoped.setattr(torch.Tensor, "item", forbidden)
            actual = model({"bytes": padded, "patches": patches})
    torch.testing.assert_close(actual[:, :65], expected, atol=2e-5, rtol=2e-5)


def test_byte_metadata_handles_different_lengths_and_empty_padding_rows():
    torch.manual_seed(17)
    model = ByteLatentLM(byte_config()).eval()
    original = examples()
    raw = torch.zeros((3, 128), dtype=torch.long)
    raw[0, :65], raw[1, :42] = original[0, :65], original[1, :42]
    patches = patch_batch(raw.numpy(), 3, 7, 64, lengths=np.array([65, 42, 0]),
                          max_patches=max_patch_count(128, 3))
    metadata = {"ends": torch.from_numpy(patches.ends), "patch_ids": torch.from_numpy(patches.patch_ids),
                "counts": torch.from_numpy(patches.counts), "boundaries": torch.from_numpy(patches.boundaries)}
    with torch.no_grad():
        actual = model({"bytes": raw, "patches": metadata})
        for row, length in enumerate((65, 42)):
            expected = model(raw[row:row + 1, :length])
            torch.testing.assert_close(actual[row:row + 1, :length], expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("patching", ["host", "device"])
def test_fast_prefill_matches_full_decoder_cache_and_batched_ragged_steps(patching):
    torch.manual_seed(21)
    model = ByteLatentLM(byte_config()).eval()
    source = examples()
    sizes = [0, 7, 63, 64, 91]
    raw = torch.stack([source[i % 2, :91] for i in range(len(sizes))])
    fast = model.init_state(len(sizes), 190, "cpu", torch.float32)
    full = model.init_state(len(sizes), 190, "cpu", torch.float32)
    decoder_lengths = []
    hook = model.decoder.register_forward_pre_hook(lambda _module, inputs: decoder_lengths.append(inputs[0].shape[1]))
    with torch.no_grad():
        fast_logits = model.prefill(fast, raw, lengths=sizes, patching=patching)
    hook.remove()
    assert decoder_lengths and max(decoder_lengths) <= 63
    with torch.no_grad():
        full_logits = model.prefill(full, raw, lengths=sizes, patching=patching, last_only=False)[:, -1:]
        torch.testing.assert_close(fast_logits, full_logits, atol=2e-5, rtol=2e-5)
        for a, b in ((fast.encoder, full.encoder), (fast.decoder, full.decoder), (fast.globals, full.globals)):
            assert a.lengths == b.lengths
            for row, length in enumerate(a.lengths):
                for ak, bk in zip(a.keys, b.keys):
                    torch.testing.assert_close(ak[row, :, :length], bk[row, :, :length], atol=2e-5, rtol=2e-5)
                for av, bv in zip(a.values, b.values):
                    torch.testing.assert_close(av[row, :, :length], bv[row, :, :length], atol=2e-5, rtol=2e-5)
        singles = [model.init_state(1, 190, "cpu", torch.float32) for _ in sizes]
        for row, size in enumerate(sizes):
            single = singles[row].logits
            for index in range(size):
                single = model.step(singles[row], raw[row:row + 1, index])
            torch.testing.assert_close(fast_logits[row:row + 1], single, atol=2e-5, rtol=2e-5)
            for batch_cache, single_cache in ((fast.encoder, singles[row].encoder),
                                               (fast.decoder, singles[row].decoder), (fast.globals, singles[row].globals)):
                length = single_cache.lengths[0]
                assert batch_cache.lengths[row] == length
                for batched_key, single_key in zip(batch_cache.keys, single_cache.keys):
                    torch.testing.assert_close(batched_key[row:row + 1, :, :length], single_key[:, :, :length], atol=2e-5, rtol=2e-5)
        for index in range(75):
            values = torch.tensor([(index * 17 + row * 11) % 256 for row in range(len(sizes))])
            actual = model.step(fast, values)
            expected = torch.cat([model.step(state, values[row:row + 1]) for row, state in enumerate(singles)])
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        assert fast.globals.lengths == [state.globals.lengths[0] for state in singles]


def test_patch_closures_use_one_global_subbatch(monkeypatch):
    model = ByteLatentLM(byte_config()).eval()
    state = model.init_state(4, 32, "cpu", torch.float32)
    with torch.no_grad():
        model.prefill(state, torch.full((4, 2), ord("a")))
        calls = []
        original = model.global_trunk.step_batched
        def capture(x, cache, rows, positions):
            calls.append(list(rows))
            return original(x, cache, rows, positions)
        monkeypatch.setattr(model.global_trunk, "step_batched", capture)
        model.step(state, torch.tensor([ord(" "), ord("a"), ord("!"), ord("b")]))
    assert calls == [[0, 2]]


@pytest.mark.parametrize("patching", ["host", "device"])
def test_ragged_prefill_masks_padding_but_rejects_invalid_real_bytes(patching):
    model = ByteLatentLM(byte_config()).eval()
    state = model.init_state(1, 8, "cpu", torch.float32)
    model.prefill(state, torch.tensor([[ord("a"), 999]]), lengths=[1], patching=patching)
    state = model.init_state(1, 8, "cpu", torch.float32)
    with pytest.raises(ValueError, match="byte"):
        model.prefill(state, torch.tensor([[999, ord("a")]]), lengths=[1], patching=patching)


EXPECTED = {
    "S-BPE50K": (44612608, 18874368, 25731584),
    "S-BPE16K": (27269632, 18874368, 8388608),
    "S-P6": (44884608, 18874368, 25198976),
    "S-P8": (51178112, 25165824, 25198976),
    "M-BPE50K": (95235072, 56623104, 38597376),
    "M-BPE16K": (69220608, 56623104, 12582912),
    "M-P6": (96110016, 56623104, 37798464),
    "M-P8": (108696000, 69206016, 37798464),
    "M-P6-nohash": (58361280, 56623104, 49728),
    "XS-BPE50K": (16275968, 3407872, 12865792),
    "XS-BPE16K": (7604480, 3407872, 4194304),
}


@pytest.mark.parametrize("name", EXPECTED)
def test_exact_parameter_counts_and_plan_totals(name):
    config = preset(name, min_len=4) if "BPE" not in name else preset(name)
    counts = parameter_counts(config)
    assert (counts["total"], counts["global_dense"], counts["embeddings_hash"]) == EXPECTED[name]
    with torch.device("meta"):
        model = BPELM(config) if isinstance(config, BPEConfig) else ByteLatentLM(config)
    assert sum(p.numel() for p in model.parameters()) == counts["total"]
    rounded = {"S-BPE50K": 44.6, "S-BPE16K": 27.3, "S-P6": 44.9, "S-P8": 51.2,
               "M-BPE50K": 95.2, "M-BPE16K": 69.2, "M-P6": 96.1, "M-P8": 108.7, "M-P6-nohash": 58.4,
               "XS-BPE50K": 16.3, "XS-BPE16K": 7.6}
    assert counts["total"] / 1_000_000 == pytest.approx(rounded[name], abs=0.051)


def test_exploratory_pool_balanced_presets():
    base, balanced, wide = (preset(name, min_len=3) for name in ("S-P6", "S-P6-poolbal", "S-P6-poolbal-w256"))
    assert (base.pool_init, balanced.pool_init, wide.pool_init) == ("normal", "balanced", "balanced")
    assert replace(balanced, name="S-P6", pool_init="normal") == base
    assert (wide.local_width, wide.local_heads, wide.local_ffn) == (256, 4, 768) and wide.trunk == base.trunk
    with torch.device("meta"):
        model = ByteLatentLM(wide)
    assert sum(p.numel() for p in model.parameters()) == parameter_counts(wide)["total"]
    torch.manual_seed(0)
    reference = ByteLatentLM(base).state_dict()
    torch.manual_seed(0)
    scaled = ByteLatentLM(balanced).state_dict()
    for name, value in reference.items():
        expected = value / math.sqrt(128) if name == "pool.weight" else value
        torch.testing.assert_close(scaled[name], expected, atol=0, rtol=0)
    with pytest.raises(ValueError, match="pool_init"):
        replace(base, pool_init="other")


def test_exploratory_global_readout_preset_starts_as_s_p6():
    base, readout = preset("S-P6", min_len=3), preset("S-P6-gread", min_len=3)
    assert readout.global_readout and not base.global_readout
    assert replace(readout, name="S-P6", global_readout=False) == base
    assert parameter_counts(readout)["total"] == parameter_counts(base)["total"] + 257 * 512
    with torch.device("meta"):
        model = ByteLatentLM(readout)
    assert sum(p.numel() for p in model.parameters()) == parameter_counts(readout)["total"]
    torch.manual_seed(0)
    reference = ByteLatentLM(base).state_dict()
    torch.manual_seed(0)
    state = ByteLatentLM(readout).state_dict()
    assert set(state) == set(reference) | {"global_readout"}
    for name, value in reference.items():
        torch.testing.assert_close(state[name], value, atol=0, rtol=0)
    assert not state["global_readout"].any()


def test_saved_configs_without_later_fields_normalize_to_defaults():
    from dataclasses import asdict
    from blt.config import normalized_model_dict
    for name in ("S-P6", "S-BPE50K"):
        config = preset(name, min_len=3) if "P6" in name else preset(name)
        old = {key: value for key, value in asdict(config).items() if key not in ("pool_init", "global_readout")}
        assert normalized_model_dict(old) == asdict(config)
    assert normalized_model_dict(asdict(preset("S-P6-gread", min_len=3)))["global_readout"] is True


def test_global_readout_adds_causal_global_logits_and_trains():
    from blt.train import _loss
    config = replace(byte_config(), global_readout=True)
    torch.manual_seed(0)
    base = ByteLatentLM(replace(config, global_readout=False))
    torch.manual_seed(0)
    model = ByteLatentLM(config)
    raw = torch.tensor([list(examples()[0][:150])], dtype=torch.long)
    torch.testing.assert_close(model(raw), base(raw), atol=0, rtol=0)
    hidden = model.hidden(raw)
    assert hidden.shape[-1] == config.local_width + config.trunk.width
    assert model.output_weight.shape == (config.output_classes, config.local_width + config.trunk.width)
    targets = raw.clone()
    _loss(model, raw, targets, chunk_size=64).backward()
    assert model.global_readout.grad is not None and model.global_readout.grad.abs().sum() > 0
    with torch.no_grad():
        model.global_readout.normal_(std=0.5)
    dec, info = model._full_hidden(raw)
    patch_ids = torch.as_tensor(info["patches"].patch_ids, dtype=torch.long)
    cond = info["global_states"][0, patch_ids[0]].unsqueeze(0)
    torch.testing.assert_close(model(raw), model.head(dec) + cond @ model.global_readout.T)
    # The readout sees only completed patches: changing later bytes leaves earlier logits unchanged.
    changed = raw.clone()
    changed[0, 100:] = (changed[0, 100:] + 7) % 256
    torch.testing.assert_close(model(changed)[:, :100], model(raw)[:, :100], atol=0, rtol=0)
    full = F.cross_entropy(model(raw).reshape(-1, config.output_classes).float(), targets.reshape(-1), reduction="sum")
    torch.testing.assert_close(_loss(model, raw, targets, chunk_size=64), full, rtol=1e-5, atol=1e-4)
    with pytest.raises(NotImplementedError, match="global readout"):
        model.init_state(1, 64, "cpu", torch.float32)


def test_uniform_logits_include_eos_probability():
    for classes, expected in ((256, 8.0), (257, math.log2(257))):
        nll = F.cross_entropy(torch.zeros(1, classes, dtype=torch.float64), torch.tensor([0]))
        assert nll.item() / math.log(2) == pytest.approx(expected, abs=1e-12)
    assert byte_config().output_classes == 257


def test_presets_require_calibration_and_record_exact_16k_vocabulary():
    with pytest.raises(ValueError, match="calibrate"):
        ByteLatentLM(preset("S-P6"))
    config = preset("S-BPE16K", vocab_size=16380, bos_id=3)
    assert config.vocab_size == 16380 and config.bos_id == 3
    assert replace(byte_config(), min_len=5).min_len == 5
