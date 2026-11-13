import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import blt.model as model_module
from blt.bench import CUDAGraphPrefill
from blt.config import BPEConfig, ByteConfig, TransformerConfig
from blt.hashes import NGRAMS, hash_history, hashes_at_ends
from blt.model import BPELM, ByteLatentLM, RotaryTables, _rotate, rope
from blt.patcher import max_patch_count, patch_batch, patch_batch_torch


def configs():
    trunk = TransformerConfig(2, 16, 2, 1, 8, 32, max_positions=256)
    return (ByteConfig("byte", trunk, 16, 2, 32, 3, 6, max_len=7, hash_rows=32),
            BPEConfig("bpe", trunk, 40, 0))


def metadata(raw, config, lengths=None):
    host = raw.detach().cpu()
    capacity = max_patch_count(host.shape[1], config.min_len, config.block_size)
    patched = patch_batch(host.numpy(), config.min_len, config.max_len, config.block_size,
                          lengths=None if lengths is None else np.asarray(lengths), max_patches=capacity)
    ends = torch.from_numpy(patched.ends)
    result = {"ends": ends, "patch_ids": torch.from_numpy(patched.patch_ids),
              "counts": torch.from_numpy(patched.counts)}
    if config.hash_enabled:
        result["hash_ids"] = hashes_at_ends(host.long(), ends, config.hash_rows, config.hash_seed)
    return {key: value.to(raw.device) for key, value in result.items()}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64])
def test_rope_tables_match_previous_formula_without_checkpoint_buffers(dtype):
    config = configs()[1].trunk
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        table = RotaryTables(config).to(device)
        x = torch.randn(3, 2, 7, config.head_dim, device=device).to(dtype)
        positions = torch.tensor([0, 1, 5, 63, 64, 127, 255], device=device)
        cos, sin = table.tables(dtype)
        actual = _rotate(x, cos[None, None, positions], sin[None, None, positions])
        torch.testing.assert_close(actual, rope(x, positions, config.rope_theta), atol=0, rtol=0)
        batched = positions[None, :].expand(3, -1).clone()
        actual = _rotate(x, cos[batched][:, None], sin[batched][:, None])
        torch.testing.assert_close(actual, rope(x, batched, config.rope_theta), atol=0, rtol=0)
        assert not table.state_dict()


def test_attention_does_not_recompute_rotary_functions(monkeypatch):
    byte_config, bpe_config = configs()
    models = [ByteLatentLM(byte_config).eval(), BPELM(bpe_config).eval()]
    byte_input = torch.tensor([[1, 2, 32, 4]], dtype=torch.uint8)
    bpe_input = torch.tensor([[0, 1, 2, 3]])
    with monkeypatch.context() as scoped:
        def forbidden(*args, **kwargs):
            raise AssertionError("attention recomputed RoPE tables")
        scoped.setattr(torch.Tensor, "cos", forbidden)
        scoped.setattr(torch.Tensor, "sin", forbidden)
        scoped.setattr(torch.Tensor, "pow", forbidden)
        models[0](byte_input)
        models[1](bpe_input)
        byte_state = models[0].init_state(1, 8, "cpu", torch.float32)
        bpe_state = models[1].init_state(1, 8, "cpu", torch.float32)
        models[0].prefill(byte_state, byte_input, patching="device")
        models[1].prefill(bpe_state, bpe_input)
        models[0].step(byte_state, torch.tensor([5]))
        models[1].step(bpe_state, torch.tensor([5]))


def legacy_attention(self, x, start=0, cache=None, layer=0):
    """The original uncached training attention, retained as a numerical reference."""
    assert cache is None and start == 0
    b, t, _ = x.shape
    c = self.config
    positions = torch.arange(t, device=x.device)
    q = self.q(x).view(b, t, c.q_heads, c.head_dim).transpose(1, 2)
    k = self.k(x).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
    v = self.v(x).view(b, t, c.kv_heads, c.head_dim).transpose(1, 2)
    q, k = rope(q, positions, c.rope_theta), rope(k, positions, c.rope_theta)
    if c.q_heads != c.kv_heads:
        k_attn = k.repeat_interleave(c.q_heads // c.kv_heads, dim=1)
        v_attn = v.repeat_interleave(c.q_heads // c.kv_heads, dim=1)
    else:
        k_attn, v_attn = k, v
    y = F.scaled_dot_product_attention(q, k_attn, v_attn, dropout_p=0.0, is_causal=t > 1)
    return self.out(y.transpose(1, 2).contiguous().view(b, t, c.width)), (k, v)


def legacy_batched_step(self, x, cache, layer, rows, positions, max_context):
    """Original per-step RoPE, cache writes, and attention."""
    b, _, _ = x.shape
    c = self.config
    q = self.q(x).view(b, 1, c.q_heads, c.head_dim).transpose(1, 2)
    k = self.k(x).view(b, 1, c.kv_heads, c.head_dim).transpose(1, 2)
    v = self.v(x).view(b, 1, c.kv_heads, c.head_dim).transpose(1, 2)
    q, k = rope(q, positions[:, None], c.rope_theta), rope(k, positions[:, None], c.rope_theta)
    cache.keys[layer][rows, :, positions, :] = k.squeeze(2)
    cache.values[layer][rows, :, positions, :] = v.squeeze(2)
    keys = cache.keys[layer][:, :, :max_context].index_select(0, rows)
    values = cache.values[layer][:, :, :max_context].index_select(0, rows)
    if c.q_heads != c.kv_heads:
        keys = keys.repeat_interleave(c.q_heads // c.kv_heads, dim=1)
        values = values.repeat_interleave(c.q_heads // c.kv_heads, dim=1)
    mask = torch.arange(max_context, device=x.device)[None, None, None, :] <= positions[:, None, None, None]
    y = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask, dropout_p=0.0, is_causal=False)
    return self.out(y.transpose(1, 2).contiguous().view(b, 1, c.width))


@pytest.mark.parametrize("family", ["byte", "bpe"])
@pytest.mark.parametrize("autocast", [False, True])
def test_training_forward_and_gradients_equal_original_attention(monkeypatch, family, autocast):
    torch.manual_seed(30)
    byte_config, bpe_config = configs()
    model = ByteLatentLM(byte_config) if family == "byte" else BPELM(bpe_config)
    inputs = torch.randint(0, 32, (2, 70))
    reference = copy.deepcopy(model)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual = model(inputs)
    actual.square().sum().backward()
    with monkeypatch.context() as scoped:
        scoped.setattr(model_module.Attention, "forward", legacy_attention)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            expected = reference(inputs)
        expected.square().sum().backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for parameter, original in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, original.grad, rtol=0, atol=0)


def test_init_allocates_only_and_empty_prefix_initializes_lazily(monkeypatch):
    model = ByteLatentLM(configs()[0]).eval()
    with monkeypatch.context() as scoped:
        def forbidden(*args, **kwargs):
            raise AssertionError("init_state ran a transformer")
        scoped.setattr(model.global_trunk, "step_batched", forbidden)
        scoped.setattr(model.decoder, "step_batched", forbidden)
        scoped.setattr(torch.Tensor, "zero_", forbidden)
        state = model.init_state(2, 100, "cpu", torch.float32)
        assert state._logits is None and state._conditioning is None
        assert state._patchers is None and state._hashes is None
        assert state.globals.lengths == [0, 0] and state.decoder.lengths == [0, 0]
    empty = model.prefill(state, torch.empty((2, 0), dtype=torch.uint8))
    assert empty is state.logits
    assert state.globals.lengths == [1, 1] and state.decoder.lengths == [1, 1]
    direct = model.init_state(2, 100, "cpu", torch.float32)
    torch.testing.assert_close(model.step(direct, torch.tensor([3, 4])),
                               model.step(state, torch.tensor([3, 4])))


@pytest.mark.parametrize("patching", ["host", "device", "preprocessed"])
def test_prefill_logits_and_caches_match_original_incremental_path(monkeypatch, patching):
    torch.manual_seed(38)
    model = ByteLatentLM(configs()[0]).eval()
    raw = torch.tensor([list((b"abcd ef!ghij,klmn " * 5)[:70]),
                        list((b"12 3456789:abc de " * 5)[:70])], dtype=torch.uint8)
    reference = model.init_state(2, 90, "cpu", torch.float32)
    with monkeypatch.context() as scoped:
        scoped.setattr(model_module.Attention, "step_batched", legacy_batched_step)
        expected = model.prefill(reference, raw[:, :0])
        for column in range(raw.shape[1]):
            expected = model.step(reference, raw[:, column])
    state = model.init_state(2, 90, "cpu", torch.float32)
    inputs = {"bytes": raw, "patches": metadata(raw, model.config)} if patching == "preprocessed" else raw
    actual = model.prefill(state, inputs, patching=patching)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    for result, old in ((state.encoder, reference.encoder), (state.decoder, reference.decoder),
                        (state.globals, reference.globals)):
        assert result.lengths == old.lengths
        for row, length in enumerate(old.lengths):
            for current, original in zip(result.keys + result.values, old.keys + old.values):
                torch.testing.assert_close(current[row, :, :length], original[row, :, :length],
                                           rtol=2e-5, atol=2e-5)


def test_preprocessed_full_logits_and_training_metadata_contract():
    torch.manual_seed(42)
    model = ByteLatentLM(configs()[0]).eval()
    raw = torch.tensor([list((b"abc, de!fghi jklm " * 4)[:70])] * 2, dtype=torch.uint8)
    lengths = [70, 45]
    patches = metadata(raw, model.config, lengths)
    state = model.init_state(2, 90, "cpu", torch.float32)
    reference = model.init_state(2, 90, "cpu", torch.float32)
    actual = model.prefill(state, {"bytes": raw, "patches": patches}, lengths=lengths,
                           patching="preprocessed", last_only=False)
    expected = model.prefill(reference, raw, lengths=lengths, patching="host", last_only=False)
    for row, length in enumerate(lengths):
        torch.testing.assert_close(actual[row, :length], expected[row, :length], rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(actual[row, -1:], expected[row, -1:], rtol=2e-5, atol=2e-5)
    # Training metadata historically ignored extra keys, including hash_ids.
    correct = model({"bytes": raw, "patches": patches})
    unrelated = dict(patches, hash_ids=torch.full_like(patches["hash_ids"], 999999))
    torch.testing.assert_close(model({"bytes": raw, "patches": unrelated}), correct, rtol=0, atol=0)


@pytest.mark.parametrize("seed", [0, 93])
def test_vector_hashes_match_scalar_for_padded_and_short_windows(seed):
    generator = torch.Generator().manual_seed(seed + 72)
    raw = torch.randint(0, 256, (3, 73), generator=generator, dtype=torch.uint8)
    ends = torch.tensor([[-1, 0, 1, 2, 7, 8, 63, 72],
                         [0, 2, -1, 9, 13, 24, 50, 70],
                         [1, 3, 4, 5, 6, 7, 8, -1]])
    actual = hashes_at_ends(raw, ends, rows=37, seed=seed)
    for row in range(len(raw)):
        for column, end in enumerate(ends[row].tolist()):
            expected = [hash_history(raw[row, :end + 1].tolist(), n, 37, seed) for n in NGRAMS]
            assert actual[row, column].tolist() == expected


@pytest.mark.parametrize("minimum,maximum,block", [(1, 5, 8), (3, 7, 16), (4, 16, 64), (16, 16, 64)])
def test_parallel_patcher_random_and_real_text(minimum, maximum, block):
    rng = np.random.default_rng(minimum * 11 + block)
    text = (Path(__file__).resolve().parents[1] / "discussion" / "round1_full_log.txt").read_bytes()[:377]
    values = rng.integers(0, 256, (4, len(text)), dtype=np.uint8)
    values[0] = np.frombuffer(text, dtype=np.uint8)
    values[1, ::3] = ord(" ")
    lengths = np.asarray([len(text), len(text) - 1, block, 0])
    capacity = max_patch_count(len(text), minimum, block)
    expected = patch_batch(values, minimum, maximum, block, lengths=lengths, max_patches=capacity)
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        actual = patch_batch_torch(torch.from_numpy(values).to(device), minimum, maximum, block,
                                   lengths=torch.from_numpy(lengths).to(device))
        for key in ("ends", "counts", "patch_ids", "boundaries"):
            np.testing.assert_array_equal(getattr(actual, key).cpu().numpy(), getattr(expected, key))


@pytest.mark.parametrize("patching", ["device", "preprocessed"])
def test_resident_prefill_defers_host_routing_and_matches_reference(monkeypatch, patching):
    torch.manual_seed(31)
    model = ByteLatentLM(configs()[0]).eval()
    raw = torch.tensor([list((b"abcdef gh,ijkl!mn " * 5)[:70]),
                        list((b"x!yz 0123/abcd ef " * 5)[:70])], dtype=torch.uint8)
    inputs = {"bytes": raw, "patches": metadata(raw, model.config)} if patching == "preprocessed" else raw
    state = model.init_state(2, 90, "cpu", torch.float32)
    reference = model.init_state(2, 90, "cpu", torch.float32)
    expected = model.prefill(reference, raw, patching="host")
    with monkeypatch.context() as scoped:
        def forbidden(*args, **kwargs):
            raise AssertionError("resident prefill read device values on the host")
        scoped.setattr(torch.Tensor, "cpu", forbidden)
        scoped.setattr(torch.Tensor, "item", forbidden)
        scoped.setattr(torch.Tensor, "tolist", forbidden)
        scoped.setattr(model_module.IncrementalPatcher, "__init__", forbidden)
        scoped.setattr(model_module.RollingHashes, "__init__", forbidden)
        # The explicitly requested upper-bound path retains the original
        # no-host-read contract; exact device patching now reads counts.max().
        actual = model.prefill(state, inputs, patching=patching, patch_capacity="upper_bound")
    assert state._routing is not None
    assert state._patchers is None and state._hashes is None
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert state.globals.lengths == reference.globals.lengths
    assert state._routing is None
    torch.testing.assert_close(model.step(state, torch.tensor([32, 33])),
                               model.step(reference, torch.tensor([32, 33])), rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("patching", ["device", "preprocessed"])
def test_resident_prefill_with_padding_wider_than_cache(patching):
    torch.manual_seed(41)
    model = ByteLatentLM(configs()[0]).eval()
    raw = torch.tensor([list((b"a,bc def ghi! jkl " * 4)[:70])] * 2, dtype=torch.uint8)
    lengths = [4, 5]
    inputs = {"bytes": raw, "patches": metadata(raw, model.config, lengths)} if patching == "preprocessed" else raw
    state = model.init_state(2, 10, "cpu", torch.float32)
    reference = model.init_state(2, 10, "cpu", torch.float32)
    actual = model.prefill(state, inputs, lengths=lengths, patching=patching)
    expected = model.prefill(reference, raw, lengths=lengths, patching="host")
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert state.globals.lengths == reference.globals.lengths
    torch.testing.assert_close(model.step(state, torch.tensor([32, 33])),
                               model.step(reference, torch.tensor([32, 33])), rtol=2e-5, atol=2e-5)


def test_zero_length_ragged_prefill_can_be_followed_by_host_prefill():
    model = ByteLatentLM(configs()[0]).eval()
    raw = torch.full((2, 12), ord("a"), dtype=torch.uint8)
    state = model.init_state(2, 30, "cpu", torch.float32)
    model.prefill(state, raw, lengths=[0, 0], patching="device")
    actual = model.prefill(state, raw, lengths=[5, 4], patching="host")
    reference = model.init_state(2, 30, "cpu", torch.float32)
    expected = model.prefill(reference, raw, lengths=[5, 4], patching="host")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert state.globals.lengths == reference.globals.lengths
    assert [p.position for p in state.patchers] == [5, 4]


@pytest.mark.parametrize("patching", ["host", "device"])
def test_rejected_prefill_preserves_an_initialized_empty_state(patching):
    model = ByteLatentLM(configs()[0]).eval()
    state = model.init_state(1, 10, "cpu", torch.float32)
    reference = model.init_state(1, 10, "cpu", torch.float32)
    model.prefill(state, torch.empty((1, 0), dtype=torch.long))
    model.prefill(reference, torch.empty((1, 0), dtype=torch.long))
    with pytest.raises(ValueError, match="byte"):
        model.prefill(state, torch.tensor([[999]]), patching=patching)
    torch.testing.assert_close(model.step(state, torch.tensor([32])),
                               model.step(reference, torch.tensor([32])), rtol=0, atol=0)


def test_transformer_fullgraph_trace_uses_shared_rope_buffers():
    torch.manual_seed(39)
    model = BPELM(configs()[1]).eval()
    compiled = copy.deepcopy(model)
    compiled.trunk.forward = torch.compile(compiled.trunk.forward, backend="eager", fullgraph=True)
    raw = torch.randint(0, 40, (2, 31))
    torch.testing.assert_close(compiled(raw), model(raw), rtol=0, atol=0)
    state = compiled.init_state(2, 40, "cpu", torch.float32)
    reference = model.init_state(2, 40, "cpu", torch.float32)
    torch.testing.assert_close(compiled.prefill(state, raw), model.prefill(reference, raw), rtol=0, atol=0)


def test_compiled_batched_tensor_step_matches_eager_prefill():
    torch.manual_seed(40)
    model = ByteLatentLM(configs()[0]).eval()
    compiled = copy.deepcopy(model)
    compiled.decoder.step_batched_tensor = torch.compile(compiled.decoder.step_batched_tensor,
                                                         backend="eager", fullgraph=True)
    raw = torch.tensor([list((b"a bc!def, ghijk lm " * 4)[:65])], dtype=torch.uint8)
    expected_state = model.init_state(1, 80, "cpu", torch.float32)
    actual_state = compiled.init_state(1, 80, "cpu", torch.float32)
    expected = model.prefill(expected_state, raw, patching="device")
    actual = compiled.prefill(actual_state, raw, patching="device")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph prefill requires CUDA")
@pytest.mark.parametrize("family,patching", [("bpe", "device"), ("byte", "device"),
                                               ("byte", "preprocessed"), ("byte_nohash", "preprocessed")])
def test_cuda_graph_replay_matches_eager_caches_and_next_step(family, patching):
    torch.manual_seed(32)
    byte_config, bpe_config = configs()
    if family == "byte_nohash":
        byte_config = replace(byte_config, hash_enabled=False)
    is_byte = family != "bpe"
    model = (ByteLatentLM(byte_config) if is_byte else BPELM(bpe_config)).cuda().eval()
    if family == "bpe":
        examples = [torch.randint(0, 40, (2, 70), device="cuda") for _ in range(2)]
    else:
        examples = [torch.tensor([list((a * 5)[:70]), list((b * 5)[:70])], dtype=torch.uint8, device="cuda")
                    for a, b in ((b"abcd efgh,ijkl!mn ", b"!  !  !  !  !  !  "),
                                 (b"1234567890abcdef  ", b"one,two three.    "))]

    def graph_input(raw):
        if is_byte and patching == "preprocessed":
            return {"bytes": raw, "patches": metadata(raw, byte_config)}
        return raw

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # These inputs have different patch counts; the upper-bound comparison
        # path deliberately keeps one graph shape for both.
        graph = CUDAGraphPrefill(model, graph_input(examples[0]), max_len=90, patching=patching,
                                 patch_capacity="upper_bound")
        for raw in examples:
            eager = model.init_state(2, 90, "cuda", torch.bfloat16)
            expected = (model.prefill(eager, raw, patching="host") if is_byte else model.prefill(eager, raw))
            captured, actual = graph.replay(graph_input(raw))
            torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
            pairs = ((captured.encoder, eager.encoder), (captured.decoder, eager.decoder),
                     (captured.globals, eager.globals)) if is_byte else ((captured.cache, eager.cache),)
            for actual_cache, expected_cache in pairs:
                if is_byte:
                    assert actual_cache.lengths == expected_cache.lengths
                    lengths = expected_cache.lengths
                else:
                    assert actual_cache.length == expected_cache.length
                    lengths = [expected_cache.length] * 2
                for row, length in enumerate(lengths):
                    for actual_tensor, expected_tensor in zip(actual_cache.keys + actual_cache.values,
                                                              expected_cache.keys + expected_cache.values):
                        torch.testing.assert_close(actual_tensor[row, :, :length], expected_tensor[row, :, :length],
                                                   rtol=2e-2, atol=2e-2)
            following = torch.tensor([12, 13], device="cuda")
            torch.testing.assert_close(model.step(captured, following), model.step(eager, following),
                                       rtol=2e-2, atol=2e-2)
