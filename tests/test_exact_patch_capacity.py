from dataclasses import replace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from blt.bench import CUDAGraphPatcher, CUDAGraphPrefill, InferenceCase, benchmark
from blt.config import BPEConfig, ByteConfig, TransformerConfig
from blt.eval_io import EvaluationModel
from blt.hashes import hashes_at_ends
from blt.model import BPELM, ByteLatentLM
from blt.patcher import max_patch_count, patch_batch, trim_patch_metadata


def byte_config(minimum=3, hashes=True):
    trunk = TransformerConfig(2, 16, 2, 1, 8, 32, max_positions=256)
    return ByteConfig("test", trunk, 16, 2, 32, minimum, 6, max_len=12,
                      hash_enabled=hashes, hash_rows=32)


def sample_bytes():
    samples = (b"An ordinary text, with spaces and punctuation! 12345. ",
               b"abcdefghijklmnopqrstuvwxyz0123456789 short words ",
               b" ! ! ! ! punctuation! and a considerably longer suffix ")
    return torch.tensor([list((sample * 3)[:78]) for sample in samples], dtype=torch.uint8)


def host_metadata(raw, config, lengths=None):
    host = raw.cpu()
    capacity = max_patch_count(host.shape[1], config.min_len, config.block_size)
    patches = patch_batch(host.numpy(), config.min_len, config.max_len, config.block_size,
                          lengths=None if lengths is None else np.asarray(lengths), max_patches=capacity)
    ends = torch.from_numpy(patches.ends)
    result = {"ends": ends, "patch_ids": torch.from_numpy(patches.patch_ids),
              "counts": torch.from_numpy(patches.counts)}
    if config.hash_enabled:
        result["hash_ids"] = hashes_at_ends(host.long(), ends, config.hash_rows, config.hash_seed)
    return {key: value.to(raw.device) for key, value in result.items()}


def assert_byte_states_close(actual, expected, *, atol=2e-5, rtol=2e-5):
    torch.testing.assert_close(actual.conditioning, expected.conditioning, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual.logits, expected.logits, atol=atol, rtol=rtol)
    for left, right in ((actual.globals, expected.globals), (actual.encoder, expected.encoder),
                        (actual.decoder, expected.decoder)):
        assert left.lengths == right.lengths
        for row, length in enumerate(right.lengths):
            for a, b in zip(left.keys + left.values, right.keys + right.values):
                torch.testing.assert_close(a[row, :, :length], b[row, :, :length], atol=atol, rtol=rtol)


@pytest.mark.parametrize("minimum", [3, 5])
@pytest.mark.parametrize("hashes", [False, True])
@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("last_only", [False, True])
def test_preprocessed_capacities_match_logits_caches_and_steps(minimum, hashes, ragged, last_only):
    torch.manual_seed(711)
    config = byte_config(minimum, hashes)
    model = ByteLatentLM(config).eval()
    raw = sample_bytes()
    lengths = [78, 41, 0] if ragged else None
    upper = host_metadata(raw, config, lengths)
    k = int(upper["counts"].max())
    limit = upper["ends"].shape[1]
    assert 0 < k < limit
    seen = []
    handle = model.global_trunk.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape[1]))
    states, outputs = [], []
    try:
        for capacity in (k, (k + limit) // 2, limit):
            state = model.init_state(3, 90, "cpu", torch.float32)
            output = model.prefill(state, {"bytes": raw, "patches": trim_patch_metadata(upper, capacity)},
                                   lengths=lengths, patching="preprocessed", last_only=last_only)
            states.append(state)
            outputs.append(output)
        assert seen == [k + 1, (k + limit) // 2 + 1, limit + 1]
    finally:
        handle.remove()
    for state, output in zip(states[:-1], outputs[:-1]):
        torch.testing.assert_close(output, outputs[-1], atol=2e-5, rtol=2e-5)
        assert_byte_states_close(state, states[-1])
    following = torch.tensor([32, 33, 65])
    next_outputs = [model.step(state, following) for state in states]
    for state, output in zip(states[:-1], next_outputs[:-1]):
        torch.testing.assert_close(output, next_outputs[-1], atol=2e-5, rtol=2e-5)
        assert_byte_states_close(state, states[-1])


@pytest.mark.parametrize("hashes", [False, True])
def test_zero_capacity_and_metadata_shape_validation(hashes):
    config = byte_config(hashes=hashes)
    model = ByteLatentLM(config).eval()
    raw = torch.tensor([[97, 98], [97, 98]], dtype=torch.uint8)
    metadata = host_metadata(raw, config, [2, 0])
    exact = trim_patch_metadata(metadata, 0)
    state, reference = (model.init_state(2, 8, "cpu", torch.float32) for _ in range(2))
    logits = model.prefill(state, {"bytes": raw, "patches": exact}, lengths=[2, 0], patching="preprocessed")
    expected = model.prefill(reference, raw, lengths=[2, 0], patching="host")
    torch.testing.assert_close(logits, expected, atol=2e-5, rtol=2e-5)
    assert_byte_states_close(state, reference)
    torch.testing.assert_close(model.step(state, torch.tensor([32, 32])),
                               model.step(reference, torch.tensor([32, 32])), atol=2e-5, rtol=2e-5)
    broken = dict(exact, ends=torch.zeros((2, max_patch_count(2, config.min_len) + 1), dtype=torch.long))
    with pytest.raises(ValueError, match="patch metadata"):
        model.prefill(model.init_state(2, 8, "cpu", torch.float32), {"bytes": raw, "patches": broken},
                      patching="preprocessed")
    broken = dict(exact, hash_ids=torch.zeros((2, 1, 6), dtype=torch.long))
    with pytest.raises(ValueError, match="hash metadata"):
        model.prefill(model.init_state(2, 8, "cpu", torch.float32), {"bytes": raw, "patches": broken},
                      patching="preprocessed")


@pytest.mark.parametrize("minimum", [3, 5])
@pytest.mark.parametrize("hashes", [False, True])
def test_device_two_stage_matches_host_and_reads_count_once(minimum, hashes, monkeypatch):
    config = byte_config(minimum, hashes)
    model = ByteLatentLM(config).eval()
    raw, lengths = sample_bytes(), [78, 50, 0]
    host = host_metadata(raw, config, lengths)
    original_item = torch.Tensor.item
    reads = []

    def tracked_item(tensor, *args, **kwargs):
        reads.append(tensor.shape)
        return original_item(tensor, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(torch.Tensor, "item", tracked_item)
        metadata = model.device_patch_metadata(raw, lengths=lengths)
        assert reads == []
        state = model.init_state(3, 90, "cpu", torch.float32)
        actual = model.prefill(state, raw, lengths=lengths, patching="device")
        assert reads == [torch.Size([])]
        reads.clear()
        upper_state = model.init_state(3, 90, "cpu", torch.float32)
        upper_logits = model.prefill(upper_state, raw, lengths=lengths, patching="device",
                                    patch_capacity="upper_bound")
        assert reads == []
    for key in host:
        torch.testing.assert_close(metadata[key], host[key], rtol=0, atol=0)
    reference = model.init_state(3, 90, "cpu", torch.float32)
    expected = model.prefill(reference, raw, lengths=lengths, patching="host")
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(upper_logits, expected, atol=2e-5, rtol=2e-5)
    assert_byte_states_close(state, reference)
    assert_byte_states_close(upper_state, reference)
    following = torch.tensor([32, 33, 65])
    result, upper_result, expected = (model.step(s, following) for s in (state, upper_state, reference))
    torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(upper_result, expected, atol=2e-5, rtol=2e-5)


def test_benchmark_prepares_exact_metadata_and_exposes_upper_bound_option():
    config = byte_config()
    model = ByteLatentLM(config).eval()
    bundle = EvaluationModel(model, config, None, torch.device("cpu"), {"kind": "byte"})
    prompt = b"A small ordinary prompt, for testing exact capacity."
    exact = InferenceCase(bundle, prompt, 3, prepare_preprocessed=True, patch_capacity="exact")
    upper = InferenceCase(bundle, prompt, 3, prepare_preprocessed=True, patch_capacity="upper_bound")
    assert exact.patch_metadata["ends"].shape[1] == int(exact.patch_metadata["counts"].max())
    assert upper.patch_metadata["ends"].shape[1] == max_patch_count(upper.device_bytes.shape[1], config.min_len, config.block_size)
    assert exact.patch_metadata["ends"].shape[1] < upper.patch_metadata["ends"].shape[1]
    for case in (exact, upper):
        _, host_logits, _ = case.prefill(resident=False)
        _, device_logits, _ = case.prefill(resident=True)
        _, preprocessed_logits, _ = case.prefill(resident=True, preprocessed=True)
        torch.testing.assert_close(device_logits, host_logits, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(preprocessed_logits, host_logits, atol=2e-5, rtol=2e-5)
    result = benchmark([bundle], [prompt], mode="prefill", warmup=0, repetitions=1, patch_capacity="upper_bound")
    assert result["patch_capacity"] == "upper_bound"
    assert "device-to-host" in result["prefill_definitions"]["gpu_resident"]
    assert "transfers" in result["prefill_definitions"]["end_to_end"]
    with pytest.raises(ValueError, match="patch_capacity"):
        benchmark([bundle], [prompt], mode="prefill", warmup=0, repetitions=1, patch_capacity="bad")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph prefill requires CUDA")
@pytest.mark.parametrize("family", ["bpe", "byte", "byte_nohash"])
@pytest.mark.parametrize("variant", ["gpu_resident", "gpu_resident_preprocessed", "end_to_end"])
def test_cuda_exact_graph_matches_eager_and_next_decode(family, variant, monkeypatch):
    torch.manual_seed(712)
    prompt = b"A short graph prompt, with spaces!"
    if family == "bpe":
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator([prompt.decode()], trainers.BpeTrainer(
            vocab_size=280, special_tokens=["<|bos|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False))
        config = BPEConfig("bpe", byte_config().trunk, tokenizer.get_vocab_size(), tokenizer.token_to_id("<|bos|>"))
        model = BPELM(config)
    else:
        tokenizer = None
        config = replace(byte_config(5), hash_enabled=family == "byte")
        model = ByteLatentLM(config)
    model = model.cuda().eval()
    bundle = EvaluationModel(model, config, tokenizer, torch.device("cuda"), {"kind": family})
    case = InferenceCase(bundle, prompt, 2, output_bytes=4,
                         prepare_resident=variant != "end_to_end",
                         prepare_preprocessed=variant == "gpu_resident_preprocessed", patch_capacity="exact")
    warm_replays = {"patcher": 0, "trunk": 0}
    original_patcher_replay, original_trunk_replay = CUDAGraphPatcher.replay, CUDAGraphPrefill.replay

    def track_patcher(self, *args, **kwargs):
        warm_replays["patcher"] += 1
        return original_patcher_replay(self, *args, **kwargs)

    def track_trunk(self, *args, **kwargs):
        warm_replays["trunk"] += 1
        return original_trunk_replay(self, *args, **kwargs)

    monkeypatch.setattr(CUDAGraphPatcher, "replay", track_patcher)
    monkeypatch.setattr(CUDAGraphPrefill, "replay", track_trunk)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        eager, expected, _ = case.prefill(resident=variant != "end_to_end", headroom=4,
                                          preprocessed=variant == "gpu_resident_preprocessed")
        inputs = (case.host_inputs() if variant == "end_to_end" else
                  case.resident_inputs(preprocessed=variant == "gpu_resident_preprocessed"))
        patcher = None
        if family != "bpe" and variant == "gpu_resident":
            patcher = CUDAGraphPatcher(model, inputs, warmup=1)
            assert warm_replays["patcher"] >= 1
            metadata = patcher.replay(inputs)
            k = int(metadata["counts"].max().item())
            inputs = {**inputs, "patches": trim_patch_metadata(metadata, k)}
        if family != "bpe":
            assert inputs["patches"]["ends"].shape[1] == int(inputs["patches"]["counts"].max().item())
        source = inputs["tokens"] if family == "bpe" else inputs["bytes"]
        graph = CUDAGraphPrefill(model, inputs, max_len=source.shape[1] + 4,
                                 patching="preprocessed" if family != "bpe" else "device", warmup=1)
        assert warm_replays["trunk"] >= 1
        if patcher is not None:
            raw = {key: value for key, value in inputs.items() if key != "patches"}
            metadata = patcher.replay(raw)
            inputs = {**raw, "patches": trim_patch_metadata(metadata, k)}
        captured, actual = graph.replay(inputs)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(case.distribution(actual), case.distribution(expected), atol=2e-2, rtol=2e-2)
        if family == "bpe":
            lengths = [eager.cache.length] * 2
            assert captured.cache.lengths == lengths
            for row, length in enumerate(lengths):
                for a, b in zip(captured.cache.keys + captured.cache.values, eager.cache.keys + eager.cache.values):
                    torch.testing.assert_close(a[row, :, :length], b[row, :, :length], atol=2e-2, rtol=2e-2)
        else:
            assert_byte_states_close(captured, eager, atol=2e-2, rtol=2e-2)
        following = torch.tensor([12, 13], device="cuda")
        torch.testing.assert_close(model.step(captured, following), model.step(eager, following), atol=2e-2, rtol=2e-2)
