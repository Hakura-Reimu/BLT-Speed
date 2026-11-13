from contextlib import nullcontext
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

import blt.bench as bench_module
from blt.bench import (CUDAGraphPatcher, CUDAGraphPrefill, GraphCache, InferenceCase,
                       _capture_missing, _run_graph_or_eager, benchmark)
from blt.config import BPEConfig, ByteConfig, TransformerConfig
from blt.eval_io import EvaluationModel
from blt.hashes import hashes_at_ends
from blt.model import BPELM, BatchedKVCache, ByteLatentLM
from blt.patcher import max_patch_count, patch_batch, patch_capacity, trim_patch_metadata


def config_byte(minimum=3, hashes=True):
    trunk = TransformerConfig(2, 16, 2, 1, 8, 32, max_positions=384)
    return ByteConfig("bucket-test", trunk, 16, 2, 32, minimum, 6, max_len=12,
                      hash_enabled=hashes, hash_rows=32)


def config_bpe():
    return BPEConfig("bucket-bpe", TransformerConfig(2, 16, 2, 1, 8, 32, max_positions=128), 64, 0)


def compare_batched_cache(batch, row, single, length, *, atol=2e-5, rtol=2e-5):
    assert batch.lengths[row] == length
    for actual, expected in zip(batch.keys + batch.values, single.keys + single.values):
        torch.testing.assert_close(actual[row, :, :length], expected[0, :, :length], atol=atol, rtol=rtol)


def compare_byte_states(actual, expected, *, atol=2e-5, rtol=2e-5):
    torch.testing.assert_close(actual.conditioning, expected.conditioning, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual.logits, expected.logits, atol=atol, rtol=rtol)
    for left, right in ((actual.globals, expected.globals), (actual.encoder, expected.encoder),
                        (actual.decoder, expected.decoder)):
        assert left.lengths == right.lengths
        for row, length in enumerate(right.lengths):
            compare_batched_cache(left, row, right[row], length, atol=atol, rtol=rtol)


@pytest.mark.parametrize("tensor_lengths", [False, True])
def test_bpe_ragged_padded_prefill_and_steps_match_individual_rows(tensor_lengths):
    torch.manual_seed(1101)
    model = BPELM(config_bpe()).eval()
    lengths = [1, 17, 45]
    tokens = torch.zeros((3, 64), dtype=torch.long)
    for row, length in enumerate(lengths):
        tokens[row, :length] = (torch.arange(length) * (row + 2) + row) % 64
    batch = model.init_state(3, 96, "cpu", torch.float32)
    logits = model.prefill(batch, tokens, lengths=torch.tensor(lengths) if tensor_lengths else lengths)
    assert isinstance(batch.cache, BatchedKVCache)
    singles = []
    for row, length in enumerate(lengths):
        single = model.init_state(1, 96, "cpu", torch.float32)
        expected = model.prefill(single, tokens[row:row + 1, :length])
        singles.append(single)
        torch.testing.assert_close(logits[row:row + 1], expected, atol=2e-5, rtol=2e-5)
        compare_batched_cache(batch.cache, row, single.cache, length)
    for step in range(2):
        following = torch.tensor([7 + step, 11 + step, 19 + step])
        actual = model.step(batch, following)
        for row, single in enumerate(singles):
            expected = model.step(single, following[row:row + 1])
            torch.testing.assert_close(actual[row:row + 1], expected, atol=2e-5, rtol=2e-5)
            compare_batched_cache(batch.cache, row, single.cache, lengths[row] + step + 1)


def byte_inputs_and_metadata(config, device="cpu"):
    lengths = [311, 233, 97]
    patterns = (b"abcdefg ", b"different words. ", b"third line! ")
    data = torch.zeros((3, 320), dtype=torch.uint8)
    for row, (pattern, length) in enumerate(zip(patterns, lengths)):
        data[row, :length] = torch.tensor(list((pattern * 320)[:length]), dtype=torch.uint8)
    upper = max_patch_count(data.shape[1], config.min_len, config.block_size)
    patched = patch_batch(data.numpy(), config.min_len, config.max_len, config.block_size,
                          lengths=np.asarray(lengths), max_patches=upper)
    ends = torch.from_numpy(patched.ends)
    metadata = {"ends": ends, "patch_ids": torch.from_numpy(patched.patch_ids),
                "counts": torch.from_numpy(patched.counts)}
    if config.hash_enabled:
        metadata["hash_ids"] = hashes_at_ends(data.long(), ends, config.hash_rows, config.hash_seed)
    return data.to(device), lengths, {key: value.to(device) for key, value in metadata.items()}


@pytest.mark.parametrize("minimum", [3, 5])
@pytest.mark.parametrize("hashes", [False, True])
def test_byte_exact_bucket_and_upper_match_host_and_cached_steps(minimum, hashes):
    torch.manual_seed(1102)
    config = config_byte(minimum, hashes)
    model = ByteLatentLM(config).eval()
    raw, lengths, metadata = byte_inputs_and_metadata(config)
    actual, upper = int(metadata["counts"].max()), metadata["ends"].shape[1]
    capacities = [patch_capacity(actual, upper, policy) for policy in ("exact", "bucket64", "upper_bound")]
    assert capacities[0] < capacities[1] < capacities[2]
    reference = model.init_state(3, 330, "cpu", torch.float32)
    expected = model.prefill(reference, raw, lengths=lengths, patching="host")
    states = []
    for capacity, use_tensor in zip(capacities, (False, True, False)):
        state = model.init_state(3, 330, "cpu", torch.float32)
        payload = {"bytes": raw, "patches": trim_patch_metadata(metadata, capacity)}
        result = model.prefill(state, payload, lengths=torch.tensor(lengths) if use_tensor else lengths,
                               patching="preprocessed")
        torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-5)
        compare_byte_states(state, reference)
        states.append(state)
    device_metadata = model.device_patch_metadata(raw, lengths=torch.tensor(lengths))
    for key in metadata:
        torch.testing.assert_close(device_metadata[key], metadata[key], atol=0, rtol=0)
    device_state = model.init_state(3, 330, "cpu", torch.float32)
    device_logits = model.prefill(device_state, raw, lengths=lengths, patching="device", patch_capacity="bucket64")
    torch.testing.assert_close(device_logits, expected, atol=2e-5, rtol=2e-5)
    compare_byte_states(device_state, reference)
    states.append(device_state)
    for following in (torch.tensor([32, 65, 33]), torch.tensor([66, 67, 68])):
        expected = model.step(reference, following)
        for state in states:
            torch.testing.assert_close(model.step(state, following), expected, atol=2e-5, rtol=2e-5)
            compare_byte_states(state, reference)


def test_graph_miss_runs_inside_interval_and_lru_is_bounded():
    timeline = []
    cache = GraphCache(2)
    class Runner:
        def replay(self, inputs):
            timeline.append("replay")
            return inputs * 2
    def eager():
        timeline.append("eager")
        return 7
    timeline.append("start")
    result, hit = _run_graph_or_eager(cache, ("trunk", 1), 3, eager)
    timeline.append("stop")
    cache.record_trial(hit)
    assert (result, hit, cache.hits, cache.misses) == (7, False, 0, 1)
    def capture():
        timeline.append("capture")
        return Runner()
    assert _capture_missing(cache, ("trunk", 1), capture, torch.device("cpu"), phase="miss", trial=0)
    assert timeline == ["start", "eager", "stop", "capture"]
    assert cache.captures[0]["phase"] == "miss" and cache.captures[0]["seconds"] >= 0
    result, hit = _run_graph_or_eager(cache, ("trunk", 1), 3, eager)
    cache.record_trial(hit)
    assert (result, cache.hits, cache.misses) == (6, 1, 1)
    cache.put(("patcher", 2), Runner())
    cache.get(("trunk", 1))
    assert cache.put(("trunk", 3), Runner()) == ("patcher", 2)
    assert len(cache.entries) == 2 and cache.get(("patcher", 2)) is None
    def replacement():
        assert len(cache.entries) == 1  # The old graph is dropped before capture starts.
        return Runner()
    assert _capture_missing(cache, ("trunk", 4), replacement, torch.device("cpu"), phase="miss", trial=1)
    assert len(cache.entries) == 2 and cache.captures[-1]["evicted"] is not None


def test_benchmark_distinct_rows_and_separate_conversion(monkeypatch):
    config = config_byte(hashes=False)
    bundle = EvaluationModel(ByteLatentLM(config).eval(), config, None, torch.device("cpu"), {"kind": "byte"})
    prompts = [b"warm one", b"warm row two", b"timed one!", b"a different timed row"]
    ticks = [0.0]
    def clock():
        ticks[0] += 0.001
        return ticks[0]
    original = InferenceCase.distribution
    def expensive_conversion(case, logits):
        ticks[0] += 1.0
        return original(case, logits)
    monkeypatch.setattr(bench_module, "time", SimpleNamespace(perf_counter=clock))
    monkeypatch.setattr(InferenceCase, "distribution", expensive_conversion)
    result = benchmark([bundle], prompts, mode="prefill", batch=2, warmup=1, repetitions=1)
    json.dumps(result)
    assert result["capacity"] == "bucket64"
    for variant in result["variants"].values():
        row = variant["models"][0]
        assert row["bytes"] == [len(prompts[2]) + len(prompts[3])]
        assert row["prompt_bytes"] == [(len(prompts[2]) + len(prompts[3])) / 2]
        assert row["prompt_bytes_per_row"] == [[len(prompts[2]), len(prompts[3])]]
        assert len(row["prompt_sha256"][0]) == 64 and len(set(row["prompt_sha256_per_row"][0])) == 2
        assert row["seconds"][0] < 0.1 and row["conversion_seconds"][0] > 1.0
        assert row["gpu_event_seconds"] == [None]
        assert row["capacities"][0]["actual"] <= row["capacities"][0]["padded"]
    with pytest.raises(ValueError, match="different"):
        benchmark([bundle], [b"same", b"same"], mode="prefill", batch=2, warmup=0, repetitions=1)
    with pytest.raises(ValueError, match="calibration"):
        benchmark([bundle], [b"same", b"same"], mode="prefill", batch=1, warmup=1, repetitions=1)


def test_bpe_benchmark_uses_ragged_row_lengths_and_bucketed_tokens():
    prompts = [b"A short row.", b"This second row is considerably longer!"]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator([prompt.decode() for prompt in prompts], trainers.BpeTrainer(
        vocab_size=275, special_tokens=["<|bos|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False))
    config = BPEConfig("bucket-bpe", config_bpe().trunk, tokenizer.get_vocab_size(), tokenizer.token_to_id("<|bos|>"))
    bundle = EvaluationModel(BPELM(config).eval(), config, tokenizer, torch.device("cpu"), {"kind": "bpe"})
    case = InferenceCase(bundle, prompts, 2, prepare_preprocessed=True)
    assert case.real_lengths[0] != case.real_lengths[1]
    assert case.tokens.shape == (2, 64)
    result = benchmark([bundle], prompts, mode="prefill", batch=2, warmup=0, repetitions=1)
    for variant in result["variants"].values():
        row = variant["models"][0]
        assert row["cache"][0]["global_positions"] == case.real_lengths
        assert row["capacities"][0]["actual"] == max(case.real_lengths)
        assert row["capacities"][0]["padded"] == 64


def test_registered_fast_encoder_gives_identical_host_tokens(monkeypatch):
    prompts = [b"A short row.", b"This second row is considerably longer!"]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator([prompt.decode() for prompt in prompts], trainers.BpeTrainer(
        vocab_size=275, special_tokens=["<|bos|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False))
    config = BPEConfig("fast-bpe", config_bpe().trunk, tokenizer.get_vocab_size(), tokenizer.token_to_id("<|bos|>"))
    bundle = EvaluationModel(BPELM(config).eval(), config, tokenizer, torch.device("cpu"),
                             {"kind": "bpe", "tokenizer_sha256": "fake-sha"})
    reference = InferenceCase(bundle, prompts, 2, prepare_resident=False).prepare_host()
    monkeypatch.setattr(bench_module, "FAST_ENCODERS", {})
    bench_module.register_fast_encoder("fake-sha", "numpy-wrapper", lambda text: np.asarray(
        tokenizer.encode(text, add_special_tokens=False).ids, dtype=np.uint32))
    case = InferenceCase(bundle, prompts, 2, prepare_resident=False)
    fast = case.prepare_host()
    assert case.encoder_name == "numpy-wrapper" and case.encode_seconds > 0
    assert torch.equal(fast["tokens"], reference["tokens"]) and torch.equal(fast["lengths"], reference["lengths"])


def test_byte_level_tiktoken_conversion_matches_hf(tmp_path):
    pytest.importorskip("tiktoken")
    from blt.fast_tokenizer import hf_byte_level_numpy_encoder
    corpus = ["The quick brown fox jumps over the lazy dog.", "It's 2024; numbers 12345 and  spaces\n\nnewlines.",
              "Unicode: café, naïve, 中文, \U0001f642."] * 20
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<|bos|>", "<|eos|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False))
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    encode = hf_byte_level_numpy_encoder(path)
    for text in [*corpus[:3], "unseen words, quickly!", "a<|bos|>b<|eos|>", "  \t trailing  ", "é \U0001f600"]:
        assert list(encode(text)) == tokenizer.encode(text, add_special_tokens=False).ids


GPT2_DIR = Path.home() / "blt/models/gpt2-tokenizer"


@pytest.mark.skipif(not (GPT2_DIR / "merges.txt").exists(), reason="local GPT-2 tokenizer files not present")
def test_local_tiktoken_gpt2_matches_hf_including_literal_special_tokens():
    pytest.importorskip("tiktoken")
    from blt.fast_tokenizer import gpt2_numpy_encoder
    hf = Tokenizer.from_file(str(GPT2_DIR / "tokenizer.json"))
    encode = gpt2_numpy_encoder(GPT2_DIR)
    for text in ("Hello world", "x<|endoftext|>y", "  spaced\n\n\ttext  ", "It's IT'S", "中文 \U0001f642 é"):
        assert list(encode(text)) == hf.encode(text, add_special_tokens=False).ids


def test_benchmark_cli_loads_enough_distinct_rows(monkeypatch, capsys):
    from scripts import bench_infer
    requested, calls = [], []
    monkeypatch.setattr(bench_infer, "verify_supply", lambda path: None)
    monkeypatch.setattr(bench_infer, "load_evaluation_model", lambda **kwargs: SimpleNamespace(model=object()))
    def prompts(supply, size, count, seed):
        requested.append(count)
        return [f"prompt {index}".encode() for index in range(count)]
    def run(bundles, values, **kwargs):
        calls.append((len(values), kwargs["batch"], kwargs["patch_capacity"], kwargs["graph_cache_size"]))
        return {"mode": kwargs["mode"]}
    monkeypatch.setattr(bench_infer, "val_prompts", prompts)
    monkeypatch.setattr(bench_infer, "benchmark", run)
    monkeypatch.setattr(bench_infer, "atomic_json", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["bench_infer", "--preset", "S-P6", "--supply", ".", "--output", "out.json",
                                    "--mode", "prefill", "--prompt-kib", "1", "--batches", "1", "4",
                                    "--capacity", "exact", "--graph-cache-size", "4"])
    bench_infer.main()
    capsys.readouterr()
    assert requested == [160]
    assert calls == [(160, 1, "exact", 4), (160, 4, "exact", 4)]


def test_validation_prompt_loader_avoids_duplicate_contents(monkeypatch):
    data = "".join(f"{index:05d}:" for index in range(200)).encode()
    monkeypatch.setattr(bench_module, "iter_eval_chunks",
                        lambda *args: [SimpleNamespace(document=0, data=data)])
    prompts = bench_module.val_prompts("unused", 20, 30, seed=11)
    assert len(prompts) == len(set(prompts)) == 30
    assert all(len(prompt) == 20 for prompt in prompts)


def test_benchmark_graph_calibration_miss_and_hit_accounting(monkeypatch):
    """Exercise the timing/cache orchestration with a deterministic fake CUDA backend."""
    ticks, calls = [0.0], []
    def clock():
        ticks[0] += 0.001
        return ticks[0]
    class FakeEvent:
        def __init__(self, **kwargs):
            pass
        def record(self, stream):
            pass
        def elapsed_time(self, other):
            return 0.5
    class FakeCase:
        def __init__(self, bundle, group, batch, output_bytes, vocabulary, prepare_resident, **kwargs):
            self.bundle, self.group, self.batch = bundle, group, batch
            self.real_lengths = [len(value) for value in group]
            self.total_prompt_tokens = sum(self.real_lengths)
            self.actual_capacity = max(self.real_lengths)
            self.padded_capacity = 128 if group[0] == b"first timed" else 64
            self.inputs = {"tokens": torch.zeros((batch, self.padded_capacity), dtype=torch.long),
                           "lengths": torch.tensor(self.real_lengths)}
        def prepare_host(self, preprocessed=True):
            calls.append(("prepare", self.group[0]))
            return self.inputs
        def to_device(self, inputs):
            return inputs
        def host_inputs(self):
            return self.inputs
        def resident_inputs(self, **kwargs):
            return self.inputs
        def distribution(self, logits):
            ticks[0] += 1.0
    class FakeGraph:
        def __init__(self, model, inputs, **kwargs):
            calls.append(("capture", inputs["tokens"].shape[1]))
            ticks[0] += 2.0
        def replay(self, inputs):
            calls.append(("replay", inputs["tokens"].shape[1]))
            ticks[0] += 0.05
            return object(), torch.zeros((1, 1, 4))
    def eager(case, inputs):
        calls.append(("eager", inputs["tokens"].shape[1]))
        ticks[0] += 0.3
        return object(), torch.zeros((1, 1, 4))
    monkeypatch.setattr(bench_module, "time", SimpleNamespace(perf_counter=clock))
    monkeypatch.setattr(bench_module, "InferenceCase", FakeCase)
    monkeypatch.setattr(bench_module, "CUDAGraphPrefill", FakeGraph)
    monkeypatch.setattr(bench_module, "TokenBytes", lambda *args: None)
    monkeypatch.setattr(bench_module, "_eager_preprocessed", eager)
    monkeypatch.setattr(bench_module, "_software_environment", lambda device: {})
    monkeypatch.setattr(bench_module, "_gpu_telemetry", lambda device: None)
    monkeypatch.setattr(bench_module, "_sync", lambda device: None)
    monkeypatch.setattr(bench_module, "autocast_for", lambda device: nullcontext())
    monkeypatch.setattr(bench_module, "cache_statistics", lambda *args: {"logical_bytes": 1})
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda device: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    bundle = SimpleNamespace(config=config_bpe(), model=object(), tokenizer=None,
                             device=SimpleNamespace(type="cuda"), identity={"kind": "fake"})
    result = benchmark([bundle], [b"calibration", b"first timed", b"second timed"], mode="prefill",
                       warmup=1, repetitions=2, execution="graph", graph_cache_size=2)
    for variant in result["variants"].values():
        row = variant["models"][0]
        assert row["graph_misses"] == [True, False]
        assert row["graph_cache"]["hits"] == 1 and row["graph_cache"]["misses"] == 1
        # Membership is frozen after calibration: the 128-position miss runs eager and is never captured.
        assert row["graph_cache"]["miss_rate"] == 0.5 and row["graph_cache"]["size"] == 1
        assert [capture["phase"] for capture in row["graph_cache"]["captures"]] == ["calibration"]
        assert row["graph_captures"] == [0, 0]
        assert all(capture["seconds"] >= 2.0 for capture in row["graph_cache"]["captures"])
        assert 0.3 <= row["seconds"][0] < 0.4 and row["seconds"][1] < 0.1
        assert all(value > 1.0 for value in row["conversion_seconds"])
    assert calls.count(("eager", 128)) == 3
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 20_000_000_000)
    with pytest.raises(bench_module.MemoryBudgetExceeded) as raised:
        benchmark([bundle], [b"calibration", b"first timed", b"second timed"], mode="prefill",
                  warmup=1, repetitions=2, execution="graph", graph_cache_size=2)
    assert raised.value.phase == "calibration" and raised.value.budget == 10_500_000_000


def test_cache_size_is_calibrated_then_reduced_and_frozen(monkeypatch, tmp_path):
    from scripts import bench_infer
    runs = []
    def run(bundles, values, *, graph_cache_size, calibrate_only=False, **kwargs):
        runs.append((graph_cache_size, calibrate_only))
        if calibrate_only:
            if graph_cache_size >= 4:
                raise bench_module.MemoryBudgetExceeded(11_000_000_000, 10_500_000_000, "calibration")
            return {"variants": {"end_to_end": {"peak_reserved_bytes": graph_cache_size}}}
        if graph_cache_size == 2:
            raise bench_module.MemoryBudgetExceeded(10_600_000_000, 10_500_000_000, "timing")
        return {"mode": "prefill"}
    monkeypatch.setattr(bench_infer, "benchmark", run)
    monkeypatch.setattr(bench_infer, "_release", lambda: None)
    policy_path = tmp_path / "policy.json"
    args = SimpleNamespace(graph_cache_size=None, cache_policy=policy_path)
    settings = {"execution": "graph", "mode": "prefill", "memory_budget_bytes": 10_500_000_000}
    policy = {}
    result = bench_infer._run_cell([], [], settings, "prefill|32|4", args, policy)
    assert runs == [(1, True), (2, True), (4, True), (2, False), (1, False)]
    assert result["graph_cache_policy"]["size"] == 1
    assert json.loads(policy_path.read_text()) == {"prefill|32|4": 1}
    runs.clear()
    frozen = bench_infer._run_cell([], [], settings, "prefill|32|4", args, json.loads(policy_path.read_text()))
    assert runs == [(1, False)] and frozen["graph_cache_policy"]["source"] == "frozen policy"


def test_byte_diagnostic_uses_one_count_read_and_separate_graph_stages(monkeypatch):
    ticks, reads, calls = [0.0], [], []
    def clock():
        ticks[0] += 0.001
        return ticks[0]
    class FakeEvent:
        def __init__(self, **kwargs):
            pass
        def record(self, stream):
            pass
        def elapsed_time(self, other):
            return 0.5
    class FakeModel:
        def device_patch_metadata(self, raw, *, lengths):
            count = {11: 10, 22: 70, 33: 11}[lengths.tolist()[0]]
            return {"ends": torch.full((1, 88), -1, dtype=torch.long),
                    "patch_ids": torch.zeros((1, 256), dtype=torch.long), "counts": torch.tensor([count])}
    class FakeState:
        patchers = [SimpleNamespace(position=1, count=1, length=0)]
    class FakeCase:
        def __init__(self, bundle, group, batch, output_bytes, vocabulary, prepare_resident,
                     prepare_preprocessed=False, **kwargs):
            self.bundle, self.group, self.batch = bundle, group, batch
            size = {b"cal": 11, b"miss": 22, b"hit": 33}[group[0]]
            count = {11: 10, 22: 70, 33: 11}[size]
            self.real_lengths = [size]
            self.actual_capacity, self.padded_capacity = count, patch_capacity(count, 88, "bucket64")
            self.raw = {"bytes": torch.zeros((1, 256), dtype=torch.uint8), "lengths": torch.tensor([size])}
            self.preprocessed = self._preprocessed()
        def _preprocessed(self):
            metadata = self.bundle.model.device_patch_metadata(self.raw["bytes"], lengths=self.raw["lengths"])
            return {**self.raw, "patches": trim_patch_metadata(metadata, self.padded_capacity)}
        def prepare_host(self, preprocessed=True):
            return self.preprocessed if preprocessed else self.raw
        def to_device(self, inputs):
            return inputs
        def host_inputs(self):
            return self.preprocessed
        def resident_inputs(self, *, preprocessed=False):
            return self.preprocessed if preprocessed else self.raw
        def distribution(self, logits):
            ticks[0] += 1.0
    class FakePatcher:
        def __init__(self, model, inputs, **kwargs):
            self.model = model
            calls.append("capture_patcher")
        def replay(self, inputs):
            calls.append("patcher")
            return self.model.device_patch_metadata(inputs["bytes"], lengths=inputs["lengths"])
    class FakeTrunk:
        def __init__(self, model, inputs, **kwargs):
            calls.append(("capture_trunk", inputs["patches"]["ends"].shape[1]))
        def replay(self, inputs):
            calls.append("trunk")
            return FakeState(), torch.zeros((1, 1, 257))
    def eager(case, inputs):
        calls.append("eager_trunk")
        return FakeState(), torch.zeros((1, 1, 257))
    original_item = torch.Tensor.item
    def count_item(tensor, *args, **kwargs):
        reads.append(tensor.shape)
        return original_item(tensor, *args, **kwargs)
    monkeypatch.setattr(bench_module, "time", SimpleNamespace(perf_counter=clock))
    monkeypatch.setattr(bench_module, "InferenceCase", FakeCase)
    monkeypatch.setattr(bench_module, "CUDAGraphPatcher", FakePatcher)
    monkeypatch.setattr(bench_module, "CUDAGraphPrefill", FakeTrunk)
    monkeypatch.setattr(bench_module, "_eager_preprocessed", eager)
    monkeypatch.setattr(bench_module, "_software_environment", lambda device: {})
    monkeypatch.setattr(bench_module, "_gpu_telemetry", lambda device: None)
    monkeypatch.setattr(bench_module, "_sync", lambda device: None)
    monkeypatch.setattr(bench_module, "autocast_for", lambda device: nullcontext())
    monkeypatch.setattr(bench_module, "cache_statistics", lambda *args: {"logical_bytes": 1})
    monkeypatch.setattr(torch.Tensor, "item", count_item)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: None)
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda device: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    bundle = SimpleNamespace(config=config_byte(hashes=False), model=FakeModel(), tokenizer=None,
                             device=SimpleNamespace(type="cuda"), identity={"kind": "fake"})
    result = benchmark([bundle], [b"cal", b"miss", b"hit"], mode="prefill", warmup=1,
                       repetitions=2, execution="graph", graph_cache_size=3)
    row = result["variants"]["gpu_resident"]["models"][0]
    assert row["graph_misses"] == [True, False]
    assert row["graph_stage_hits"] == [{"patcher": True, "trunk": False}, {"patcher": True, "trunk": True}]
    assert [(entry["actual"], entry["padded"]) for entry in row["capacities"]] == [(70, 88), (11, 64)]
    # Membership is frozen after calibration: the timed miss is not captured.
    assert [capture["stage"] for capture in row["graph_cache"]["captures"]] == ["patcher", "trunk"]
    assert len(reads) == 3 and all(shape == torch.Size([]) for shape in reads)  # calibration plus two trials
    host_e2e = result["variants"]["end_to_end"]["models"][0]
    assert all(hits["patcher"] is None for hits in host_e2e["graph_stage_hits"])
    reads.clear()
    device = benchmark([bundle], [b"cal", b"miss", b"hit"], mode="prefill", warmup=1, repetitions=2,
                       execution="graph", graph_cache_size=3, byte_end_to_end_patching="device")
    e2e = device["variants"]["end_to_end"]["models"][0]
    assert device["byte_end_to_end_patching"] == "device"
    assert e2e["graph_stage_hits"] == [{"patcher": True, "trunk": False}, {"patcher": True, "trunk": True}]
    assert [(entry["actual"], entry["padded"]) for entry in e2e["capacities"]] == [(70, 88), (11, 64)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph prefill requires CUDA")
@pytest.mark.parametrize("family", ["bpe", "byte"])
def test_cuda_replay_changes_contents_and_real_lengths_within_bucket(family):
    torch.manual_seed(1103)
    device = torch.device("cuda")
    if family == "bpe":
        model = BPELM(config_bpe()).to(device).eval()
        lengths = ([11, 22], [28, 15])
        examples = []
        for trial, sizes in enumerate(lengths):
            tokens = torch.zeros((2, 64), dtype=torch.long, device=device)
            for row, size in enumerate(sizes):
                tokens[row, :size] = (torch.arange(size, device=device) * (row + 2) + trial + 1) % 64
            examples.append({"tokens": tokens, "lengths": torch.tensor(sizes, device=device)})
    else:
        config = config_byte(hashes=True)
        model = ByteLatentLM(config).to(device).eval()
        bundle = EvaluationModel(model, config, None, device, {"kind": "byte"})
        prompts = [[(b"abcdefghijkl" * 20)[:200], (b"first text " * 20)[:147]],
                   [(b"abcde " * 40)[:218], (b"second line " * 20)[:127]]]
        cases = [InferenceCase(bundle, rows, 2, prepare_preprocessed=True) for rows in prompts]
        examples = [case.resident_inputs(preprocessed=True) for case in cases]
        counts = [int(example["patches"]["counts"].max().item()) for example in examples]
        assert counts[0] != counts[1]
        assert examples[0]["patches"]["ends"].shape == examples[1]["patches"]["ends"].shape
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        if family == "byte":
            patch_inputs = [{key: value for key, value in example.items() if key != "patches"} for example in examples]
            patcher = CUDAGraphPatcher(model, patch_inputs[0], warmup=1)
            for data, example in zip(patch_inputs, examples):
                metadata = patcher.replay(data)
                capacity = example["patches"]["ends"].shape[1]
                for key, expected in example["patches"].items():
                    torch.testing.assert_close(trim_patch_metadata(metadata, capacity)[key], expected, atol=0, rtol=0)
        graph = CUDAGraphPrefill(model, examples[0], max_len=examples[0]["tokens" if family == "bpe" else "bytes"].shape[1] + 4,
                                 patching="preprocessed" if family == "byte" else "device", warmup=1)
        for example in examples:
            source = example["tokens" if family == "bpe" else "bytes"]
            eager = model.init_state(2, source.shape[1] + 4, device, torch.bfloat16)
            expected = (model.prefill(eager, source, lengths=example["lengths"]) if family == "bpe" else
                        model.prefill(eager, example, lengths=example["lengths"], patching="preprocessed"))
            captured, actual = graph.replay(example)
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
            if family == "bpe":
                for row, length in enumerate(eager.cache.lengths):
                    compare_batched_cache(captured.cache, row, eager.cache[row], length, atol=2e-2, rtol=2e-2)
            else:
                compare_byte_states(captured, eager, atol=2e-2, rtol=2e-2)
            for following in (torch.tensor([12, 13], device=device), torch.tensor([14, 15], device=device)):
                torch.testing.assert_close(model.step(captured, following), model.step(eager, following),
                                           atol=2e-2, rtol=2e-2)
                if family == "bpe":
                    for row, length in enumerate(eager.cache.lengths):
                        compare_batched_cache(captured.cache, row, eager.cache[row], length, atol=2e-2, rtol=2e-2)
                else:
                    compare_byte_states(captured, eager, atol=2e-2, rtol=2e-2)

