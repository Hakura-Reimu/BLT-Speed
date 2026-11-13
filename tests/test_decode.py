import copy
import pickle
from types import SimpleNamespace

import pytest
import torch

from blt.config import BPEConfig, ByteConfig, TransformerConfig
from blt.decode import (CUDAGraphStepper, CUDASplitGraphStepper, DeviceRouting, StaticBPEState, StaticByteState,
                        _common_prefix, _transformer_chunk, speculate_bpe, speculate_byte)
from blt.eval_io import EvaluationModel
from blt.model import BPELM, ByteLatentLM


def trunk(max_positions=512, layers=1):
    return TransformerConfig(layers, 16, 2, 1, 8, 32, max_positions=max_positions)


def byte_model(hash_enabled=True, *, global_layers=1, decoder_layers=1):
    return ByteLatentLM(ByteConfig("tiny", trunk(layers=global_layers), 16, 2, 32, 2, 4, max_len=5, block_size=8,
                                   dec_layers=decoder_layers, hash_enabled=hash_enabled, hash_rows=32)).eval()


def bpe_model():
    return BPELM(BPEConfig("tiny", trunk(), 32, 0)).eval()


def byte_prompt(batch, device="cpu"):
    return torch.tensor([[97 + row % 128, 32, 98, 33, 99, 100, 44, 101, 102]
                         for row in range(batch)], dtype=torch.long, device=device)


@pytest.mark.parametrize("with_hashes", [False, True])
def test_device_routing_matches_incremental_patchers_and_hashes(monkeypatch, with_hashes):
    torch.manual_seed(6)
    model = byte_model(with_hashes)
    raw = byte_prompt(5)
    sizes = [0, 1, 5, 8, 9]
    reference = model.init_state(5, 100, "cpu", torch.float32)
    model.prefill(reference, raw, lengths=sizes)
    routing = DeviceRouting.from_reference(reference, model)
    for index in range(80):
        values = torch.tensor([(index * 37 + row * 17) % 256 for row in range(5)])
        expected_closed = []
        expected_hashes = []
        for row, value in enumerate(values.tolist()):
            expected_closed.append(reference.patchers[row].step(value))
            if with_hashes:
                expected_hashes.append(reference.hashes[row].step(value))
        with monkeypatch.context() as scoped:
            def forbidden(*args, **kwargs):
                raise AssertionError("routing read device values on the host")
            scoped.setattr(torch.Tensor, "cpu", forbidden)
            scoped.setattr(torch.Tensor, "item", forbidden)
            scoped.setattr(torch.Tensor, "tolist", forbidden)
            closed, hashes = routing.step(values, need_hashes=with_hashes)
        assert closed.tolist() == expected_closed
        if with_hashes:
            assert hashes.tolist() == expected_hashes
        else:
            assert hashes is None
        assert routing.positions.tolist() == [p.position for p in reference.patchers]
        assert routing.patch_lengths.tolist() == [p.length for p in reference.patchers]
        assert routing.patch_counts.tolist() == [p.count for p in reference.patchers]


@pytest.mark.parametrize("with_hashes", [False, True])
def test_static_byte_cache_matches_host_subbatch_and_masks_unclosed_rows(monkeypatch, with_hashes):
    torch.manual_seed(11)
    model = byte_model(with_hashes)
    raw = byte_prompt(5)
    sizes = [0, 2, 5, 8, 9]
    reference = model.init_state(5, 80, "cpu", torch.float32)
    model.prefill(reference, raw, lengths=sizes)
    static = StaticByteState(model, reference)
    for index in range(32):
        values = torch.tensor([32 if (index + row) % 4 == 0 else (index * 13 + row * 7) % 256
                               for row in range(5)])
        positions_before = list(reference.globals.lengths)
        keys_before = [key.clone() for key in static.globals.keys]
        values_before = [value.clone() for value in static.globals.values]
        with monkeypatch.context() as scoped:
            def forbidden(*args, **kwargs):
                raise AssertionError("static byte step read device values on the host")
            scoped.setattr(torch.Tensor, "cpu", forbidden)
            scoped.setattr(torch.Tensor, "item", forbidden)
            scoped.setattr(torch.Tensor, "tolist", forbidden)
            actual = static.step(values)
        expected = model.step(reference, values)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        assert static.global_positions.tolist() == reference.globals.lengths
        for row, (before, after) in enumerate(zip(positions_before, reference.globals.lengths)):
            if before == after:
                for old, new in zip(keys_before, static.globals.keys):
                    assert torch.equal(old[row], new[row])
                for old, new in zip(values_before, static.globals.values):
                    assert torch.equal(old[row], new[row])


def test_static_bpe_matches_eager_and_snapshot_restores_all_state():
    torch.manual_seed(12)
    model = bpe_model()
    prefix = torch.tensor([[0, 2, 4, 6], [0, 3, 5, 7]])
    reference = model.init_state(2, 64, "cpu", torch.float32)
    logits = model.prefill(reference, prefix)
    allowed = torch.ones(32, dtype=torch.bool)
    allowed[0] = False
    static = StaticBPEState(model, reference, logits, allowed)
    snapshot = static.snapshot()
    original_addresses = [value.data_ptr() for value in static.tensors()]
    for index in range(24):
        tokens = torch.tensor([1 + index % 31, 1 + (index * 7) % 31])
        expected = model.step(reference, tokens)
        actual = static.step(tokens)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        assert torch.equal(static.current, expected[:, -1].masked_fill(~allowed, -torch.inf).argmax(dim=-1))
    static.restore(snapshot)
    assert [value.data_ptr() for value in static.tensors()] == original_addresses
    assert static.consumed == prefix.shape[1]
    for value, saved in zip(static.tensors(), snapshot[0]):
        assert torch.equal(value, saved)


def test_static_bpe_tracks_distinct_row_lengths_and_masks_padding():
    torch.manual_seed(120)
    model = bpe_model()
    prompt = torch.tensor([[0, 1, 2, 0, 0, 0], [0, 3, 4, 5, 0, 0], [0, 6, 7, 8, 9, 10]])
    lengths = [3, 4, 6]
    reference = model.init_state(3, 64, "cpu", torch.float32)
    logits = model.prefill(reference, prompt, lengths=lengths)
    static = StaticBPEState(model, reference, logits)
    assert static.positions.tolist() == lengths
    for index in range(24):
        tokens = torch.tensor([1 + index % 31, 1 + index * 3 % 31, 1 + index * 7 % 31])
        expected = model.step(reference, tokens)
        actual = static.step(tokens)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        assert static.positions.tolist() == reference.cache.lengths


@pytest.mark.parametrize("with_hashes", [False, True])
def test_conditional_byte_step_skips_empty_closures_and_matches_reference(monkeypatch, with_hashes):
    torch.manual_seed(121)
    model = byte_model(with_hashes)
    reference = model.init_state(3, 80, "cpu", torch.float32)
    model.prefill(reference, byte_prompt(3), lengths=[5, 7, 9])
    masked = StaticByteState(model, reference)
    conditional = StaticByteState(model, reference)
    original = conditional._global_tensor
    calls = []

    def count_global(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(conditional, "_global_tensor", count_global)
    branches = []
    for index in range(32):
        values = torch.tensor([32 if (index + row) % 7 == 0 else 97 + (row + index) % 20 for row in range(3)])
        branches.append(any(copy.copy(p).step(v) for p, v in zip(reference.patchers, values.tolist())))
        expected = model.step(reference, values)
        actual = conditional.conditional_step(values)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(masked.step(values), expected, atol=3e-5, rtol=3e-5)
        assert conditional.routing.positions.tolist() == [p.position for p in reference.patchers]
        assert conditional.global_positions.tolist() == reference.globals.lengths
    assert any(branches) and not all(branches)
    assert len(calls) == sum(branches)


def test_bpe_parallel_verification_matches_sequential_logits():
    torch.manual_seed(13)
    model = bpe_model()
    prefix = torch.tensor([[0, 2, 4], [0, 3, 5]])
    proposal = torch.tensor([[6, 7, 8, 9], [9, 8, 7, 6]])
    reference = model.init_state(2, 32, "cpu", torch.float32)
    logits = model.prefill(reference, prefix)
    static = StaticBPEState(model, reference, logits)
    parallel = static.step_chunk(proposal)
    sequential = torch.cat([model.step(reference, token) for token in proposal.unbind(dim=1)], dim=1)
    torch.testing.assert_close(parallel, sequential, atol=3e-5, rtol=3e-5)
    assert static.consumed == reference.cache.length
    assert torch.equal(static.current, sequential[:, -1].argmax(dim=-1))


def test_common_prefix_stops_at_the_first_mismatching_row():
    matches = torch.tensor([[True, True, True, True], [True, True, False, True]])
    assert _common_prefix(matches) == 2
    assert _common_prefix(torch.zeros((2, 3), dtype=torch.bool)) == 0
    assert _common_prefix(torch.ones((2, 3), dtype=torch.bool)) == 3


def test_byte_draft_freezes_global_state_across_patch_closures():
    torch.manual_seed(29)
    model = byte_model()
    reference = model.init_state(2, 32, "cpu", torch.float32)
    model.prefill(reference, byte_prompt(2)[:, :1])
    static = StaticByteState(model, reference)
    conditioning = static.conditioning.clone()
    positions = static.global_positions.clone()
    counts = static.routing.patch_counts.clone()
    global_tensors = [tensor.clone() for tensor in static.globals.tensors()]
    for _ in range(12):
        static.draft_step(torch.tensor([32, 33]))
    assert torch.all(static.routing.patch_counts > counts)
    assert torch.equal(static.global_positions, positions)
    assert torch.equal(static.conditioning, conditioning)
    for current, saved in zip(static.globals.tensors(), global_tensors):
        assert torch.equal(current, saved)


def test_packed_global_verification_handles_ragged_patch_counts():
    torch.manual_seed(30)
    model = byte_model(global_layers=2)
    reference = model.init_state(4, 64, "cpu", torch.float32)
    model.prefill(reference, byte_prompt(4), lengths=[0, 3, 7, 9])
    static = StaticByteState(model, reference)
    values = torch.randn(4, 3, model.config.trunk.width)
    counts = [0, 1, 3, 2]
    expected = torch.zeros_like(values)
    unchanged = [key[0].clone() for key in static.globals.keys]
    with torch.inference_mode():
        actual = _transformer_chunk(model.global_trunk, values, static.globals, static.global_positions,
                                    valid_counts=counts)
        for offset in range(3):
            rows = [row for row, count in enumerate(counts) if offset < count]
            positions = [reference.globals.lengths[row] for row in rows]
            hidden = model.global_trunk.step_batched(values[rows, offset:offset + 1], reference.globals,
                                                     rows, positions)
            expected[rows, offset:offset + 1] = hidden
    for row, count in enumerate(counts):
        torch.testing.assert_close(actual[row, :count], expected[row, :count], atol=3e-5, rtol=3e-5)
    for old, key in zip(unchanged, static.globals.keys):
        assert torch.equal(old, key[0])
    for row, length in enumerate(reference.globals.lengths):
        for actual_key, expected_key in zip(static.globals.keys, reference.globals.keys):
            torch.testing.assert_close(actual_key[row, :, :length], expected_key[row, :, :length], atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("with_hashes", [False, True])
@pytest.mark.parametrize("k", [4, 8, 16])
def test_byte_speculation_matches_plain_greedy(with_hashes, k):
    torch.manual_seed(14)
    model = byte_model(with_hashes)
    reference = model.init_state(2, 60, "cpu", torch.float32)
    logits = model.prefill(reference, byte_prompt(2))
    static = StaticByteState(model, reference)
    generated = 0
    while generated < 24:
        result = speculate_byte(static, min(k, 24 - generated))
        for tokens in result.tokens.unbind(dim=1):
            expected = logits[:, -1, :256].argmax(dim=-1)
            assert torch.equal(tokens, expected)
            logits = model.step(reference, tokens)
        generated += result.tokens.shape[1]
    assert torch.equal(static.current, logits[:, -1, :256].argmax(dim=-1))
    assert static.global_positions.tolist() == reference.globals.lengths
    assert static.routing.positions.tolist() == [p.position for p in reference.patchers]
    assert static.routing.patch_lengths.tolist() == [p.length for p in reference.patchers]
    if with_hashes:
        for row, hashes in enumerate(reference.hashes):
            assert static.routing.history[row].tolist() == list(hashes.history)


def test_byte_speculation_rolls_back_rejected_suffix(monkeypatch):
    torch.manual_seed(15)
    model = byte_model()
    reference = model.init_state(2, 40, "cpu", torch.float32)
    logits = model.prefill(reference, byte_prompt(2))
    static = StaticByteState(model, reference)
    before = static.snapshot()
    static.step()
    correct_second = static.current.clone()
    static.restore(before)
    original = static.draft_step

    def force_second_mismatch(*args, **kwargs):
        result = original(*args, **kwargs)
        static.current.copy_((correct_second + 1).remainder(256))
        return result

    monkeypatch.setattr(static, "draft_step", force_second_mismatch)
    result = speculate_byte(static, 4)
    assert result.accepted_draft_positions == 1
    assert result.tokens.shape[1] == 2
    for tokens in result.tokens.unbind(dim=1):
        assert torch.equal(tokens, logits[:, -1, :256].argmax(dim=-1))
        logits = model.step(reference, tokens)
    assert torch.equal(static.current, logits[:, -1, :256].argmax(dim=-1))
    assert static.global_positions.tolist() == reference.globals.lengths
    assert static.routing.patch_lengths.tolist() == [p.length for p in reference.patchers]


def test_byte_all_accepted_commits_packed_patches_and_local_block_reset(monkeypatch):
    torch.manual_seed(16)
    model = byte_model(global_layers=2, decoder_layers=2)
    reference = model.init_state(3, 48, "cpu", torch.float32)
    logits = model.prefill(reference, byte_prompt(3), lengths=[0, 3, 7])
    static = StaticByteState(model, reference)
    expected = []
    for _ in range(12):
        token = logits[:, -1, :256].argmax(dim=-1)
        expected.append(token.clone())
        logits = model.step(reference, token)
    original = static.draft_step
    calls = 0

    def force_matching_draft(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls < len(expected):
            static.current.copy_(expected[calls])
        return result

    monkeypatch.setattr(static, "draft_step", force_matching_draft)
    result = speculate_byte(static, 12)
    assert result.accepted_draft_positions == 12
    assert torch.equal(result.tokens, torch.stack(expected, dim=1))
    assert torch.equal(static.current, logits[:, -1, :256].argmax(dim=-1))
    assert static.global_positions.tolist() == reference.globals.lengths
    assert static.routing.patch_lengths.tolist() == [p.length for p in reference.patchers]
    torch.testing.assert_close(static.conditioning, reference.conditioning, atol=3e-5, rtol=3e-5)
    for _ in range(4):
        token = logits[:, -1, :256].argmax(dim=-1)
        logits = model.step(reference, token)
        actual = static.step(token)
        torch.testing.assert_close(actual, logits, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("k", [2, 4, 8])
def test_bpe_speculation_matches_plain_greedy_and_corrects_mismatch(k):
    torch.manual_seed(20)
    target_model = bpe_model()
    torch.manual_seed(21)
    draft_model = bpe_model()
    prompt = torch.tensor([[0, 1, 2, 3], [0, 3, 2, 1]])
    target_reference = target_model.init_state(2, 48, "cpu", torch.float32)
    draft_reference = draft_model.init_state(2, 48, "cpu", torch.float32)
    logits = target_model.prefill(target_reference, prompt)
    draft_logits = draft_model.prefill(draft_reference, prompt)
    target = StaticBPEState(target_model, target_reference, logits)
    drafter = StaticBPEState(draft_model, draft_reference, draft_logits)
    drafter.current.copy_((target.current + 1).remainder(32))
    result = speculate_bpe(target, drafter, k)
    assert result.accepted_draft_positions == 0 and result.tokens.shape[1] == 1
    logits = target_model.step(target_reference, result.tokens[:, 0])
    generated = 1
    while generated < 20:
        result = speculate_bpe(target, drafter, min(k, 20 - generated))
        for tokens in result.tokens.unbind(dim=1):
            assert torch.equal(tokens, logits[:, -1].argmax(dim=-1))
            logits = target_model.step(target_reference, tokens)
        generated += result.tokens.shape[1]
    assert torch.equal(target.current, logits[:, -1].argmax(dim=-1))
    assert target.consumed == drafter.consumed == target_reference.cache.length


def test_bpe_partial_acceptance_replaces_only_the_rejected_suffix(monkeypatch):
    torch.manual_seed(22)
    model = bpe_model()
    draft_model = copy.deepcopy(model)
    prompt = torch.tensor([[0, 1, 2, 3], [0, 3, 2, 1]])
    reference = model.init_state(2, 32, "cpu", torch.float32)
    logits = model.prefill(reference, prompt)
    draft_reference = draft_model.init_state(2, 32, "cpu", torch.float32)
    draft_logits = draft_model.prefill(draft_reference, prompt)
    target = StaticBPEState(model, reference, logits)
    drafter = StaticBPEState(draft_model, draft_reference, draft_logits)
    first = logits[:, -1].argmax(dim=-1)
    second_logits = model.step(reference, first)
    second = second_logits[:, -1].argmax(dim=-1)
    original = drafter.step
    calls = 0

    def force_mismatch(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            drafter.current.copy_((second + 1).remainder(32))
        return result

    monkeypatch.setattr(drafter, "step", force_mismatch)
    result = speculate_bpe(target, drafter, 4)
    assert result.accepted_draft_positions == 1
    assert torch.equal(result.tokens, torch.stack((first, second), dim=1))
    final = model.step(reference, second)
    assert torch.equal(target.current, final[:, -1].argmax(dim=-1))
    assert target.consumed == drafter.consumed == reference.cache.length


def test_bpe_all_accepted_keeps_both_caches_advanced():
    torch.manual_seed(23)
    model = bpe_model()
    draft_model = copy.deepcopy(model)
    prompt = torch.tensor([[0, 4, 6, 8], [0, 5, 7, 9]])
    reference = model.init_state(2, 32, "cpu", torch.float32)
    logits = model.prefill(reference, prompt)
    draft_reference = draft_model.init_state(2, 32, "cpu", torch.float32)
    draft_logits = draft_model.prefill(draft_reference, prompt)
    target = StaticBPEState(model, reference, logits)
    drafter = StaticBPEState(draft_model, draft_reference, draft_logits)
    result = speculate_bpe(target, drafter, 4)
    assert result.accepted_draft_positions == 4
    for token in result.tokens.unbind(dim=1):
        assert torch.equal(token, logits[:, -1].argmax(dim=-1))
        logits = model.step(reference, token)
    assert torch.equal(target.current, logits[:, -1].argmax(dim=-1))
    assert target.consumed == drafter.consumed == reference.cache.length


def test_cpu_decode_benchmark_structure_and_output_byte_counts():
    from scripts.bench_decode import benchmark_decode

    torch.manual_seed(25)
    model = byte_model(False)
    bundle = EvaluationModel(model, model.config, None, torch.device("cpu"), {"kind": "random"})
    result = benchmark_decode([bundle], [b"An ordinary prompt!", b"Another prompt text"], batch=2, output_bytes=4,
                              modes=["eager", "device", "speculative"], warmup=0, repetitions=1)
    assert {variant["decoder_variant"] for variant in result} == {"eager", "device", "speculative_byte4_bpe2",
                                                                     "speculative_byte8_bpe4", "speculative_byte16_bpe8"}
    for variant in result:
        row = variant["variants"]["decode"]["models"][0]
        assert row["bytes"] == [8]
        if variant["decoder_mode"] != "speculative":
            assert row["overshoot_bytes"] == [[4, 4]]
        assert row["median_seconds"] > 0 and row["cache"][0]["logical_bytes"] > 0
        if variant["decoder_mode"] == "speculative":
            assert len(row["speculation_rounds"][0]) > 0
            assert row["accepted_bytes_per_verification"] >= 0


def test_decode_benchmark_pairs_all_cross_model_decoder_modes():
    from scripts.bench_decode import benchmark_decode

    torch.manual_seed(24)
    models = [byte_model(False), byte_model(False)]
    bundles = [EvaluationModel(model, model.config, None, torch.device("cpu"), {"kind": "random", "seed": i})
               for i, model in enumerate(models)]
    results = benchmark_decode(bundles, [b"Paired prompt."], batch=1, output_bytes=3,
                               modes=["eager", "device"], warmup=0, repetitions=1)
    for result in results:
        paired = result["variants"]["decode"]
        assert paired["paired_bootstrap"]["speed_a_over_b"] > 0
        assert set(paired["cross_decoder_paired_bootstrap"]) == {"eager", "device"}
        assert all(value["speed_a_over_b"] > 0 for value in paired["cross_decoder_paired_bootstrap"].values())


def test_decode_benchmark_preserves_other_results_after_one_mode_fails(monkeypatch):
    import scripts.bench_decode as bench

    model = byte_model(False)
    bundle = EvaluationModel(model, model.config, None, torch.device("cpu"), {"kind": "random"})
    original = bench._prepare

    def fail_device(case, mode, *args):
        if mode == "device":
            raise RuntimeError("simulated device failure")
        return original(case, mode, *args)

    monkeypatch.setattr(bench, "_prepare", fail_device)
    results = bench.benchmark_decode([bundle], [b"A prompt."], batch=1, output_bytes=3,
                                     modes=["eager", "device"], warmup=0, repetitions=1)
    rows = {result["decoder_variant"]: result["variants"]["decode"]["models"][0] for result in results}
    assert rows["eager"]["bytes"] == [3] and rows["eager"]["bytes_per_second"] > 0
    assert "simulated device failure" in rows["device"]["error"]
    assert "bytes_per_second" not in rows["device"]


def test_decode_benchmark_records_memory_budget_error(monkeypatch):
    import scripts.bench_decode as bench
    from blt.bench import MemoryBudgetExceeded

    model = byte_model(False)
    bundle = EvaluationModel(model, model.config, None, torch.device("cpu"), {"kind": "random"})

    def check(device, budget, phase):
        if phase == "decode preparation":
            raise MemoryBudgetExceeded(101, 100, phase)
        return 0

    monkeypatch.setattr(bench, "_check_budget", check)
    result = bench.benchmark_decode([bundle], [b"A prompt."], batch=1, output_bytes=3, modes=["device"],
                                    warmup=0, repetitions=1, memory_budget_bytes=100)[0]
    row = result["variants"]["decode"]["models"][0]
    assert "reserved memory 101 exceeds" in row["error"]
    assert "bytes_per_second" not in row


def test_bpe_benchmark_counts_output_bytes_and_overshoot():
    from scripts.bench_decode import benchmark_decode

    class TinyTokenizer:
        def get_added_tokens_decoder(self):
            return {0: SimpleNamespace(special=True)}

        def id_to_token(self, index):
            return ["<bos>", "a", "bb"][index]

        def encode(self, text, add_special_tokens=False):
            assert set(text) <= {"a"}
            return SimpleNamespace(ids=[1] * len(text))

        def decode(self, ids, skip_special_tokens=False):
            return "".join(self.id_to_token(index) for index in ids)

    tokenizer = TinyTokenizer()
    target_config = BPEConfig("target", trunk(layers=2), 3, 0)
    draft_config = BPEConfig("draft", trunk(layers=1), 3, 0)
    target, drafter = BPELM(target_config).eval(), BPELM(draft_config).eval()
    with torch.no_grad():
        for model in (target, drafter):
            model.embedding.weight[0].zero_()
            model.embedding.weight[1].fill_(1)
            model.embedding.weight[2].fill_(2)
    identity = {"kind": "random", "tokenizer_sha256": "shared"}
    target_bundle = EvaluationModel(target, target_config, tokenizer, torch.device("cpu"), identity)
    draft_bundle = EvaluationModel(drafter, draft_config, tokenizer, torch.device("cpu"), identity)
    results = benchmark_decode([target_bundle], [b"aaaa", b"aaa"], batch=2, output_bytes=3,
                               modes=["eager", "device", "speculative"], drafter=draft_bundle,
                               warmup=0, repetitions=1)
    for result in results:
        row = result["variants"]["decode"]["models"][0]
        assert row["bytes"] == [6]
        for actual, overshoot, units, ratio in zip(row["output_bytes_per_row"][0], row["overshoot_bytes"][0],
                                                   row["generated_units_per_row"][0], row["bytes_per_token"][0]):
            assert actual >= 3 and overshoot == actual - 3
            assert ratio == pytest.approx(actual / units)


@pytest.mark.parametrize("mode,k", [("device", 0), ("speculative", 4)])
def test_checkpoint_style_verifier_counts_bytes_and_compares_greedy(mode, k):
    from scripts.bench_decode import verify_decode

    torch.manual_seed(26)
    model = byte_model(False)
    bundle = EvaluationModel(model, model.config, None, torch.device("cpu"), {"kind": "random"})
    result = verify_decode(bundle, [b"First prompt!", b"Second prompt?"], batch=2, output_bytes=4, mode=mode, k=k)
    assert result["checked_output_bytes"] == 8
    assert result["greedy_mismatches"] == 0
    assert len(set(result["prompt_sha256"])) == 2


class _RaggedTokenizer:
    def get_added_tokens_decoder(self):
        return {0: SimpleNamespace(special=True)}

    def id_to_token(self, index):
        return "<bos>" if index == 0 else chr(32 + index)

    def encode(self, text, add_special_tokens=False):
        width = 1 if text.startswith("a") else 2
        return SimpleNamespace(ids=[1 + ord(text[i]) % 31 for i in range(0, len(text), width)])


@pytest.mark.parametrize("family", ["byte", "bpe"])
@pytest.mark.parametrize("mode", ["eager", "device", "device-conditional"])
def test_distinct_rows_verify_each_row_for_requested_length(family, mode):
    from scripts.bench_decode import verify_decode

    torch.manual_seed(122)
    model = byte_model(False) if family == "byte" else bpe_model()
    tokenizer = None if family == "byte" else _RaggedTokenizer()
    bundle = EvaluationModel(model, model.config, tokenizer, torch.device("cpu"), {"kind": "random"})
    prompts = [b"aaaaaa  ", b"bbbbbbbb", b"cc ccc c"]
    result = verify_decode(bundle, prompts, batch=3, output_bytes=24, mode=mode)
    assert result["greedy_mismatches"] == 0
    assert len(set(result["prompt_sha256"])) == 3
    assert min(result["checked_output_bytes_per_row"][0]) >= 24
    if family == "byte":
        assert min(result["generated_patch_closures_per_row"][0]) >= 2
        assert min(result["local_block_resets_per_row"][0]) >= 3


def test_polling_uses_device_counters_and_counts_all_overshoot(monkeypatch):
    from scripts.bench_decode import _decode, _finish_decode

    class State:
        def __init__(self):
            self.current = torch.tensor([1, 2])
            self.steps = 0

        def step(self):
            self.steps += 1

    config = BPEConfig("tiny", trunk(), 3, 0)
    case = SimpleNamespace(bundle=SimpleNamespace(config=config, device=torch.device("cpu")),
                           batch=2, output_bytes=9, _output_sizes=torch.tensor([0, 1, 3]))
    state = State()
    with monkeypatch.context() as scoped:
        def forbidden(*args, **kwargs):
            raise AssertionError("token IDs were read on the host between polls")
        scoped.setattr(torch.Tensor, "tolist", forbidden)
        generated = _decode(case, "device", state, state.current, None, None, None, 0)
    generated = _finish_decode(case, "device", state, generated)
    assert generated["output_bytes_per_row"] == [16, 48]
    assert generated["overshoot_bytes"] == [7, 39]
    assert generated["generated_units"] == [16, 16]
    assert generated["polls"] == 2 and generated["native_steps"] == 16 and state.steps == 15
    assert generated["cached_output_bytes"] == 60


def test_mode_selection_is_per_model_and_requires_all_calibrations():
    from scripts.bench_decode import select_modes

    def result(label, times):
        return {"decoder_variant": label, "variants": {"decode": {"models": [
            {"seconds": [t, t], "median_seconds": t, "prompt_sha256": ["a", "b"],
             "prompt_sha256_per_row": [["a"], ["b"]], "prompt_bytes_per_row": [[8], [8]]} for t in times]}}}

    data = [result("device", [3., 1.]), result("eager", [1., 4.])]
    choices, details = select_modes(data, ["device", "eager"], 2)
    assert choices == ["eager", "device"]
    assert [detail["selected_mode"] for detail in details] == choices
    data[0]["variants"]["decode"]["models"][0]["error"] = "OOM"
    with pytest.raises(ValueError, match="calibration failed"):
        select_modes(data, ["device", "eager"], 2)


def test_selected_benchmark_pairs_distinct_row_prompts_and_requested_work():
    from scripts.bench_decode import benchmark_decode

    torch.manual_seed(123)
    models = [byte_model(False), byte_model(False)]
    bundles = [EvaluationModel(model, model.config, None, torch.device("cpu"), {"kind": "random", "seed": i})
               for i, model in enumerate(models)]
    prompts = [b"One prompt here!", b"Two prompt here!"]
    result = benchmark_decode(bundles, prompts, batch=2, output_bytes=5, modes=["eager", "device-conditional"],
                              selected_modes=["eager", "device-conditional"], warmup=0, repetitions=1)[0]
    first, second = result["variants"]["decode"]["models"]
    assert [first["decoder_mode"], second["decoder_mode"]] == ["eager", "device-conditional"]
    assert first["prompt_sha256_per_row"] == second["prompt_sha256_per_row"]
    assert len(set(first["prompt_sha256_per_row"][0])) == 2
    assert first["bytes"] == second["bytes"] == [10]
    assert first["polls"] == second["polls"] == [1]
    assert all(n == 8 for n in first["generated_units_per_row"][0])
    with pytest.raises(ValueError, match="distinct"):
        benchmark_decode(bundles, [prompts[0], prompts[0]], batch=2, output_bytes=5, modes=["eager"],
                         warmup=0, repetitions=1)


def test_generated_patch_counts_include_the_final_polled_output():
    from blt.bench import InferenceCase
    from scripts.bench_decode import benchmark_decode

    torch.manual_seed(126)
    model = byte_model(False)
    bundle = EvaluationModel(model, model.config, None, torch.device("cpu"), {"kind": "random"})
    prompts = [b"one! ab ", b"two xyz "]
    case = InferenceCase(bundle, prompts, 2, 5)
    reference, logits, _ = case.prefill(resident=True, headroom=16)
    before = [p.count for p in reference.patchers]
    for _ in range(8):
        logits = model.step(reference, logits[:, -1, :256].argmax(dim=-1))
    expected = [p.count - initial for p, initial in zip(reference.patchers, before)]
    results = benchmark_decode([bundle], prompts, batch=2, output_bytes=5,
                               modes=["eager", "device", "device-conditional"], warmup=0, repetitions=1)
    for result in results:
        row = result["variants"]["decode"]["models"][0]
        assert row["generated_patch_closures_per_row"] == [expected]
        assert row["output_bytes_per_row"] == [[8, 8]]


def test_calibration_and_real_benchmark_output_pass_summary_check(tmp_path):
    import hashlib
    import json

    from scripts.bench_decode import benchmark_decode, select_modes, verify_decode
    from scripts.diag.decode_summary import problems
    from blt.score import configure_matmul_precision

    class ShortTokenizer(_RaggedTokenizer):
        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(ids=[1 + ord(text[i]) % 31 for i in range(0, len(text), 64)])

    configure_matmul_precision()
    torch.manual_seed(125)
    global_trunk = trunk(max_positions=2048)
    byte = ByteLatentLM(ByteConfig("tiny", global_trunk, 16, 2, 32, 2, 4, max_len=5, block_size=8,
                                    hash_enabled=False, hash_rows=32)).eval()
    bpe = BPELM(BPEConfig("tiny", global_trunk, 32, 0)).eval()
    identities = [{"kind": "checkpoint", "path": f"/runs/{name}-seed0/ckpt/bytes-3000000000.pt",
                   "sha256": hashlib.sha256(name.encode()).hexdigest()} for name in ("byte", "bpe")]
    bundles = [EvaluationModel(byte, byte.config, None, torch.device("cpu"), identities[0]),
               EvaluationModel(bpe, bpe.config, ShortTokenizer(), torch.device("cpu"), identities[1])]
    prompts = [f"{i:04d}".encode() + b"x" * 1020 for i in range(12)]
    warm, calibration, timed, parity = prompts[:2], prompts[2:6], prompts[6:10], prompts[10:12]
    modes = ["eager", "device-conditional"]
    verified = []
    for bundle in bundles:
        for mode in modes:
            item = verify_decode(bundle, parity, batch=2, output_bytes=24, mode=mode)
            item["identity"] = bundle.identity
            verified.append(item)
    kwargs = dict(batch=2, output_bytes=5, modes=modes, warmup=1, repetitions=2)
    calibration_results = benchmark_decode(bundles, warm + calibration, **kwargs)
    choices, details = select_modes(calibration_results, modes, len(bundles))
    result = benchmark_decode(bundles, warm + timed, selected_modes=choices, **kwargs)[0]
    result.update(prompt_kib=1, verification=verified, verify_bytes_per_row=24,
                  calibration={"requests": 2, "requested_modes": modes, "models": details})
    path = tmp_path / "actual.json"
    path.write_text(json.dumps({"results": [result]}), encoding="utf-8")
    assert problems(path, ["byte", "bpe"], [2], [1], 2, warmup=1, output_bytes=5,
                    calibration_requests=2, verify_bytes=24) == []


def test_main_calibrates_then_writes_checked_pair(tmp_path, monkeypatch, capsys):
    import hashlib
    import json
    import sys

    import scripts.bench_decode as bench
    from scripts.diag.decode_summary import problems

    class ShortTokenizer(_RaggedTokenizer):
        def encode(self, text, add_special_tokens=False):
            return SimpleNamespace(ids=[1 + ord(text[i]) % 31 for i in range(0, len(text), 64)])

    torch.manual_seed(127)
    global_trunk = trunk(max_positions=2048)
    byte = ByteLatentLM(ByteConfig("tiny", global_trunk, 16, 2, 32, 2, 4, max_len=5, block_size=8,
                                    hash_enabled=False, hash_rows=32)).eval()
    bpe = BPELM(BPEConfig("tiny", global_trunk, 32, 0)).eval()
    names = ("byte", "bpe")
    identities = [{"kind": "checkpoint", "path": f"/runs/{name}-seed0/ckpt/bytes-3000000000.pt",
                   "sha256": hashlib.sha256(name.encode()).hexdigest()} for name in names]
    bundles = [EvaluationModel(byte, byte.config, None, torch.device("cpu"), identities[0]),
               EvaluationModel(bpe, bpe.config, ShortTokenizer(), torch.device("cpu"), identities[1])]
    monkeypatch.setattr(bench, "verify_supply", lambda _: None)
    monkeypatch.setattr(bench, "load_evaluation_model",
                        lambda **kwargs: bundles[0 if kwargs["run"].name.startswith("byte") else 1])
    monkeypatch.setattr(bench, "exact_val_prompts", lambda supply, size, count, seed:
                        [f"{i:08d}".encode() + b"x" * (size - 8) for i in range(count)])
    path = tmp_path / "pair.json"
    monkeypatch.setattr(sys, "argv", ["bench_decode.py", "--compare", "/runs/byte-seed0", "/runs/bpe-seed0",
                                     "--supply", "unused", "--output", str(path), "--device", "cpu",
                                     "--mode", "device-conditional", "--prompt-kib", "1", "--batches", "1",
                                     "--output-bytes", "8", "--calibration-requests", "2"])
    bench.main()
    capsys.readouterr()
    result = json.loads(path.read_text(encoding="utf-8"))["results"][0]
    assert [model["selected_mode"] for model in result["calibration"]["models"]] == [
        "device-conditional", "device-conditional"]
    assert problems(path, list(names), [1], [1], 30, output_bytes=8, calibration_requests=2) == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph parity requires CUDA")
@pytest.mark.parametrize("family", ["byte", "bpe"])
@pytest.mark.parametrize("mode", ["graph", "graph-split"])
def test_cuda_graph_modes_match_distinct_rows_and_ragged_cache(family, mode):
    torch.manual_seed(124)
    model = (byte_model() if family == "byte" else bpe_model()).cuda().eval()
    if family == "byte":
        prompt = torch.tensor([[97, 32, 98, 99, 100, 101, 102, 103, 104],
                               [97, 98, 99, 100, 101, 102, 103, 104, 105],
                               [32, 97, 32, 98, 32, 99, 32, 100, 101]], device="cuda")
        lengths = [9, 9, 9]
    else:
        prompt = torch.tensor([[0, 1, 2, 0, 0, 0, 0, 0, 0],
                               [0, 3, 4, 5, 6, 0, 0, 0, 0],
                               [0, 7, 8, 9, 10, 11, 12, 13, 14]], device="cuda")
        lengths = [3, 5, 9]
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = model.init_state(3, 96, "cuda", torch.bfloat16)
        logits = model.prefill(reference, prompt, lengths=lengths)
        allowed = torch.ones(32, dtype=torch.bool, device="cuda")
        allowed[0] = False
        static = (StaticByteState(model, reference) if family == "byte" else
                  StaticBPEState(model, reference, logits, allowed))
        graph = (CUDASplitGraphStepper(static) if family == "byte" and mode == "graph-split"
                 else CUDAGraphStepper(static))
        start = static.snapshot()
        for _ in range(64):
            expected = (logits[:, -1, :256].argmax(dim=-1) if family == "byte" else
                        logits[:, -1].masked_fill(~allowed, -torch.inf).argmax(dim=-1))
            assert torch.equal(static.current, expected)
            logits = model.step(reference, expected)
            graph.step()
        graph.reset(start)
        assert static.consumed == max(lengths)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph parity requires CUDA")
@pytest.mark.parametrize("family", ["byte", "bpe"])
@pytest.mark.parametrize("batch", [1, 8, 32])
def test_cuda_graph_greedy_matches_eager_over_many_prompt_bytes(family, batch):
    # Batch 32 alone contributes 10,240 generated bytes for each family; rows
    # carry different prompts, and the smaller fixed batch sizes are covered too.
    torch.manual_seed(27)
    model = (byte_model() if family == "byte" else bpe_model()).cuda().eval()
    steps = 320 if batch == 32 else 24
    prompt = (byte_prompt(batch, "cuda") if family == "byte" else
              torch.tensor([[0, 1 + row % 30, 1 + row // 30, 3, 4] for row in range(batch)], device="cuda"))
    assert len({tuple(row) for row in prompt.cpu().tolist()}) == batch
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = model.init_state(batch, prompt.shape[1] + steps + 1, "cuda", torch.bfloat16)
        logits = model.prefill(reference, prompt)
        allowed = torch.ones(32, dtype=torch.bool, device="cuda")
        allowed[0] = False  # The other 31 synthetic vocabulary entries each represent one byte.
        static = (StaticByteState(model, reference) if family == "byte" else
                  StaticBPEState(model, reference, logits, allowed))
        graph = CUDAGraphStepper(static)
        start = static.snapshot()
        with pytest.raises(TypeError, match="recapture"):
            pickle.dumps(graph)
        eager_tokens = []
        for _ in range(steps):
            token = (logits[:, -1, :256].argmax(dim=-1) if family == "byte" else
                     logits[:, -1].masked_fill(~allowed, -torch.inf).argmax(dim=-1))
            eager_tokens.append(token)
            logits = model.step(reference, token)
        graph_tokens = []
        for _ in range(steps):
            graph_tokens.append(static.current.clone())
            graph.step()
        assert torch.equal(torch.stack(graph_tokens), torch.stack(eager_tokens))
        graph.reset(start)
        assert torch.equal(static.current, eager_tokens[0])
        graph.step()
        assert torch.equal(static.current, eager_tokens[1])
        # A resumed process must construct a new graph around the restored tensors.
        resumed = CUDAGraphStepper(static)
        resumed.step()
        assert torch.equal(static.current, eager_tokens[2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA speculation parity requires CUDA")
@pytest.mark.parametrize("family", ["byte", "bpe"])
def test_cuda_speculation_matches_plain_greedy(family):
    torch.manual_seed(31)
    model = (byte_model() if family == "byte" else bpe_model()).cuda().eval()
    torch.manual_seed(32)
    draft_model = bpe_model().cuda().eval() if family == "bpe" else None
    batch, steps = 32, 320
    prompt = (byte_prompt(batch, "cuda") if family == "byte" else
              torch.tensor([[0, 1 + row % 30, 2 + row // 30, 3] for row in range(batch)], device="cuda"))
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = model.init_state(batch, prompt.shape[1] + steps + 1, "cuda", torch.bfloat16)
        logits = model.prefill(reference, prompt)
        allowed = torch.ones(32, dtype=torch.bool, device="cuda")
        allowed[0] = False
        static = (StaticByteState(model, reference) if family == "byte" else
                  StaticBPEState(model, reference, logits, allowed))
        drafter = None
        draft_graph = CUDAGraphStepper(static, byte_draft=True) if family == "byte" else None
        if family == "bpe":
            draft_reference = draft_model.init_state(batch, prompt.shape[1] + steps + 1, "cuda", torch.bfloat16)
            drafter = StaticBPEState(draft_model, draft_reference, draft_model.prefill(draft_reference, prompt), allowed)
            draft_graph = CUDAGraphStepper(drafter)
        generated = 0
        while generated < steps:
            result = (speculate_byte(static, min(8, steps - generated), draft_graph=draft_graph) if family == "byte"
                      else speculate_bpe(static, drafter, min(4, steps - generated), draft_graph=draft_graph))
            for tokens in result.tokens.unbind(dim=1):
                expected = (logits[:, -1, :256].argmax(dim=-1) if family == "byte" else
                            logits[:, -1].masked_fill(~allowed, -torch.inf).argmax(dim=-1))
                assert torch.equal(tokens, expected)
                logits = model.step(reference, tokens)
            generated += result.tokens.shape[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA speculation parity requires CUDA")
def test_cuda_bpe_accepted_block_matches_reference():
    torch.manual_seed(33)
    model = bpe_model().cuda().eval()
    draft_model = copy.deepcopy(model)
    prompt = torch.tensor([[0, 1, 2, 3], [0, 3, 4, 5]], device="cuda")
    allowed = torch.ones(32, dtype=torch.bool, device="cuda")
    allowed[0] = False
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        reference = model.init_state(2, 64, "cuda", torch.bfloat16)
        logits = model.prefill(reference, prompt)
        draft_reference = draft_model.init_state(2, 64, "cuda", torch.bfloat16)
        draft_logits = draft_model.prefill(draft_reference, prompt)
        target = StaticBPEState(model, reference, logits, allowed)
        drafter = StaticBPEState(draft_model, draft_reference, draft_logits, allowed)
        graph = CUDAGraphStepper(drafter)
        accepted = 0
        for _ in range(8):
            result = speculate_bpe(target, drafter, 4, draft_graph=graph)
            accepted += result.accepted_draft_positions
            for tokens in result.tokens.unbind(dim=1):
                assert torch.equal(tokens, logits[:, -1].masked_fill(~allowed, -torch.inf).argmax(dim=-1))
                logits = model.step(reference, tokens)
        assert accepted > 0
        assert torch.equal(target.current, logits[:, -1].masked_fill(~allowed, -torch.inf).argmax(dim=-1))
