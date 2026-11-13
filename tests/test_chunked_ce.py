from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from blt.config import BPEConfig, ByteConfig, TransformerConfig
from blt.model import BPELM, ByteLatentLM
from blt.patcher import patch_batch
from blt.train import _loss


def make_case(family, *, length=7, device="cpu", hash_enabled=True):
    trunk = TransformerConfig(1, 8, 2, 1, 4, 16, max_positions=32)
    if family == "bpe":
        model = BPELM(BPEConfig("tiny-bpe", trunk, 29, 0))
        inputs = {"tokens": torch.arange(3 * length).reshape(3, length).remainder(29).to(device)}
        classes = 29
    else:
        config = ByteConfig("tiny-byte", trunk, 8, 2, 16, 2, 4, max_len=4, block_size=8,
                            dec_layers=1, hash_enabled=hash_enabled, hash_rows=16)
        model = ByteLatentLM(config)
        raw = np.full((3, length), ord("a"), dtype=np.uint8)
        patches = patch_batch(raw, config.min_len, config.max_len, config.block_size)
        inputs = {"bytes": torch.from_numpy(raw.astype(np.int64)).to(device),
                  "patches": {"ends": torch.from_numpy(patches.ends).to(device),
                              "patch_ids": torch.from_numpy(patches.patch_ids).to(device)}}
        classes = config.output_classes
    targets = torch.arange(3 * length, device=device).reshape(3, length).remainder(classes)
    targets[0, -2:] = -100
    targets[1] = -100
    targets[2, 0] = -100
    return model.to(device), inputs, targets


def reference_loss(model, inputs, targets):
    logits = model(inputs)
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
                           ignore_index=-100, reduction="sum")


def assert_gradients_close(actual, expected, *, rtol, atol):
    for (name, left), (other_name, right) in zip(actual.named_parameters(), expected.named_parameters()):
        assert name == other_name
        assert (left.grad is None) == (right.grad is None), name
        if left.grad is not None:
            torch.testing.assert_close(left.grad, right.grad, rtol=rtol, atol=atol, msg=lambda msg: f"{name}: {msg}")


def assert_gradients_close_normalized(actual, expected, *, tolerance):
    for (name, left), (_, right) in zip(actual.named_parameters(), expected.named_parameters()):
        if right.grad is not None:
            error = (left.grad - right.grad).abs().max() / right.grad.abs().max().clamp_min(1e-12)
            assert error <= tolerance, f"{name}: normalised gradient error {error:.3g} > {tolerance}"


@pytest.mark.parametrize("family,hash_enabled", [("bpe", True), ("byte", True), ("byte", False)])
def test_chunked_loss_matches_full_head_and_gradients_on_cpu(family, hash_enabled):
    torch.manual_seed(10)
    model, inputs, targets = make_case(family, hash_enabled=hash_enabled)
    reference = copy.deepcopy(model)

    hidden = model.hidden(inputs)
    assert torch.equal(hidden, model(inputs, return_hidden=True))
    assert torch.equal(model(inputs), F.linear(hidden, model.output_weight))
    assert model.output_weight is (model.embedding.weight if family == "bpe" else model.head.weight)
    assert model.state_dict().keys() == reference.state_dict().keys()

    # Four positions cut across sequences of length seven; one whole chunk is ignored.
    actual = _loss(model, inputs, targets, chunk_size=4)
    expected = reference_loss(reference, inputs, targets)
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=1e-5)
    actual.backward()
    expected.backward()
    assert_gradients_close(model, reference, rtol=2e-6, atol=2e-6)


def test_chunked_loss_matches_gpt2_vocabulary_with_tied_embedding():
    torch.manual_seed(19)
    trunk = TransformerConfig(1, 8, 2, 1, 4, 16, max_positions=16)
    model = BPELM(BPEConfig("tiny-50k", trunk, 50257, 50256))
    reference = copy.deepcopy(model)
    tokens = torch.tensor([[50256, 1, 50255, 4, 8, 15, 16], [50256, 23, 42, 9, 5, 3, 2]])
    targets = torch.tensor([[1, 50255, 4, 8, 15, -100, -100], [23, 42, 9, -100, 3, 2, 50256]])
    actual = _loss(model, {"tokens": tokens}, targets, chunk_size=4)
    expected = reference_loss(reference, {"tokens": tokens}, targets)
    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=1e-5)
    actual.backward()
    expected.backward()
    assert_gradients_close(model, reference, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("family", ["bpe", "byte"])
def test_head_is_bounded_and_recomputed_during_backward(monkeypatch, family):
    torch.manual_seed(11)
    model, inputs, targets = make_case(family)
    classes = model.output_weight.shape[0]
    observed, saved_shapes = [], []
    cross_entropy = F.cross_entropy

    def record(logits, target, *args, **kwargs):
        observed.append((tuple(logits.shape), logits.dtype))
        return cross_entropy(logits, target, *args, **kwargs)

    def save(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    monkeypatch.setattr(F, "cross_entropy", record)
    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        loss = _loss(model, inputs, targets, chunk_size=4)
    expected_shapes = [(4, classes)] * 5 + [(1, classes)]
    assert [shape for shape, _ in observed] == expected_shapes
    assert all(dtype == torch.float32 for _, dtype in observed)
    assert not any(shape in saved_shapes for shape in expected_shapes)
    loss.backward()
    assert sorted(shape for shape, _ in observed[6:]) == sorted(expected_shapes)

    observed.clear()
    with torch.no_grad():
        validation = _loss(model, inputs, targets, chunk_size=4)
    assert [shape for shape, _ in observed] == expected_shapes
    torch.testing.assert_close(validation, loss.detach(), rtol=0, atol=0)


@pytest.mark.parametrize("family", ["bpe", "byte"])
def test_all_ignored_targets_produce_zero_loss_and_gradients(family):
    torch.manual_seed(12)
    model, inputs, targets = make_case(family)
    targets.fill_(-100)
    loss = _loss(model, inputs, targets, chunk_size=4)
    assert loss.item() == 0
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.count_nonzero(parameter.grad) == 0


def test_default_chunk_size_limits_head_positions(monkeypatch):
    class HiddenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.output_weight = torch.nn.Parameter(torch.randn(29, 8))

        def forward(self, inputs, *, return_hidden=False):
            assert return_hidden
            return inputs

    model = HiddenModel()
    hidden = torch.randn(2, 1031, 8, requires_grad=True)
    targets = torch.zeros((2, 1031), dtype=torch.long)
    lengths = []
    cross_entropy = F.cross_entropy

    def record(logits, target, *args, **kwargs):
        lengths.append(logits.shape[0])
        return cross_entropy(logits, target, *args, **kwargs)

    monkeypatch.setattr(F, "cross_entropy", record)
    loss = _loss(model, hidden, targets)
    assert lengths == [1024, 1024, 14]
    loss.backward()
    assert sorted(lengths[3:]) == [14, 1024, 1024]


@pytest.mark.parametrize("family", ["bpe", "byte"])
def test_compiled_model_reuses_graphs_for_repeated_bucket_shapes(family):
    torch._dynamo.reset()
    torch.manual_seed(13)
    model, _, _ = make_case(family)
    graphs = []

    def backend(graph, _example_inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)
    graph_counts = []
    for length in (6, 10, 6, 10):
        _, inputs, targets = make_case(family, length=length)
        model.zero_grad(set_to_none=True)
        _loss(compiled, inputs, targets, chunk_size=4).backward()
        graph_counts.append(len(graphs))
    assert graph_counts == [1, 2, 2, 2]


@pytest.mark.parametrize("family", ["bpe", "byte"])
def test_aot_compiled_hidden_and_checkpointed_head_match_eager(family):
    torch._dynamo.reset()
    torch.manual_seed(15)
    model, inputs, targets = make_case(family)
    reference = copy.deepcopy(model)
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True, dynamic=False)
    actual = _loss(compiled, inputs, targets, chunk_size=4)
    expected = _loss(reference, inputs, targets, chunk_size=4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    assert_gradients_close(model, reference, rtol=2e-5, atol=1e-5)


@pytest.mark.parametrize("family", ["bpe", "byte"])
def test_full_loss_can_be_compiled_with_checkpointing(family):
    torch._dynamo.reset()
    torch.manual_seed(17)
    model, inputs, targets = make_case(family)
    reference = copy.deepcopy(model)

    def calculate(batch, expected):
        return _loss(model, batch, expected, chunk_size=4)

    compiled = torch.compile(calculate, backend="aot_eager", fullgraph=True, dynamic=False)
    actual = compiled(inputs, targets)
    expected = _loss(reference, inputs, targets, chunk_size=4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    assert_gradients_close(model, reference, rtol=2e-5, atol=1e-5)


@pytest.mark.parametrize("family", ["bpe", "byte"])
def test_cpu_bf16_loss_and_gradients_match_full_head(monkeypatch, family):
    torch.manual_seed(16)
    model, inputs, targets = make_case(family)
    reference = copy.deepcopy(model)
    observed = []
    linear = F.linear

    def record(hidden, weight, bias=None):
        output = linear(hidden, weight, bias)
        if weight is model.output_weight:
            observed.append((hidden.shape[0], output.dtype))
        return output

    with monkeypatch.context() as scoped:
        scoped.setattr(F, "linear", record)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = _loss(model, inputs, targets, chunk_size=4)
        actual.backward()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = reference_loss(reference, inputs, targets)
    expected.backward()
    assert len(observed) == 12
    assert all(size <= 4 and dtype == torch.bfloat16 for size, dtype in observed)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
    assert_gradients_close(model, reference, rtol=0.05, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 parity requires a CUDA-enabled PyTorch build")
@pytest.mark.parametrize("family", ["bpe", "byte"])
def test_cuda_bf16_loss_and_gradients_match_full_head(monkeypatch, family):
    torch.manual_seed(14)
    model, inputs, targets = make_case(family, device="cuda")
    reference = copy.deepcopy(model)
    observed = []
    linear = F.linear

    def record(hidden, weight, bias=None):
        output = linear(hidden, weight, bias)
        if weight is model.output_weight:
            observed.append((hidden.shape[0], output.dtype))
        return output

    with monkeypatch.context() as scoped:
        scoped.setattr(F, "linear", record)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actual = _loss(model, inputs, targets, chunk_size=4)
        actual.backward()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = reference_loss(reference, inputs, targets)
    expected.backward()
    assert len(observed) == 12
    assert all(size <= 4 and dtype == torch.bfloat16 for size, dtype in observed)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.02)
    assert_gradients_close(model, reference, rtol=0.05, atol=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA bucket compilation requires a CUDA-enabled PyTorch build")
def test_cuda_compiled_buckets_reuse_graphs_across_chunked_head(monkeypatch):
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    torch.manual_seed(18)
    trunk = TransformerConfig(1, 8, 2, 1, 4, 16, max_positions=512)
    model = BPELM(BPEConfig("tiny-cuda", trunk, 29, 0)).cuda()
    reference = copy.deepcopy(model)
    compiled = torch.compile(model, fullgraph=True, dynamic=False)
    cross_entropy = F.cross_entropy
    observed, graph_deltas = [], []

    def record(logits, target, *args, **kwargs):
        observed.append(logits.shape[0])
        return cross_entropy(logits, target, *args, **kwargs)

    for length, expected_chunks in ((256, [768]), (512, [1024, 512]), (256, [768])):
        tokens = torch.arange(3 * length, device="cuda").reshape(3, length).remainder(29)
        targets = tokens.roll(1, dims=1)
        targets[1, length // 2:] = -100
        model.zero_grad(set_to_none=True)
        reference.zero_grad(set_to_none=True)
        graphs_before = counters["stats"].get("unique_graphs", 0)
        observed.clear()
        with monkeypatch.context() as scoped:
            scoped.setattr(F, "cross_entropy", record)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                actual = _loss(compiled, {"tokens": tokens}, targets)
            assert observed == expected_chunks
            actual.backward()
        assert sorted(observed[len(expected_chunks):]) == sorted(expected_chunks)
        graph_deltas.append(counters["stats"].get("unique_graphs", 0) - graphs_before)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            expected = reference_loss(reference, {"tokens": tokens}, targets)
        expected.backward()
        torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.1)
        # Compiled bf16 reorders reductions, so a few near-zero gradient entries miss an elementwise
        # tolerance; exactness is covered by the fp32 tests, here we bound the error relative to each tensor's scale.
        assert_gradients_close_normalized(model, reference, tolerance=0.05)
    assert graph_deltas[0] >= 1 and graph_deltas[1] >= 1 and graph_deltas[2] == 0


def test_invalid_loss_chunk_size_is_rejected():
    model, inputs, targets = make_case("bpe")
    with pytest.raises(ValueError, match="chunk size"):
        _loss(model, inputs, targets, chunk_size=0)
