from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from blt.router import (RouterProgram, _replay_boundaries, adjust_boundary_logprobs, adjust_joint_logits, audit,
                        edit_paired_rows, restore, snapshot, symmetrise_qk)
from blt.trackA import ReferenceModel, bolmo_ids, folded_log_probs, load_reference, score_bolmo_chunk, score_olmo_chunk
from blt.trackA_bench import forced_boundary, forced_mask, run_bolmo


class Predictor(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.q_proj_layer = nn.Linear(width, width, bias=False)
        self.k_proj_layer = nn.Linear(width, width, bias=False)
        self.boundary_predictor_lookahead = 1
        self.boundary_threshold = "sample:0"

    def forward(self, hidden_states):
        logp = torch.full(hidden_states.shape[:2], -1.0, device=hidden_states.device)
        logp[:, 0] = 0
        logp[:, -1] = -100000
        return logp, logp > math.log(0.5)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.boundary_predictor_module = Predictor()

    def forward(self, input_ids):
        h = input_ids.float().unsqueeze(-1).expand(-1, -1, 4) / 100
        logp, mask = self.boundary_predictor_module(h)
        return h, h, logp, mask


class FakeBolmo(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.local_encoder = Encoder()
        self.lm_head = nn.Linear(4, 520, bias=False)
        self.register_buffer("scalar", torch.tensor(2))

    def forward(self, input_ids, logits_to_keep=0, boundary_state=None):
        hidden, _, _, _ = self.model.local_encoder(input_ids)
        return self.lm_head(hidden[:, -logits_to_keep:] if logits_to_keep else hidden)


def test_folded_bpb_uses_full_softmax_and_keeps_special_mass():
    logits = torch.zeros(2, 3, 520)
    logits[..., 0] = 4
    folded = folded_log_probs(logits)
    expected = torch.logaddexp(logits[..., 4:260], logits[..., 264:520]) - torch.logsumexp(logits, -1, keepdim=True)
    torch.testing.assert_close(folded, expected)
    assert (folded.exp().sum(-1) < 1).all()


def test_bolmo_scorer_aligns_bos_and_every_raw_target():
    class Causal(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.local_encoder = Encoder()
            self.model.local_encoder.free_inference_cache = lambda: None
            self.model.local_decoder = SimpleNamespace(free_inference_cache=lambda: None)
            self.seen = None
        def forward(self, input_ids, use_cache=False):
            self.seen = input_ids.clone()
            self.model.local_encoder(input_ids)
            logits = torch.zeros(*input_ids.shape, 520)
            logits[0, 0, 4 + ord("A")] = math.log(5)
            logits[0, 1, 264 + ord("?")] = math.log(7)
            return SimpleNamespace(logits=logits)
    model = Causal()
    bundle = ReferenceModel("bolmo", model, None, {}, torch.device("cpu"), 1)
    result = score_bolmo_chunk(bundle, b"A?", byte_nll=True)
    assert model.seen.tolist() == [[1, 4 + ord("A"), 4 + ord("?")]]
    assert result["patches"] == 1
    expected = [math.log(524 / 6), math.log(526 / 8)]
    assert result["byte_nll"] == pytest.approx(expected, rel=1e-6)


def test_olmo_scorer_uses_bos_then_previous_tokens():
    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            assert text == "ab" and not add_special_tokens
            return [2, 3]
        def decode(self, ids, **kwargs):
            assert ids == [2, 3]
            return "ab"
    class Causal(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(max_position_embeddings=10)
            self.seen = None
        def forward(self, input_ids, use_cache=False):
            self.seen = input_ids.clone()
            logits = torch.zeros(1, 2, 5)
            logits[0, 0, 2] = math.log(5)
            logits[0, 1, 3] = math.log(7)
            return SimpleNamespace(logits=logits)
    model = Causal()
    bundle = ReferenceModel("olmo", model, Tokenizer(), {}, torch.device("cpu"), 1)
    result = score_olmo_chunk(bundle, b"ab")
    assert model.seen.tolist() == [[1, 2]]
    assert result["nats"] == pytest.approx(math.log(9 / 5) + math.log(11 / 7), rel=1e-6)
    assert result["bytes_per_token"] == 1


def test_arm_p_preserves_each_folded_mass_and_changes_boundary_odds():
    torch.manual_seed(41)
    logits = torch.randn(2, 3, 520)
    prior = torch.randn(2, 3, 256)
    modified = adjust_joint_logits(logits, prior, beta=-0.8, strength=1.7)
    torch.testing.assert_close(folded_log_probs(modified), folded_log_probs(logits), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(modified[..., :4], logits[..., :4])
    torch.testing.assert_close(modified[..., 260:264], logits[..., 260:264])
    expected = logits[..., 264:520] - logits[..., 4:260] - 0.8 + 1.7 * prior
    torch.testing.assert_close(modified[..., 264:520] - modified[..., 4:260], expected, rtol=1e-5, atol=2e-6)


def test_prefill_log_odds_transform_preserves_sentinels():
    lp = torch.tensor([0.0, math.log(0.25), math.log(0.8), -100000.0])
    shifted = adjust_boundary_logprobs(lp, torch.tensor([3.0, 0.7, -0.4, -3.0]))
    expected = torch.sigmoid(torch.logit(lp[1:3].exp()) + torch.tensor([0.7, -0.4]))
    torch.testing.assert_close(shifted[1:3].exp(), expected)
    assert shifted[0] == 0 and shifted[-1] == -100000


def test_arm_w_and_snapshot_are_reversible():
    torch.manual_seed(4)
    model = FakeBolmo()
    saved = snapshot(model)
    q = model.model.local_encoder.boundary_predictor_module.q_proj_layer.weight.detach().clone()
    k = model.model.local_encoder.boundary_predictor_module.k_proj_layer.weight.detach().clone()
    old_head = model.lm_head.weight.detach().clone()
    direction, eta = torch.arange(4.0), torch.linspace(-2, 2, 256)
    symmetrise_qk(model, 0.25)
    edit_paired_rows(model, direction, eta)
    torch.testing.assert_close(model.model.local_encoder.boundary_predictor_module.q_proj_layer.weight, (q+k)/2 + 0.25*(q-k)/2)
    torch.testing.assert_close(model.model.local_encoder.boundary_predictor_module.k_proj_layer.weight, (q+k)/2 - 0.25*(q-k)/2)
    delta = eta[:, None] * direction[None, :] / 2
    torch.testing.assert_close(model.lm_head.weight[4:260], old_head[4:260] - delta)
    torch.testing.assert_close(model.lm_head.weight[264:520], old_head[264:520] + delta)
    changed = audit(model, saved)
    assert {row["name"] for row in changed} == {"lm_head.weight", "model.local_encoder.boundary_predictor_module.q_proj_layer.weight",
                                                "model.local_encoder.boundary_predictor_module.k_proj_layer.weight"}
    assert all(row["sha256_before"] != row["sha256_after"] and row["max_abs_delta"] > 0 for row in changed)
    restore(model, saved)
    assert audit(model, saved) == []


def test_router_program_hooks_direct_encoder_and_full_forward():
    model = FakeBolmo()
    ids = torch.tensor([[1, 4+10, 4+20, 4+30]])
    table = torch.zeros(256, 256)
    table[10, 20] = 2
    old_logits = model(ids)
    with RouterProgram(model):
        assert torch.equal(model(ids), old_logits)
    with RouterProgram(model, table, beta_in=0.4, beta_out=0.2, strength=1):
        _, _, lp, mask = model.model.local_encoder.forward(ids)
        assert lp[0, 0] == 0 and lp[0, -1] == -100000
        assert bool(mask[0, 2])
        edited_logits = model(ids)
        torch.testing.assert_close(folded_log_probs(edited_logits), folded_log_probs(old_logits), atol=2e-6, rtol=1e-5)
        old_odds = old_logits[0, 1, 264+20] - old_logits[0, 1, 4+20]
        new_odds = edited_logits[0, 1, 264+20] - edited_logits[0, 1, 4+20]
        torch.testing.assert_close(new_odds - old_odds, torch.tensor(2.2), atol=2e-6, rtol=1e-5)
        with pytest.raises(RuntimeError):
            with RouterProgram(model):
                pass
    torch.testing.assert_close(model(ids), old_logits)


@pytest.mark.parametrize("p", [4.4, 6, 8, 10, 12, 16])
def test_forced_layout_has_bos_and_expected_patch_count(p):
    ids = bolmo_ids(b"x" * 221, "cpu", batch=2)
    mask = forced_mask(ids, p)
    assert mask[:, 0].all() and not mask[:, -1].any()
    expected = 1 + sum(forced_boundary(i, p) for i in range(1, 221))
    assert mask.sum(-1).tolist() == [expected, expected]
    if p == 4.4:
        assert sum(forced_boundary(i, p) for i in range(1, 221)) == 50


def _checkpoint(env, default):
    path = Path(os.environ.get(env, default)).expanduser()
    if not torch.cuda.is_available() or not path.is_dir():
        pytest.skip("requires CUDA and local Track A checkpoints")
    return path


@pytest.fixture(scope="module")
def bolmo():
    path = _checkpoint("TRACKA_BOLMO", "~/blt/models/Bolmo-1B")
    return load_reference("bolmo", path, bolmo_subword_tokenizer=os.environ.get("TRACKA_BOLMO_TOKENIZER"))


@pytest.fixture(scope="module")
def olmo():
    return load_reference("olmo", _checkpoint("TRACKA_OLMO", "~/blt/models/OLMo-2-0425-1B"))


@torch.inference_mode()
def test_cuda_bolmo_folded_bpb_matches_slow_calculation(bolmo):
    raw = b"A short test."
    result = score_bolmo_chunk(bolmo, raw)
    logits = bolmo.model(input_ids=bolmo_ids(raw, bolmo.device), use_cache=False).logits[0, :-1].float()
    slow = sum(float((torch.logsumexp(row, 0) - torch.logaddexp(row[4 + byte], row[264 + byte])).item())
               for row, byte in zip(logits, raw))
    assert math.isfinite(result["bpb"])
    assert result["nats"] == pytest.approx(slow, rel=2e-5, abs=2e-5)


@torch.inference_mode()
def test_cuda_forced_mask_reaches_stock_generation(bolmo):
    raw = b"0123456789abcdefghijklmnopqrstuvwxyz"
    seen, encoder_masks = [], []
    old_forward = bolmo.model.forward
    encoder = bolmo.model.model.local_encoder
    old_encoder = encoder.forward
    def capture(*args, **kwargs):
        if kwargs.get("boundary_mask") is not None:
            seen.append(kwargs["boundary_mask"].clone())
        return old_forward(*args, **kwargs)
    def capture_encoder(*args, **kwargs):
        result = old_encoder(*args, **kwargs)
        if result[3] is not None:
            encoder_masks.append(result[3].clone())
        return result
    bolmo.model.forward = capture
    encoder.forward = capture_encoder
    try:
        run_bolmo(bolmo, raw, 1, 2, p=6)
    finally:
        bolmo.model.forward = old_forward
        encoder.forward = old_encoder
    expected = forced_mask(bolmo_ids(raw, bolmo.device), 6)
    assert len(seen) == 1 and torch.equal(seen[0], expected[:, :-1])
    assert len(encoder_masks) >= 2 and torch.equal(encoder_masks[0], expected)
    assert torch.equal(encoder_masks[1], expected[:, :-1])


def test_cuda_replay_routes_every_byte_from_bos(bolmo):
    raw = b"ab=19;"
    with RouterProgram(bolmo.model):
        decisions = _replay_boundaries(bolmo.model, raw)
    assert len(decisions) == len(raw) and all(isinstance(value, bool) for value in decisions)


@torch.inference_mode()
def test_cuda_olmo_last_only_prefill_matches_full(olmo):
    ids = torch.tensor([[olmo.bos_id, *olmo.tokenizer.encode("A short test.", add_special_tokens=False)]], device=olmo.device)
    full = olmo.model(input_ids=ids, use_cache=False).logits[:, -1:]
    last = olmo.model(input_ids=ids, use_cache=False, logits_to_keep=1).logits
    torch.testing.assert_close(last, full, atol=2e-2, rtol=2e-2)
