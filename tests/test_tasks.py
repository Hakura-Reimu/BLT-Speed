import math
from types import SimpleNamespace

import pytest
import torch

from blt.config import BPEConfig, ByteConfig, TransformerConfig
from blt.model import ByteLatentLM
from blt.tasks import (continuation_nll, evaluate_candidates, format_candidates, retrieval_cases,
                       robustness_pairs, states_at_aligned_bytes)


def trunk():
    return TransformerConfig(1, 16, 2, 1, 8, 32, max_positions=512)


class HandByteModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = ByteConfig("tiny", trunk(), 16, 2, 32, 3, 6, max_len=7, hash_enabled=False)

    def forward(self, inputs):
        logits = torch.zeros((*inputs.shape, 257), device=inputs.device)
        logits[..., ord("a")] = 2
        logits[..., ord("b")] = 1
        return logits


class MergeTokenizer:
    tokens = {"a": 0, "b": 1, "ab": 2}
    def encode(self, text, add_special_tokens=False):
        return SimpleNamespace(ids=[self.tokens[text]] if text else [])
    def decode(self, ids, skip_special_tokens=False):
        inverse = {value: key for key, value in self.tokens.items()}
        return "".join(inverse[value] for value in ids)


class HandBPEModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = BPEConfig("tiny", trunk(), 5, 4)

    def forward(self, inputs):
        logits = torch.zeros((*inputs.shape, 5), device=inputs.device)
        logits[..., 1] = torch.where(inputs == 0, 2.0, 0.0)
        return logits


def test_retrieval_cases_are_seeded_and_have_requested_depth():
    filler = ("Held-out filler with words, accents é and punctuation. " * 40).encode("utf-8")
    cases = retrieval_cases(filler, 512, 0.5, 4, 12)
    assert cases == retrieval_cases(filler, 512, 0.5, 4, 12)
    for case in cases:
        raw = case["context"].encode("utf-8")
        marker = f"The pass key is {case['key']}.".encode("ascii")
        assert len(raw) == 512
        assert raw[case["insertion_byte"]:case["insertion_byte"] + len(marker)] == marker
        assert abs(case["insertion_byte"] / 512 - 0.5) <= 3 / 512
        assert len(set(case["distractors"])) == 9 and case["key"] not in case["distractors"]


def test_piqa_and_hellaswag_candidate_scores_have_hand_checked_answer():
    model = HandByteModel()
    log_z = math.log(math.exp(2) + math.exp(1) + 255)
    piqa = {"goal": "pick a letter", "sol1": "a", "sol2": "b", "label": 0}
    context, choices, label = format_candidates("piqa", piqa)
    assert context == "Question: pick a letter\nAnswer:" and choices == [" a", " b"] and label == 0
    assert continuation_nll(model, context, choices[0]) == pytest.approx(2 * log_z - 2, abs=1e-5)
    assert evaluate_candidates(model, [piqa], "piqa", limit=1)["accuracy"] == 1
    record = {"activity_label": "Activity", "ctx_a": "first", "ctx_b": "SECOND",
              "endings": ["a", "b", "c", "d"], "label": "0"}
    context, choices, _ = format_candidates("hellaswag", record)
    assert context == "Activity: first Second" and choices == [" a", " b", " c", " d"]
    result = evaluate_candidates(model, [record], "hellaswag", limit=1)
    assert result["accuracy"] == result["accuracy_byte_normalised"] == 1


def test_bpe_continuation_forces_boundary_when_a_merge_would_cross_it():
    tokenizer, model = MergeTokenizer(), HandBPEModel()
    assert tokenizer.encode("ab").ids == [2]
    assert tokenizer.encode("a").ids + tokenizer.encode("b").ids == [0, 1]
    expected = math.log(math.exp(2) + 4) - 2
    assert continuation_nll(model, "a", "b", tokenizer=tokenizer) == pytest.approx(expected, abs=1e-6)


def test_robustness_edits_and_state_hook_are_causal():
    pairs = robustness_pairs("A courier delivers parcel number 42 to the front door.", limit=20)
    assert {"extra_space", "newline", "casing", "digit"}.issubset({pair["kind"] for pair in pairs})
    config = ByteConfig("tiny", trunk(), 16, 2, 32, 3, 6, max_len=7, hash_rows=32)
    model = ByteLatentLM(config).eval()
    states = states_at_aligned_bytes(model, ["hello abc", "hello xyz"], [5, 5])
    for name in ("global", "final"):
        torch.testing.assert_close(states[0][name], states[1][name], atol=1e-6, rtol=1e-6)
