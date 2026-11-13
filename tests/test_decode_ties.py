import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bench_decode import TIE_ULPS, bf16_ulp, tie_flip  # noqa: E402


def test_bf16_ulp_spacing():
    assert bf16_ulp(1.0) == 2 ** -7 and bf16_ulp(7.125) == 2 ** -5 and bf16_ulp(9.6875) == 2 ** -4
    assert bf16_ulp(-3.36) == 2 ** -6


def test_exact_and_one_ulp_ties_are_tolerated():
    reference = torch.tensor([7.125, 7.125, 1.0])
    tested = torch.tensor([7.125, 7.15625, 1.0])  # tested prefers index 1 by one ulp
    record = tie_flip(reference, tested, expected=0, selected=1)
    assert record is not None and record["reference_margin"] == 0 and record["tested_margin"] == 2 ** -5
    assert record["tolerance"] == TIE_ULPS * 2 ** -5


def test_real_mismatches_are_not_tolerated():
    reference = torch.tensor([8.0, 7.0, 1.0])
    assert tie_flip(reference, torch.tensor([8.0, 7.0, 1.0]), expected=0, selected=1) is None  # reference margin 1.0
    assert tie_flip(torch.tensor([8.0, 8.0]), torch.tensor([0.0, 9.0]), expected=0, selected=1) is None  # tested margin
    assert tie_flip(torch.tensor([float("nan"), 1.0]), None, expected=0, selected=1) is None
    assert tie_flip(torch.tensor([8.0, 8.0]), torch.tensor([float("nan"), 8.0]), expected=0, selected=1) is None


def test_tested_scores_are_required():
    assert tie_flip(torch.tensor([8.0, 8.0]), None, expected=0, selected=1) is None


def test_checker_recomputes_tie_records():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "diag"))
    from decode_summary import _valid_tie
    good = {"reference_margin": 0.0, "tested_margin": 2 ** -5, "tolerance": 4 * 2 ** -5, "reference_top_logit": 7.125,
            "row": 3, "step": 89, "expected": 101, "selected": 32}
    assert _valid_tie(good, 8)
    assert not _valid_tie({**good, "tolerance": 1.0}, 8)
    assert not _valid_tie({**good, "tested_margin": 0.2}, 8)
    assert not _valid_tie({**good, "row": 8}, 8)
    assert not _valid_tie({**good, "selected": 101}, 8)
    assert not _valid_tie({**good, "reference_margin": float("nan")}, 8)
    legacy = {k: v for k, v in good.items() if k not in ("expected", "selected")}
    assert _valid_tie(legacy, 8)
