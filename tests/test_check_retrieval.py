import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "diag"))
from check_retrieval import problems  # noqa: E402


def _result(path, kibs=(2,), trials=2, fmt="v2: x"):
    cells = []
    for kib in kibs:
        for depth in (0.1, 0.5, 0.9):
            rows = [{"context_bytes": kib * 1024, "depth": depth, "candidate_nll": [0.0] * 10, "greedy_exact": i == 0}
                    for i in range(trials)]
            cells.append({"context_kib": kib, "depth": depth, "seed": kib * 1000 + int(depth * 100), "count": trials,
                          "trials": rows, "greedy_accuracy": 1 / trials, "prompt_format": fmt})
    path.write_text(json.dumps({"task": "retrieval", "prompt_format": fmt, "checkpoint": {"path": "/c.pt"}, "cells": cells}))
    return path


def test_complete_v2_result_passes(tmp_path):
    assert problems(_result(tmp_path / "r.json"), "/c.pt", [2], 2, 0) == []


def test_incomplete_or_v1_results_fail(tmp_path):
    assert problems(_result(tmp_path / "a.json"), "/c.pt", [2, 4], 2, 0)
    assert problems(_result(tmp_path / "b.json", trials=1), "/c.pt", [2], 2, 0)
    assert problems(_result(tmp_path / "c.json", fmt="v1"), "/c.pt", [2], 2, 0)
    assert problems(_result(tmp_path / "d.json"), "/other.pt", [2], 2, 0)
