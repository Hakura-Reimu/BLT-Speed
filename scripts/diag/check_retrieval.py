"""Check a retrieval-screen result for completeness: prompt format v2 at result and cell level, the expected
checkpoint path, all KiB x depth cells (depths 0.1/0.5/0.9) with the frozen cell seeds, >= 200 trials per cell, and
trial metadata consistent with its cell. Exit status 0 only if the file is complete.

usage: python scripts/diag/check_retrieval.py FILE --checkpoint PATH [--kib 2 4 8] [--trials 200] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEPTHS = (0.1, 0.5, 0.9)


def problems(path: Path, checkpoint: str, kibs: list[int], trials: int, seed: int) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return [f"unreadable: {error}"]
    errors = []
    if data.get("task") != "retrieval" or not str(data.get("prompt_format", "")).startswith("v2"):
        errors.append("not a v2 retrieval result")
    if data.get("checkpoint", {}).get("path") != checkpoint:
        errors.append(f"checkpoint {data.get('checkpoint', {}).get('path')} is not {checkpoint}")
    cells = {(cell.get("context_kib"), cell.get("depth")): cell for cell in data.get("cells", [])}
    if len(cells) != len(data.get("cells", [])):
        errors.append("duplicate cells")
    for kib in kibs:
        for depth in DEPTHS:
            cell = cells.get((kib, depth))
            if cell is None:
                errors.append(f"missing cell {kib} KiB depth {depth}")
                continue
            rows = cell.get("trials", [])
            if (cell.get("seed") != seed + kib * 1000 + int(depth * 100) or cell.get("count") != len(rows)
                    or len(rows) < trials or not str(cell.get("prompt_format", "")).startswith("v2")
                    or any(row.get("context_bytes") != kib * 1024 or row.get("depth") != depth
                           or len(row.get("candidate_nll", [])) != 10 for row in rows)
                    or abs(sum(row["greedy_exact"] for row in rows) / max(1, len(rows)) - cell.get("greedy_accuracy", -1)) > 1e-12):
                errors.append(f"cell {kib} KiB depth {depth} is incomplete or inconsistent")
    if set(cells) - {(kib, depth) for kib in kibs for depth in DEPTHS}:
        errors.append("unexpected cells")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kib", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    errors = problems(args.file, args.checkpoint, args.kib, args.trials, args.seed)
    for error in errors:
        print(f"{args.file}: {error}", file=sys.stderr)
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
