"""Summarise PIQA/HellaSwag results under the frozen task rules (~/blt/results/frozen-analysis.json, "tasks").

Primary metric: byte-normalised accuracy; secondary: raw accuracy. Aggregate: equal-weight mean of the two tasks.
Per model: accuracy with a 95% item-bootstrap interval and the above-chance check (PIQA >= 55% and HellaSwag >= 30%,
each interval excluding 50% / 25%). Primary pair M-P6 vs M-BPE16K per seed: paired item bootstrap (2000 draws, seed 0,
items resampled independently within each task) of the aggregate difference byte - BPE in points, with the one-sided
95% lower bound. Retention: unsupported unless both models are above chance; non-inferior if the lower bound is
>= -2 points; otherwise "point estimate within 2 points" if the drop is <= 2, else fail.

usage: python scripts/diag/task_summary.py [ROOT] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TASKS = {"piqa": (1838, 0.55, 0.50), "hellaswag": (10042, 0.30, 0.25)}
RUNS = ["s-bpe50k-seed0", "s-bpe16k-seed0", "s-p6-seed0", "s-p8-seed0", "m-bpe50k-seed0", "m-bpe16k-seed0",
        "m-p6-seed0", "m-p8-seed0", "m-p6-nohash-seed0", "xs-bpe50k-seed0", "xs-bpe16k-seed0",
        "m-bpe16k-seed1", "m-p6-seed1"]
PAIRS = [("m-p6-seed0", "m-bpe16k-seed0"), ("m-p6-seed1", "m-bpe16k-seed1")]
DRAWS, SEED = 2000, 0


def load(root: Path, run: str, task: str) -> dict | None:
    path = root / f"{run}-{task}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    count = TASKS[task][0]
    examples = sorted(data["examples"], key=lambda row: row["index"])
    if data["count"] != count or len(examples) != count or [row["index"] for row in examples] != list(range(count)):
        raise SystemExit(f"{path}: expected the full {count}-item validation split")
    correct = {metric: np.array([row[key] == row["label"] for row in examples], dtype=float)
               for metric, key in (("byte_normalised", "prediction_byte_normalised"), ("raw", "prediction"))}
    return {"correct": correct, "items": [(row["context"], tuple(row["candidates"]), row["label"]) for row in examples],
            "checkpoint_sha256": data["checkpoint"]["sha256"]}


def interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    draws = values[rng.integers(0, len(values), (DRAWS, len(values)))].mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=Path.home() / "blt/results/tasks")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    summary: dict = {"models": {}, "pairs": {}, "rules": "frozen-analysis.json tasks"}
    results = {run: {task: load(args.root, run, task) for task in TASKS} for run in RUNS}
    print("| model | PIQA (norm) | HellaSwag (norm) | aggregate (norm) | PIQA raw | HellaSwag raw | above chance |")
    print("|---|---|---|---|---|---|---|")
    for run, tasks in results.items():
        if any(value is None for value in tasks.values()):
            print(f"| {run} | missing | | | | | |")
            continue
        entry, above = {}, True
        for task, (_, floor, chance) in TASKS.items():
            for metric in ("byte_normalised", "raw"):
                values = tasks[task]["correct"][metric]
                low, high = interval(values, np.random.default_rng(SEED))
                entry[f"{task}_{metric}"] = {"accuracy": float(values.mean()), "ci95": [low, high]}
            primary = entry[f"{task}_byte_normalised"]
            above = above and primary["accuracy"] >= floor and primary["ci95"][0] > chance
        entry["aggregate_byte_normalised"] = float(np.mean([entry[f"{t}_byte_normalised"]["accuracy"] for t in TASKS]))
        entry["aggregate_raw"] = float(np.mean([entry[f"{t}_raw"]["accuracy"] for t in TASKS]))
        entry["above_chance"] = above
        summary["models"][run] = entry
        print(f"| {run} | {100 * entry['piqa_byte_normalised']['accuracy']:.1f} | "
              f"{100 * entry['hellaswag_byte_normalised']['accuracy']:.1f} | {100 * entry['aggregate_byte_normalised']:.1f} | "
              f"{100 * entry['piqa_raw']['accuracy']:.1f} | {100 * entry['hellaswag_raw']['accuracy']:.1f} | {'yes' if above else 'no'} |")
    print()
    for byte_run, bpe_run in PAIRS:
        if byte_run not in summary["models"] or bpe_run not in summary["models"]:
            print(f"{byte_run} vs {bpe_run}: missing")
            continue
        record = {}
        for metric in ("byte_normalised", "raw"):
            rng = np.random.default_rng(SEED)
            point, draws = 0.0, np.zeros(DRAWS)
            for task in TASKS:
                a, b = results[byte_run][task], results[bpe_run][task]
                if a["items"] != b["items"]:
                    raise SystemExit(f"{task}: the two models did not score identical items")
                difference = a["correct"][metric] - b["correct"][metric]
                point += 0.5 * difference.mean()
                draws += 0.5 * difference[rng.integers(0, len(difference), (DRAWS, len(difference)))].mean(axis=1)
            record[metric] = {"difference_points": 100 * point,
                              "ci95_points": [100 * float(np.quantile(draws, 0.025)), 100 * float(np.quantile(draws, 0.975))],
                              "one_sided_lower95_points": 100 * float(np.quantile(draws, 0.05))}
        primary = record["byte_normalised"]
        if not (summary["models"][byte_run]["above_chance"] and summary["models"][bpe_run]["above_chance"]):
            verdict = "unsupported: a model is not above chance"
        elif primary["one_sided_lower95_points"] >= -2:
            verdict = "pass: non-inferior within 2 points"
        elif primary["difference_points"] >= -2:
            verdict = "point estimate within 2 points; non-inferiority not shown"
        else:
            verdict = "fail: aggregate drop larger than 2 points"
        record["verdict"] = verdict
        summary["pairs"][f"{byte_run}--{bpe_run}"] = record
        print(f"{byte_run} - {bpe_run}: aggregate (norm) {primary['difference_points']:+.2f} points "
              f"[{primary['ci95_points'][0]:+.2f}, {primary['ci95_points'][1]:+.2f}], one-sided lower "
              f"{primary['one_sided_lower95_points']:+.2f}; raw {record['raw']['difference_points']:+.2f}; {verdict}")
    if args.json:
        args.json.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
