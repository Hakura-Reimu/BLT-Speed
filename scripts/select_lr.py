from __future__ import annotations

"""Learning-rate selection frozen after GPT-6.1 Sol reviews (review rounds 7 and 8).

screen:  the 150 MB screening runs (cosine to 150 MB) are scored on the full validation split. For each anchor
         (S-BPE50K, S-P6, M-BPE50K, M-P8) the best two eligible screened LRs are rerun from fresh initialisation to
         300 MB on the 3 GB schedule; each 16K baseline gets 150 MB on the 3 GB schedule at the transferred 50K
         screening winner and its better adjacent screened LR. Stages training and scoring jobs; never enqueues them.
confirm: lowest final validation BPB wins; the lower LR wins within 0.005 BPB. For an anchor whose winner is an edge
         of the screened grid, the adjacent grid neighbour must have a completed, eligible confirmation (it is staged
         first if missing); if the winner beats it by more than 0.01 BPB, one outward extension is staged (recorded in
         extensions.json, at most one per anchor). Otherwise writes the learning-rate map for make_jobs.py.
         Transfers: P6 -> P8 (S), P8 -> P6 and P6-nohash (M).

A run is eligible when it completed, its training log has only finite losses and gradient norms, and its score is
finite and belongs to the run's final checkpoint. Paused, running or unscored runs make the selection pending
(exit status 3); failed or non-finite runs are excluded and listed.
"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys


ANCHORS = ("S-BPE50K", "S-P6", "M-BPE50K", "M-P8")
CONTROLS = {"S-BPE16K": "S-BPE50K", "M-BPE16K": "M-BPE50K"}
TRANSFERS = {"S-P8": "S-P6", "M-P6": "M-P8", "M-P6-nohash": "M-P8"}
TIE_BPB = 0.005
EXTENSION_BPB = 0.01
CONFIRM_BYTES = {"anchor": 300_000_000, "control": 150_000_000}
SCHEDULE_BYTES = 3_000_000_000
MICRO_BATCH_16K = {"S-BPE16K": 4, "M-BPE16K": 2}  # pilot3, peak reserved <= 10.5 GB
PENDING_EXIT = 3


class Pending(Exception):
    pass


def _arg(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _set(command: list[str], flag: str, value: str) -> None:
    command[command.index(flag) + 1] = value


def _state(run: Path, score_path: Path) -> tuple[str, float | None]:
    """'pending', 'excluded: <reason>' or 'eligible', with the validation BPB when eligible."""
    status_path = run / "status.json"
    if not status_path.exists():
        return "pending", None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status["status"] == "failed":
        return "excluded: run failed", None
    with (run / "log.jsonl").open(encoding="utf-8") as log:
        for line in log:
            entry = json.loads(line)
            if not all(math.isfinite(float(entry[key])) for key in ("loss_nats_per_byte", "grad_norm")):
                return "excluded: non-finite training loss or gradient norm", None
    if status["status"] != "complete" or not score_path.exists():
        return "pending", None
    score = json.loads(score_path.read_text(encoding="utf-8"))
    identity = score.get("checkpoint") or {}
    if Path(identity.get("path", "")).parent.parent.resolve() != run.resolve() or identity.get("step") != status["step"]:
        raise ValueError(f"{score_path} does not belong to the final checkpoint of {run}")
    if not math.isfinite(float(score["bpb"])):
        return "excluded: non-finite validation BPB", None
    return "eligible", float(score["bpb"])


def _results(jobs: Path, scores: Path, runs: Path) -> dict[str, list[dict]]:
    by_model: dict[str, list[dict]] = {}
    for path in sorted(jobs.glob("*.json")):
        job = json.loads(path.read_text(encoding="utf-8"))
        command = job.get("command", [])  # skips scoring jobs and summaries
        if "scripts/train.py" not in command:
            continue
        state, bpb = _state(runs / job["name"], scores / f"{job['name']}.json")
        by_model.setdefault(_arg(command, "--model"), []).append(
            {"name": job["name"], "lr": float(_arg(command, "--learning-rate")), "val_bpb": bpb, "state": state,
             "command": command})
    pending = [row["name"] for rows in by_model.values() for row in rows if row["state"] == "pending"]
    if pending:
        raise Pending("selection pending; not yet complete and scored: " + ", ".join(pending))
    return by_model


def _eligible(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row["state"] == "eligible"]


def _winner(rows: list[dict]) -> dict:
    """Lowest BPB among eligible runs; the lowest LR within TIE_BPB of it wins."""
    eligible = _eligible(rows)
    if not eligible:
        raise ValueError(f"no eligible run among {[row['name'] for row in rows]}")
    best = min(row["val_bpb"] for row in eligible)
    return min((row for row in eligible if row["val_bpb"] <= best + TIE_BPB), key=lambda row: row["lr"])


def _summary(rows: list[dict]) -> list[dict]:
    return [{key: row[key] for key in ("lr", "val_bpb", "state")} for row in sorted(rows, key=lambda row: row["lr"])]


def _job(name: str, command: list[str], priority: int, created: str) -> dict:
    return {"name": name, "priority": priority, "created": created, "command": command}


def _score_job(train: dict, scores: Path, runs: Path, created: str) -> dict:
    command = train["command"]
    score = ["python", "scripts/score.py", "--run", str(runs / train["name"]), "--supply", _arg(command, "--supply"),
             "--split", "val", "--output", str(scores / f"{train['name']}.json"), "--device", "cuda"]
    if "--tokenizer" in command:
        score += ["--tokenizer", _arg(command, "--tokenizer")]
    return _job(f"score-{train['name']}", score, train["priority"] - 50, created)


def _confirm(template: list[str], model: str, lr: float, args: argparse.Namespace, priority: int, created: str) -> dict:
    command = list(template)
    name = f"confirm-{model.lower()}-{lr:.1e}"
    _set(command, "--model", model)
    _set(command, "--run-name", name)
    _set(command, "--learning-rate", repr(lr))
    _set(command, "--stop-bytes", str(CONFIRM_BYTES["control" if model in CONTROLS else "anchor"]))
    _set(command, "--schedule-bytes", str(SCHEDULE_BYTES))
    if model in MICRO_BATCH_16K:
        _set(command, "--micro-batch-size", str(MICRO_BATCH_16K[model]))
        _set(command, "--tokenizer", str(args.tokenizer_16k))
    return _job(name, command, priority, created)


def stage_confirmation(args: argparse.Namespace) -> None:
    results = _results(args.screen_jobs, args.screen_scores, args.runs)
    created = datetime.now(timezone.utc).isoformat()
    summary: dict[str, dict] = {}
    jobs: list[dict] = []
    priority = 900
    for anchor in ANCHORS:
        ranked = sorted(_eligible(results[anchor]), key=lambda row: (row["val_bpb"], row["lr"]))
        if len(ranked) < 2:
            raise ValueError(f"{anchor}: fewer than two eligible screened learning rates: {_summary(results[anchor])}")
        summary[anchor] = {"screen": _summary(results[anchor]), "confirm_lrs": [row["lr"] for row in ranked[:2]]}
        for row in ranked[:2]:
            jobs.append(_confirm(row["command"], anchor, row["lr"], args, priority, created))
            priority -= 1
    for control, source in CONTROLS.items():
        rows = sorted(results[source], key=lambda row: row["lr"])
        transferred = _winner(rows)
        index = rows.index(transferred)
        neighbours = _eligible([rows[i] for i in (index - 1, index + 1) if 0 <= i < len(rows)])
        if not neighbours:
            raise ValueError(f"{control}: the transferred LR {transferred['lr']} has no eligible screened neighbour")
        candidates = [transferred, min(neighbours, key=lambda row: row["val_bpb"])]
        summary[control] = {"transferred_from": source, "confirm_lrs": [row["lr"] for row in candidates]}
        for row in candidates:
            jobs.append(_confirm(row["command"], control, row["lr"], args, priority, created))
            priority -= 1
    jobs += [_score_job(job, args.confirm_scores, args.runs, created) for job in list(jobs)]
    _write(args.output_dir, jobs)
    (args.output_dir / "screen_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def select(args: argparse.Namespace) -> None:
    screened = _results(args.screen_jobs, args.screen_scores, args.runs)
    confirmed = _results(args.confirm_jobs, args.confirm_scores, args.runs)
    created = datetime.now(timezone.utc).isoformat()
    extensions_path = args.confirm_jobs / "extensions.json"
    extensions = json.loads(extensions_path.read_text(encoding="utf-8")) if extensions_path.exists() else {}
    rates: dict[str, float] = {}
    staged: list[dict] = []
    record: dict[str, dict] = {}
    for model in (*ANCHORS, *CONTROLS):
        rows = confirmed[model]
        winner = _winner(rows)
        record[model] = {"confirm": _summary(rows), "selected": winner["lr"], "extension": extensions.get(model)}
        rates[model] = winner["lr"]
        grid = sorted(row["lr"] for row in screened.get(model, []))
        if model not in ANCHORS or model in extensions or winner["lr"] not in (grid[0], grid[-1]):
            continue
        neighbour_lr = grid[1] if winner["lr"] == grid[0] else grid[-2]
        neighbour = next((row for row in rows if row["lr"] == neighbour_lr), None)
        if neighbour is None:
            staged.append(_confirm(winner["command"], model, neighbour_lr, args, 900, created))
            record[model]["staged"] = f"adjacent neighbour {neighbour_lr}"
        elif neighbour["state"] == "eligible" and neighbour["val_bpb"] - winner["val_bpb"] > EXTENSION_BPB:
            factor = grid[1] / grid[0]
            outward = winner["lr"] * factor if winner["lr"] == grid[-1] else winner["lr"] / factor
            staged.append(_confirm(winner["command"], model, outward, args, 900, created))
            extensions[model] = outward
            record[model]["staged"] = f"outward extension {outward}"
    if staged:
        _write(args.output_dir, staged + [_score_job(job, args.confirm_scores, args.runs, created) for job in staged])
        extensions_path.write_text(json.dumps(extensions, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(record, indent=2))
        print(f"Staged {len(staged)} further confirmations; rerun `confirm` after they are scored.")
        return
    for model, source in TRANSFERS.items():
        rates[model] = rates[source]
    args.output.write_text(json.dumps({"learning_rates": rates, "record": record, "rules": __doc__}, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(rates, indent=2))


def _write(directory: Path, jobs: list[dict]) -> None:
    if directory.resolve().name.lower() == "pending":
        raise ValueError("output must be a staging directory, not a live pending queue")
    directory.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        with (directory / f"{job['name']}.json").open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(job, handle, indent=1)
            handle.write("\n")
    print(f"Staged {len(jobs)} jobs in {directory}; no jobs were started.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    home = Path.home() / "blt"
    parser.add_argument("stage", choices=("screen", "confirm"))
    parser.add_argument("--screen-jobs", type=Path, default=home / "jobs/staging/lr-pilot")
    parser.add_argument("--screen-scores", type=Path, default=home / "results/lr-screen")
    parser.add_argument("--confirm-jobs", type=Path, default=home / "jobs/staging/lr-confirm")
    parser.add_argument("--confirm-scores", type=Path, default=home / "results/lr-confirm")
    parser.add_argument("--runs", type=Path, default=home / "runs")
    parser.add_argument("--tokenizer-16k", type=Path, default=home / "data/tokenizer16k-v1/tokenizer.json")
    parser.add_argument("--output-dir", type=Path, help="staging directory for new jobs (default: --confirm-jobs)")
    parser.add_argument("--output", type=Path, default=home / "results/learning_rates.json")
    args = parser.parse_args()
    # Further confirmations are staged beside the first ones so that a rerun of `confirm` includes them.
    args.output_dir = args.output_dir or args.confirm_jobs
    try:
        stage_confirmation(args) if args.stage == "screen" else select(args)
    except Pending as pending:
        print(pending, file=sys.stderr)
        sys.exit(PENDING_EXIT)


if __name__ == "__main__":
    main()
