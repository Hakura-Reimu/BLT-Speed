from __future__ import annotations

"""Exploratory smaller BPE control for the quality-speed frontier (added after review round 13; exploratory, not registered).

XS-BPE50K and XS-BPE16K (4 layers, width 256, 4/4 heads, FFN 768, tied embeddings) get the same learning-rate
procedure as the registered models, using scripts/select_lr.py's rules unchanged:
  screen:  stage three 150 MB XS-BPE50K screening runs (cosine to 150 MB) and their full-validation scores.
  select:  run select_lr.py's `screen` or `confirm` stage with XS-BPE50K as the only anchor and XS-BPE16K as its
           16K control (confirmations, possible edge extension, then results/learning_rates_xs.json).
  main:    stage the two 3 GB seed-0 runs, copied from the S-BPE main commands (same data, seeds, schedule, hashes).
  enqueue: copy staged jobs that are not yet queued into the live queue below the registered work, scores with an
           unchanged resume_command (score.py rejects --resume-if-exists).
Staging never starts anything; only `enqueue` writes to the queue.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys


HOME = Path.home() / "blt"
GRID = (6e-4, 1.2e-3, 2.4e-3)  # the S grid (3e-4..1.2e-3) moved up one step for half the width
MICRO_BATCH = 4  # S-BPE pilots were flat in speed across micro-batches 2-8; XS needs less memory than S
DIRS = {"screen_jobs": HOME / "jobs/staging/lr-pilot-xs", "screen_scores": HOME / "results/lr-screen-xs",
        "confirm_jobs": HOME / "jobs/staging/lr-confirm-xs", "confirm_scores": HOME / "results/lr-confirm-xs",
        "main_jobs": HOME / "jobs/staging/xs-bpe", "rates": HOME / "results/learning_rates_xs.json"}
TRAIN_PRIORITY, SCORE_PRIORITY = 74.0, 73.0  # below every registered main (M-P6-nohash is 75)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file(job: dict) -> str:
    # Main runs keep make_jobs.py's <name>-main.json file name, which queue_milestone_scores.sh looks up.
    return f"{job['name']}-{job['phase']}.json" if job.get("phase") else f"{job['name']}.json"


def _write(directory: Path, jobs: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        with (directory / _file(job)).open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(job, handle, indent=1)
            handle.write("\n")
    print(f"Staged {len(jobs)} jobs in {directory}; no jobs were started.")


def _score(name: str, run_root: Path, output: Path, tokenizer: str, created: str) -> dict:
    command = ["python", "scripts/score.py", "--run", str(run_root / name), "--supply", str(HOME / "data/supply-v1"),
               "--split", "val", "--output", str(output), "--device", "cuda", "--tokenizer", tokenizer]
    return {"name": f"score-{name}", "priority": SCORE_PRIORITY, "created": created, "command": command}


def stage_screen(args: argparse.Namespace) -> None:
    created, jobs = _now(), []
    tokenizer = str(HOME / "models/gpt2-tokenizer/tokenizer.json")
    for lr in GRID:
        name = f"lr-xs-bpe50k-{lr:.1e}"
        command = ["python", "scripts/train.py", "--model", "XS-BPE50K", "--run-name", name,
                   "--supply", str(HOME / "data/supply-v1"), "--global-batch-bytes", "131072",
                   "--micro-batch-size", str(MICRO_BATCH), "--stop-bytes", "150000000", "--schedule-bytes", "150000000",
                   "--learning-rate", repr(lr), "--val-interval-steps", "250", "--val-windows", "256", "--compile",
                   "--tokenizer", tokenizer]
        jobs.append({"name": name, "priority": TRAIN_PRIORITY, "created": created, "command": command})
        jobs.append(_score(name, HOME / "runs", DIRS["screen_scores"] / f"{name}.json", tokenizer, created))
    _write(DIRS["screen_jobs"], jobs)


def select(args: argparse.Namespace) -> None:
    path = Path(__file__).resolve().parents[1] / "select_lr.py"
    spec = importlib.util.spec_from_file_location("select_lr", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The rules are select_lr.py's; only the model set and the directories change.
    module.ANCHORS = ("XS-BPE50K",)
    module.CONTROLS = {"XS-BPE16K": "XS-BPE50K"}
    module.TRANSFERS = {}
    module.MICRO_BATCH_16K = {"XS-BPE16K": MICRO_BATCH}
    sys.argv = [str(path), args.stage, "--screen-jobs", str(DIRS["screen_jobs"]),
                "--screen-scores", str(DIRS["screen_scores"]), "--confirm-jobs", str(DIRS["confirm_jobs"]),
                "--confirm-scores", str(DIRS["confirm_scores"]), "--output", str(DIRS["rates"]),
                "--runs", str(HOME / "runs"), "--tokenizer-16k", str(HOME / "data/tokenizer16k-v1/tokenizer.json")]
    module.main()


def stage_main(args: argparse.Namespace) -> None:
    rates = json.loads(DIRS["rates"].read_text(encoding="utf-8"))["learning_rates"]
    created, jobs = _now(), []
    for model, source in (("XS-BPE50K", "s-bpe50k-seed0"), ("XS-BPE16K", "s-bpe16k-seed0")):
        command = list(json.loads((HOME / f"queue/done/{source}-main.json").read_text(encoding="utf-8"))["command"])
        name = f"{model.lower()}-seed0"
        for flag, value in (("--model", model), ("--run-name", name), ("--micro-batch-size", str(MICRO_BATCH)),
                            ("--learning-rate", repr(rates[model]))):
            command[command.index(flag) + 1] = value
        jobs.append({"name": name, "priority": TRAIN_PRIORITY, "created": created, "command": command,
                     "phase": "main", "learning_rate_source": str(DIRS["rates"])})
    _write(DIRS["main_jobs"], jobs)


def enqueue(args: argparse.Namespace) -> None:
    queue = HOME / "queue"
    for path in sorted(args.staged.glob("*.json")):
        job = json.loads(path.read_text(encoding="utf-8"))
        if "command" not in job:
            continue  # summaries such as screen_summary.json
        target = queue / "pending" / _file(job)
        if any((queue / state / target.name).exists() for state in ("pending", "running", "done", "failed")):
            print("skip", job["name"])
            continue
        is_score = "scripts/score.py" in job["command"]
        job["priority"] = SCORE_PRIORITY if is_score else TRAIN_PRIORITY
        if is_score:
            job["resume_command"] = job["command"]
        tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(job, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, target)
        print("queued", job["name"], job["priority"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("screen").set_defaults(run=stage_screen)
    choose = sub.add_parser("select")
    choose.add_argument("stage", choices=("screen", "confirm"))
    choose.set_defaults(run=select)
    sub.add_parser("main").set_defaults(run=stage_main)
    put = sub.add_parser("enqueue")
    put.add_argument("staged", type=Path)
    put.set_defaults(run=enqueue)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
