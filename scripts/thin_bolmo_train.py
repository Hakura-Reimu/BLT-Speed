from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import signal
import threading

from blt.atomic import atomic_json
from blt.data import WindowIndex, iter_eval_chunks, sha256_file, verify_supply
from blt.thin_bolmo import DistillConfig, DistillTrainer, build_student
from blt.trackA import load_reference
from blt.train import seed_everything


def _run(stop: threading.Event) -> None:
    parser = argparse.ArgumentParser(description="Distill a selected Thin-Bolmo student from its online teacher")
    parser.add_argument("--bolmo", type=Path, required=True, help="local unmodified Bolmo checkpoint")
    parser.add_argument("--bolmo-subword-tokenizer", type=Path)
    parser.add_argument("--selection", type=Path, required=True, help="researcher-authored selection JSON")
    parser.add_argument("--supply", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--window-bytes", type=int, choices=(2048, 4096), default=2048)
    parser.add_argument("--windows-per-step", type=int, default=1, help="one window at a time; gradients accumulate")
    parser.add_argument("--stop-bytes", type=int, default=1_500_000_000)
    parser.add_argument("--schedule-bytes", type=int, default=1_500_000_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--val-interval-steps", type=int, default=500)
    parser.add_argument("--val-windows", type=int, default=16)
    parser.add_argument("--mse-weight", type=float, default=0.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--boundary-weight", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, choices=(0, 32), default=0)
    parser.add_argument("--unfreeze-suffix", action="store_true")
    parser.add_argument("--unfreeze-head", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--checkpoint-seconds", type=float, default=900)
    parser.add_argument("--pause-file", type=Path, default=Path("control/PAUSE"))
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true")
    resume.add_argument("--resume-if-exists", action="store_true")
    args = parser.parse_args()
    if min(args.stop_bytes, args.schedule_bytes, args.windows_per_step, args.val_windows) <= 0:
        parser.error("byte targets and window counts must be positive")
    done = threading.Event()

    def watch_pause() -> None:
        while not done.wait(2.0):
            if args.pause_file.exists():
                stop.set()

    watcher = threading.Thread(target=watch_pause, daemon=True)
    watcher.start()
    try:
        if args.pause_file.exists():
            stop.set()
        verify_supply(args.supply, stop)
        if stop.is_set():
            raise InterruptedError("paused during startup")
        has_checkpoint = (args.run_dir / "ckpt/latest.json").exists()
        if has_checkpoint and not (args.resume or args.resume_if_exists):
            parser.error("run already has a checkpoint; pass --resume")
        if args.resume and not has_checkpoint:
            parser.error("--resume requires an existing checkpoint")
        if args.resume_if_exists and not has_checkpoint and any((args.run_dir / name).exists()
                                                                 for name in ("log.jsonl", "status.json")):
            parser.error("run has prior activity but its checkpoint is missing")
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        manifest_hash = sha256_file(args.supply / "manifest.json", stop)
        val = list(itertools.islice(iter_eval_chunks(args.supply, "val", args.window_bytes), args.val_windows))
        bundle = load_reference("bolmo", args.bolmo,
                                bolmo_subword_tokenizer=str(args.bolmo_subword_tokenizer) if args.bolmo_subword_tokenizer else None)
        seed_everything(args.init_seed)
        student = build_student(bundle.model, selection)
        schedule_steps = math.ceil(args.schedule_bytes / (args.window_bytes * args.windows_per_step))
        cfg = DistillConfig(window_bytes=args.window_bytes, windows_per_step=args.windows_per_step,
                            init_seed=args.init_seed, data_seed=args.data_seed, learning_rate=args.learning_rate,
                            warmup_steps=args.warmup_steps, schedule_steps=schedule_steps,
                            val_interval_steps=args.val_interval_steps, mse_weight=args.mse_weight,
                            ce_weight=args.ce_weight, boundary_weight=args.boundary_weight, lora_rank=args.lora_rank,
                            unfreeze_suffix=args.unfreeze_suffix, unfreeze_head=args.unfreeze_head,
                            gradient_checkpointing=not args.no_gradient_checkpointing)
        with WindowIndex(args.supply / "train.bin", args.window_bytes) as windows:
            trainer = DistillTrainer(bundle.model, student, selection, bundle.identity, windows, manifest_hash,
                                     val, args.run_dir, cfg)
            if args.resume or (args.resume_if_exists and has_checkpoint):
                trainer.load()
            trainer.run(args.stop_bytes, args.pause_file, checkpoint_seconds=args.checkpoint_seconds, stop_event=stop)
    finally:
        done.set()
        watcher.join()


def main() -> None:
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stop.set())
    try:
        _run(stop)
    except InterruptedError:
        outcome = os.environ.get("BLT_QUEUE_OUTCOME_FILE")
        if outcome:
            atomic_json(Path(outcome), {"status": "paused"})
        print("Startup interrupted before training; the queued job can start again safely.", flush=True)


if __name__ == "__main__":
    main()
