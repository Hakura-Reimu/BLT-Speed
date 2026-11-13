from __future__ import annotations

"""Prepare reviewable Track B queue files; never enqueue or launch them."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any


BASE_MODELS = (
    "S-BPE50K", "S-P6", "S-P8", "S-BPE16K",
    "M-BPE50K", "M-P6", "M-P8", "M-BPE16K", "M-P6-nohash",
)
MEMORY_LIMIT_BYTES = 10_500_000_000  # 10.5 GB, not 10.5 GiB
ENDPOINT = 3_000_000_000
PROGRESS_CONTINUATION = 750_000_000
WINDOW_BYTES = 8192


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def select_pilots(pilot_dir: Path, manifest_hash: str, tokenizer_hashes: dict[str, str], *,
                  confirm_legacy_8k: bool = False) -> dict[str, dict]:
    """Pick the fastest eligible compiled measurement, preserving its provenance."""
    selected: dict[str, dict] = {}
    for path in sorted(pilot_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            pilot = json.load(handle)
        model = pilot.get("model")
        if model not in BASE_MODELS or pilot.get("manifest_sha256") != manifest_hash:
            continue
        expected_tokenizer = tokenizer_hashes.get("50K" if "50K" in model else "16K") if "BPE" in model else None
        if pilot.get("tokenizer_sha256") != expected_tokenizer or not str(pilot.get("device", "")).startswith("cuda"):
            continue
        # The current bench_train JSON omits this field even when its CLI was overridden.
        if (("window_bytes" not in pilot and not confirm_legacy_8k)
                or pilot.get("window_bytes", WINDOW_BYTES) != WINDOW_BYTES):
            continue
        batch = pilot.get("global_batch_bytes")
        if (not isinstance(batch, int) or isinstance(batch, bool) or not 65536 <= batch <= 131072
                or batch % WINDOW_BYTES):
            continue
        for record in pilot.get("results", []):
            micro = record.get("micro_batch_size")
            speed, memory = record.get("bytes_per_second"), record.get("peak_reserved_bytes")
            if (record.get("status") != "ok" or record.get("compile") is not True
                    or not isinstance(micro, int) or isinstance(micro, bool) or not 1 <= micro <= batch // WINDOW_BYTES
                    or not _finite_positive(speed) or not _finite_positive(memory) or memory > MEMORY_LIMIT_BYTES):
                continue
            candidate = {"global_batch_bytes": batch, "micro_batch_size": micro, "bytes_per_second": speed,
                         "peak_reserved_bytes": memory, "pilot": str(path),
                         "manifest_sha256": manifest_hash, "tokenizer_sha256": expected_tokenizer,
                         "pilot_window_bytes_recorded": "window_bytes" in pilot}
            if model not in selected or speed > selected[model]["bytes_per_second"]:
                selected[model] = candidate
    missing = [model for model in BASE_MODELS if model not in selected]
    if missing:
        raise ValueError("no eligible compiled CUDA 8 KiB pilot at or below 10.5 GB for: " + ", ".join(missing)
                         + "; for legacy JSONs, --confirm-legacy-pilot-8k attests the benchmark window size")
    return selected


def _command(args: argparse.Namespace, model: str, name: str, seed: int, pilot: dict) -> list[str]:
    command = [args.python, "scripts/train.py", "--model", model, "--run-name", name,
               "--run-root", str(args.run_root), "--supply", str(args.supply),
               "--global-batch-bytes", str(pilot["global_batch_bytes"]),
               "--micro-batch-size", str(pilot["micro_batch_size"]), "--window-bytes", str(WINDOW_BYTES),
               "--expected-manifest-sha256", pilot["manifest_sha256"],
               "--stop-bytes", str(ENDPOINT), "--schedule-bytes", str(ENDPOINT),
               "--init-seed", str(seed), "--data-seed", "0", "--hash-seed", "0", "--compile",
               "--learning-rate", repr(args.learning_rates[model])]
    if "BPE" in model:
        command += ["--tokenizer", str(args.tokenizer_50k if "50K" in model else args.tokenizer_16k),
                    "--expected-tokenizer-sha256", pilot["tokenizer_sha256"]]
    return command


def _replace_arg(command: list[str], flag: str, value: int) -> None:
    command[command.index(flag) + 1] = str(value)


def make_jobs(args: argparse.Namespace) -> list[dict]:
    if args.output_dir.resolve().name.lower() == "pending":
        raise ValueError("output must be a staging directory, not a live pending queue")
    if args.include_repeats or args.include_continuations or args.include_long_context:
        if args.selected_bpe not in BASE_MODELS or "BPE" not in args.selected_bpe:
            raise ValueError("a validation-selected --selected-bpe is required")
        if args.selected_byte not in BASE_MODELS or "BPE" in args.selected_byte or args.selected_byte.endswith("-nohash"):
            raise ValueError("a validation-selected --selected-byte is required")
    if args.include_long_context and (args.long_context_bytes is None or args.long_context_bytes <= 0):
        raise ValueError("--long-context-bytes must give the same positive target for both models")
    if (not isinstance(args.learning_rates, dict) or set(args.learning_rates) != set(BASE_MODELS)
            or not all(_finite_positive(value) for value in args.learning_rates.values())):
        raise ValueError("--learning-rates must give one positive peak learning rate for every base model")
    manifest_hash = _sha256(args.supply / "manifest.json")
    tokenizer_hashes = {"50K": _sha256(args.tokenizer_50k), "16K": _sha256(args.tokenizer_16k)}
    pilots = select_pilots(args.pilot_dir, manifest_hash, tokenizer_hashes, confirm_legacy_8k=args.confirm_legacy_pilot_8k)
    created = datetime.now(timezone.utc).isoformat()
    jobs: list[dict] = []

    def add(model: str, seed: int, phase: str, priority: int, *, long_context: bool = False) -> None:
        stem = f"{model.lower()}-seed{seed}"
        name = stem if phase == "main" else f"{stem}-{phase}"
        command = _command(args, model, name, seed, pilots[model])
        if phase != "main" and phase != "repeat":
            command += ["--init-from", str(args.run_root / stem / "ckpt" / f"bytes-{ENDPOINT}.pt")]
            _replace_arg(command, "--stop-bytes", args.long_context_bytes if long_context else PROGRESS_CONTINUATION)
            if long_context:
                command += ["--long-context"]
        if phase == "repeat":
            # Repeats are the source runs from which the seed-1 continuations start.
            name = stem
            command[command.index("--run-name") + 1] = name
        jobs.append({"name": name, "priority": priority, "created": created, "command": command,
                     "phase": phase, "pilot": pilots[model]})

    # The schedule puts the no-hash ablation after repeats and before continuation.
    for rank, model in enumerate(BASE_MODELS):
        add(model, 0, "main", 100 - rank if model != "M-P6-nohash" else 75)
    pair = (args.selected_bpe, args.selected_byte)
    if args.include_repeats:
        for rank, model in enumerate(pair):
            add(model, 1, "repeat", 80 - rank)
    if args.include_continuations:
        for rank, (model, seed) in enumerate((model, seed) for model in pair for seed in (0, 1)):
            add(model, seed, "progress-continuation", 70 - rank)
    if args.include_long_context:
        for rank, model in enumerate(pair):
            add(model, 0, "long-context", 60 - rank, long_context=True)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--supply", type=Path, required=True)
    parser.add_argument("--tokenizer-50k", type=Path, required=True)
    parser.add_argument("--tokenizer-16k", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("jobs/staging/track-b"))
    parser.add_argument("--run-root", type=Path, default=Path.home() / "blt/runs")
    parser.add_argument("--python", default="python")
    parser.add_argument("--confirm-legacy-pilot-8k", action="store_true",
                        help="attest that pilot JSONs without window metadata were measured at 8192 bytes")
    parser.add_argument("--selected-bpe", choices=[m for m in BASE_MODELS if "BPE" in m])
    parser.add_argument("--selected-byte", choices=[m for m in BASE_MODELS if "BPE" not in m and not m.endswith("-nohash")])
    parser.add_argument("--include-repeats", action="store_true")
    parser.add_argument("--include-continuations", action="store_true")
    parser.add_argument("--include-long-context", action="store_true")
    parser.add_argument("--long-context-bytes", type=int)
    parser.add_argument("--learning-rates", type=Path, required=True,
                        help="JSON {model: peak learning rate} chosen by scripts/select_lr.py")
    args = parser.parse_args()
    try:
        with args.learning_rates.open(encoding="utf-8") as handle:
            args.learning_rates = json.load(handle).get("learning_rates")
        jobs = make_jobs(args)
        paths = [args.output_dir / f"{job['name']}-{job['phase']}.json" for job in jobs]
        if len(set(paths)) != len(paths) or any(path.exists() for path in paths):
            raise FileExistsError("one or more staged job files already exist")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path, job in zip(paths, jobs):
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(job, handle, indent=2, sort_keys=True)
                handle.write("\n")
        print(f"Staged {len(jobs)} jobs in {args.output_dir}; no jobs were started.")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
