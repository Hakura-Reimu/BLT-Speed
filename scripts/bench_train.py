from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import tempfile
import time

from tokenizers import Tokenizer
import torch
import torch._dynamo

from blt.atomic import atomic_json
from blt.config import BPEConfig, ByteConfig, preset
from blt.data import WindowIndex, sample_window_index, sha256_file, verify_supply
from blt.patcher import calibrate_min_len
from blt.token_cache import TokenCache
from blt.train import Trainer, TrainingConfig, make_model, seed_everything


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bucket_histogram(metrics: list[dict]) -> list[dict]:
    counts: dict[tuple[int, ...], int] = {}
    for step in metrics:
        for shape in step["padding_bucket_shapes"]:
            key = tuple(shape)
            counts[key] = counts.get(key, 0) + 1
    return [{"shape": list(shape), "microbatches": count} for shape, count in sorted(counts.items())]


def measure_training(trainer: Trainer, device: torch.device, *, steps: int, warmup_steps: int,
                     max_attempts: int) -> dict:
    """Time whole updates, excluding each update that caused a new compiled graph."""
    if steps <= 0 or warmup_steps < 0 or max_attempts < steps:
        raise ValueError("invalid warm-up or measurement counts")
    warmup = [trainer.train_step() for _ in range(warmup_steps)]
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    attempted: list[dict] = []
    steady: list[dict] = []
    for _ in range(max_attempts):
        before_bytes = trainer.consumed_bytes
        _synchronize(device)
        start = time.perf_counter()
        metrics = trainer.train_step()
        _synchronize(device)
        row = {"metrics": metrics, "seconds": time.perf_counter() - start,
               "bytes": trainer.consumed_bytes - before_bytes}
        attempted.append(row)
        if metrics.get("compile_new_graphs", 0) == 0:
            steady.append(row)
            if len(steady) == steps:
                break
    attempts = [row["metrics"] for row in attempted]
    selected = [row["metrics"] for row in steady]
    result = {"status": "ok" if len(steady) == steps else "insufficient_steady_steps",
              "warmup_steps": warmup_steps, "requested_steady_steps": steps,
              "measurement_attempts": len(attempted), "steady_steps": len(steady),
              "excluded_compile_steps": len(attempted) - len(steady),
              "excluded_compile_seconds": sum(row["seconds"] for row in attempted if row["metrics"].get("compile_new_graphs", 0)),
              "warmup_compile_new_graphs": sum(row.get("compile_new_graphs", 0) for row in warmup),
              "warmup_compile_recompiles": sum(row.get("compile_recompiles", 0) for row in warmup),
              "compile_new_graphs": sum(row.get("compile_new_graphs", 0) for row in attempts),
              "compile_recompiles": sum(row.get("compile_recompiles", 0) for row in attempts),
              "warmup_bucket_histogram": _bucket_histogram(warmup),
              "bucket_histogram": _bucket_histogram(warmup + attempts),
              "steady_bucket_histogram": _bucket_histogram(selected),
              "distinct_padding_buckets": len(_bucket_histogram(warmup + attempts)),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}
    if steady:
        elapsed = sum(row["seconds"] for row in steady)
        consumed = sum(row["bytes"] for row in steady)
        real = sum(row["padding_real_positions"] for row in selected)
        padded = sum(row["padding_padded_positions"] for row in selected)
        units = sum(row["units_with_tail_per_window"] * trainer.config.windows_per_step for row in selected)
        result.update(bytes_per_second=consumed / elapsed, step_seconds=elapsed / len(steady),
                      data_wait_fraction=sum(row["data_wait_seconds"] for row in selected) / elapsed,
                      padding_real_positions=real, padding_padded_positions=padded,
                      padding_efficiency=real / padded if padded else 1.0,
                      mean_unit_bytes=consumed / units if units else None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure training throughput on the immutable supply")
    parser.add_argument("--model", required=True)
    parser.add_argument("--supply", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-measure-attempts", type=int, help="cap attempts to collect compile-free steps (default: 3 * steps)")
    parser.add_argument("--micro-batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--global-batch-bytes", type=int, default=128 * 1024)
    parser.add_argument("--window-bytes", type=int, default=8192)
    parser.add_argument("--calibration-windows", type=int, default=128)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--hash-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    max_attempts = args.max_measure_attempts if args.max_measure_attempts is not None else 3 * args.steps
    if (args.steps <= 0 or args.warmup_steps < 0 or max_attempts < args.steps or args.window_bytes <= 0 or args.calibration_windows <= 0 or
            not 64 * 1024 <= args.global_batch_bytes <= 128 * 1024 or
            args.global_batch_bytes % args.window_bytes or any(size <= 0 for size in args.micro_batches)):
        parser.error("invalid step count, calibration, window, or batch sizes")
    verify_supply(args.supply)
    manifest_hash = sha256_file(args.supply / "manifest.json")
    tokenizer = Tokenizer.from_file(str(args.tokenizer)) if args.tokenizer else None
    tokenizer_hash = sha256_file(args.tokenizer) if args.tokenizer else None
    initial = preset(args.model, hash_seed=args.hash_seed)
    device = torch.device(args.device)
    windows_per_step = args.global_batch_bytes // args.window_bytes
    output: dict = {"model": args.model, "device": str(device), "warmup_steps": args.warmup_steps, "measured_steps": args.steps,
                    "window_bytes": args.window_bytes, "max_measure_attempts": max_attempts,
                    "global_batch_bytes": args.global_batch_bytes, "manifest_sha256": manifest_hash,
                    "tokenizer_sha256": tokenizer_hash, "results": []}
    had_error = False
    with WindowIndex(args.supply / "train.bin", args.window_bytes) as windows:
        if isinstance(initial, BPEConfig):
            if tokenizer is None:
                parser.error("BPE models need --tokenizer")
            bos = tokenizer.token_to_id("<|endoftext|>") if "50K" in args.model else tokenizer.token_to_id("<|bos|>")
            if bos is None:
                parser.error("tokenizer is missing its expected BOS token")
            if "16K" in args.model:
                metadata = json.loads((args.tokenizer.parent / "metadata.json").read_text(encoding="utf-8"))
                if (metadata.get("tokenizer_sha256") != tokenizer_hash or metadata.get("train_manifest_sha256") != manifest_hash
                        or metadata.get("vocab_size") != tokenizer.get_vocab_size()):
                    raise ValueError("16K tokenizer metadata does not match this tokenizer and supply")
            model_config = preset(args.model, vocab_size=tokenizer.get_vocab_size(), bos_id=bos)
            cache = TokenCache(args.supply / "tokenized" / tokenizer_hash, windows, manifest_hash,
                               tokenizer_hash, model_config.vocab_size)
        else:
            if tokenizer is not None:
                parser.error("byte models do not use a tokenizer")
            sample_count = min(args.calibration_windows, len(windows))
            samples = [windows[sample_window_index(args.calibration_seed, i, 0, 1, len(windows))].data
                       for i in range(sample_count)]
            calibration = calibrate_min_len(samples, initial.target_patch_mean, initial.max_len, initial.block_size)
            model_config = preset(args.model, min_len=calibration.min_len, hash_seed=args.hash_seed)
            output["calibration"] = asdict(calibration)
            cache = None
        output["model_config"] = asdict(model_config)
        for compiled in (False, True):
            for micro in args.micro_batches:
                record = {"compile": compiled, "micro_batch_size": micro}
                output["results"].append(record)
                if micro > windows_per_step:
                    record["status"] = "exceeds_global_batch"
                    continue
                trainer = None
                model = None
                try:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if compiled:
                        torch._dynamo.reset()
                    seed_everything(args.init_seed)
                    config = TrainingConfig(model_config, args.init_seed, args.data_seed, windows_per_step, micro,
                                            args.window_bytes, 6e-4 if args.model.startswith("S-") else 3e-4,
                                            max(100, args.warmup_steps + max_attempts + 1), compile_model=compiled)
                    with tempfile.TemporaryDirectory(prefix="blt-bench-") as directory:
                        model = make_model(model_config)
                        trainer = Trainer(config, model, windows, directory, device, manifest_hash, tokenizer_hash, tokenizer, cache)
                        trainer.start_prefetch()
                        measurement = measure_training(trainer, device, steps=args.steps, warmup_steps=args.warmup_steps,
                                                       max_attempts=max_attempts)
                        unit_bytes = measurement.pop("mean_unit_bytes", None)
                        record.update(measurement)
                        record["mean_patch_bytes" if isinstance(model_config, ByteConfig) else "bytes_per_token"] = unit_bytes
                        had_error |= record["status"] != "ok"
                except Exception as error:
                    is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
                    record.update(status="oom" if is_oom else "error", error=str(error))
                    had_error |= not is_oom
                    break
                finally:
                    if trainer is not None:
                        trainer.stop_prefetch()
                    del trainer, model
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    atomic_json(args.output, output)
    print(json.dumps(output, indent=2, sort_keys=True))
    if had_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
