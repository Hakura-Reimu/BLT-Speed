from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import threading

from tokenizers import Tokenizer
import torch

from blt.atomic import atomic_json
from blt.config import BPEConfig, preset
from blt.data import WindowIndex, sample_window_index, sha256_file, verify_supply
from blt.patcher import calibrate_min_len
from blt.token_cache import TokenCache, build_token_cache
from blt.train import LongContextTrainer, Trainer, TrainingConfig, make_model, seed_everything


LONG_CONTEXT_LENGTHS = (8192, 16384, 32768)


def prepare_long_context_caches(supply: Path, tokenizer_path: Path, tokenizer: Tokenizer) -> list[Path]:
    """Build separately named caches using the existing, immutable cache builder.

    The builder writes to a fixed path, so build in a same-filesystem shadow of the
    verified supply and atomically publish each completed cache under its length.
    """
    manifest = verify_supply(supply)
    manifest_hash, tokenizer_hash = sha256_file(supply / "manifest.json"), sha256_file(tokenizer_path)
    token_root = supply / "tokenized"
    token_root.mkdir(parents=True, exist_ok=True)
    cache_root = token_root / tokenizer_hash
    cache_root.mkdir(exist_ok=True)
    result = []
    for size in LONG_CONTEXT_LENGTHS:
        destination = cache_root / f"window-{size}"
        legacy = cache_root if size == LONG_CONTEXT_LENGTHS[0] else destination
        existing = destination if (destination / "meta.json").is_file() else legacy
        if (existing / "meta.json").is_file():
            with WindowIndex(supply / "train.bin", size) as windows:
                TokenCache(existing, windows, manifest_hash, tokenizer_hash, tokenizer.get_vocab_size())
            result.append(existing)
            continue
        if destination.exists():
            raise FileExistsError(f"incomplete cache already exists: {destination}")
        with tempfile.TemporaryDirectory(prefix=".longctx-cache-", dir=token_root) as temporary:
            shadow = Path(temporary) / "supply"
            shadow.mkdir()
            for name in ("manifest.json", *manifest["files"]):
                (shadow / name).symlink_to((supply / name).resolve())
            built = build_token_cache(shadow, tokenizer_path, tokenizer, window_bytes=size)
            os.rename(built, destination)
        result.append(destination)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or resume a Track B run")
    parser.add_argument("--model")
    parser.add_argument("--run-name")
    parser.add_argument("--run-root", type=Path, default=Path.home() / "blt/runs")
    parser.add_argument("--supply", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, help="local tokenizer.json for BPE")
    parser.add_argument("--expected-manifest-sha256", help="require the supply used for the pilot")
    parser.add_argument("--expected-tokenizer-sha256", help="require the tokenizer used for the pilot")
    parser.add_argument("--global-batch-bytes", type=int, default=64 * 1024)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--window-bytes", type=int, default=8192)
    parser.add_argument("--stop-bytes", type=int, default=3_000_000_000)
    parser.add_argument("--schedule-bytes", type=int, default=3_000_000_000)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--hash-seed", type=int, default=0)
    parser.add_argument("--calibration-seed", type=int, default=0)
    parser.add_argument("--calibration-windows", type=int, default=128)
    parser.add_argument("--val-interval-steps", type=int, default=500)
    parser.add_argument("--val-windows", type=int, default=16)
    parser.add_argument("--pause-file", type=Path, default=Path("control/PAUSE"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--init-from", type=Path, help="initialize a new run from another run's checkpoint")
    parser.add_argument("--long-context", action="store_true", help="continue using 8/16/32 KiB byte buckets")
    parser.add_argument("--prepare-long-context-caches", action="store_true",
                        help="pretokenize all three buckets, then exit without training")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true")
    resume.add_argument("--resume-if-exists", action="store_true", help="resume a checkpoint, or start if no update has ever run")
    return parser


def _run(stop: threading.Event) -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.prepare_long_context_caches:
        if args.tokenizer is None:
            parser.error("--prepare-long-context-caches requires --tokenizer")
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        for path in prepare_long_context_caches(args.supply, args.tokenizer, tokenizer):
            print(path)
        return
    if not args.model or not args.run_name:
        parser.error("--model and --run-name are required for training")
    def watch_pause() -> None:
        while not stop.wait(2.0):
            if args.pause_file.exists():
                stop.set()
    threading.Thread(target=watch_pause, daemon=True).start()
    if args.window_bytes <= 0 or args.global_batch_bytes % args.window_bytes or not 64 * 1024 <= args.global_batch_bytes <= 128 * 1024:
        parser.error("global batch must be 64–128 KiB and a multiple of the nominal window size")
    if args.long_context and (args.window_bytes != LONG_CONTEXT_LENGTHS[0]
                              or args.global_batch_bytes % LONG_CONTEXT_LENGTHS[-1]):
        parser.error("long-context batches require the 8 KiB base and a global batch divisible by 32 KiB")
    if args.stop_bytes <= 0 or args.schedule_bytes <= 0 or args.calibration_windows <= 0:
        parser.error("byte targets and calibration window count must be positive")
    verify_supply(args.supply, stop)
    windows_per_step = args.global_batch_bytes // args.window_bytes
    run_dir = args.run_root / args.run_name
    if Path(args.run_name).name != args.run_name:
        parser.error("run name must be a single path component")
    has_checkpoint = (run_dir / "ckpt/latest.json").exists()
    if args.long_context and not args.init_from and not has_checkpoint:
        parser.error("a new long-context run requires --init-from")
    if has_checkpoint and not (args.resume or args.resume_if_exists):
        parser.error("run already has a checkpoint; pass --resume")
    if args.resume_if_exists and not has_checkpoint and ((run_dir / "log.jsonl").exists() or (run_dir / "status.json").exists()):
        parser.error("run has prior activity but its checkpoint is missing")
    tokenizer = Tokenizer.from_file(str(args.tokenizer)) if args.tokenizer else None
    tokenizer_hash = sha256_file(args.tokenizer) if args.tokenizer else None
    manifest_hash = sha256_file(args.supply / "manifest.json")
    if args.expected_manifest_sha256 and args.expected_manifest_sha256 != manifest_hash:
        raise ValueError("supply manifest differs from the pilot used to create the job")
    if args.expected_tokenizer_sha256 and args.expected_tokenizer_sha256 != tokenizer_hash:
        raise ValueError("tokenizer differs from the pilot used to create the job")
    if tokenizer is not None and "BPE16K" in args.model:
        metadata_path = args.tokenizer.parent / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata.get("tokenizer_sha256") != tokenizer_hash or
                metadata.get("train_manifest_sha256") != manifest_hash or
                metadata.get("vocab_size") != tokenizer.get_vocab_size()):
            raise ValueError("16K tokenizer metadata does not match this tokenizer and training supply")
    with ExitStack() as stack:
        sizes = LONG_CONTEXT_LENGTHS if args.long_context else (args.window_bytes,)
        window_buckets = {size: stack.enter_context(WindowIndex(args.supply / "train.bin", size)) for size in sizes}
        windows = window_buckets[args.window_bytes]
        initial = preset(args.model, hash_seed=args.hash_seed)
        if isinstance(initial, BPEConfig):
            if tokenizer is None:
                parser.error("BPE models need --tokenizer")
            size = tokenizer.get_vocab_size()
            bos = tokenizer.token_to_id("<|endoftext|>") if "50K" in args.model else tokenizer.token_to_id("<|bos|>")
            if bos is None:
                parser.error("tokenizer is missing its expected BOS token")
            model_config = preset(args.model, vocab_size=size, bos_id=bos)
        else:
            if tokenizer is not None:
                parser.error("byte models do not use a tokenizer")
            sample_count = min(args.calibration_windows, len(windows))
            samples = [windows[sample_window_index(args.calibration_seed, i, 0, 1, len(windows))].data for i in range(sample_count)]
            calibration = calibrate_min_len(samples, initial.target_patch_mean, initial.max_len, initial.block_size, stop.is_set)
            model_config = preset(args.model, min_len=calibration.min_len, hash_seed=args.hash_seed)
            run_dir.mkdir(parents=True, exist_ok=True)
            calibration_record = {**asdict(calibration), "seed": args.calibration_seed, "windows": sample_count,
                                  "data_manifest_sha256": manifest_hash}
            calibration_path = run_dir / "calibration.json"
            serialized = json.dumps(calibration_record, indent=2, sort_keys=True) + "\n"
            if calibration_path.exists() and calibration_path.read_text(encoding="utf-8") != serialized:
                raise ValueError("calibration differs from the existing run")
            calibration_path.write_text(serialized, encoding="utf-8")
        peak_lr = args.learning_rate or (6e-4 if args.model.startswith("S-") else 3e-4)
        schedule_steps = math.ceil(args.schedule_bytes / args.global_batch_bytes)
        config = TrainingConfig(model_config, args.init_seed, args.data_seed, windows_per_step, args.micro_batch_size,
                                args.window_bytes, peak_lr, schedule_steps, val_interval_steps=args.val_interval_steps,
                                val_windows=args.val_windows, compile_model=args.compile)
        seed_everything(args.init_seed)
        token_cache = None
        token_caches = {}
        if isinstance(model_config, BPEConfig):
            assert tokenizer_hash is not None
            cache_root = args.supply / "tokenized" / tokenizer_hash
            for size in sizes:
                sized = cache_root / f"window-{size}"
                if not args.long_context:
                    cache_path = cache_root if (cache_root / "meta.json").is_file() else sized
                else:
                    cache_path = sized if (sized / "meta.json").is_file() else cache_root if size == 8192 else sized
                if not (cache_path / "meta.json").is_file():
                    raise FileNotFoundError(f"pretokenized windows not found at {cache_path}; "
                                            "run scripts/train.py --prepare-long-context-caches or scripts/pretokenize.py")
                token_caches[size] = TokenCache(cache_path, window_buckets[size], manifest_hash, tokenizer_hash,
                                                model_config.vocab_size, stop=stop)
            token_cache = token_caches[args.window_bytes]
        if args.long_context:
            trainer = LongContextTrainer(config, make_model(model_config), window_buckets, run_dir, torch.device(args.device),
                                         manifest_hash, tokenizer_hash, tokenizer, token_caches)
        else:
            trainer = Trainer(config, make_model(model_config), windows, run_dir, torch.device(args.device),
                              manifest_hash, tokenizer_hash, tokenizer, token_cache)
        if args.resume or (args.resume_if_exists and has_checkpoint):
            trainer.load()
            if args.init_from and (trainer.initialization is None or
                                   trainer.initialization["checkpoint"] != str(args.init_from.resolve())):
                raise ValueError("--init-from differs from the checkpoint's source run")
        elif args.init_from:
            trainer.initialize_from(args.init_from)
        trainer.run(args.supply, args.stop_bytes, args.pause_file, stop_event=stop)


def main() -> None:
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stop.set())
    try:
        _run(stop)
    except InterruptedError:
        # No optimizer update has run before startup verification/calibration completes.
        outcome = os.environ.get("BLT_QUEUE_OUTCOME_FILE")
        if outcome:
            atomic_json(Path(outcome), {"status": "paused"})
        print("Startup interrupted before training; the queued job can start again safely.")


if __name__ == "__main__":
    main()
