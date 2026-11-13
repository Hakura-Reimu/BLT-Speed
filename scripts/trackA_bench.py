from __future__ import annotations

import argparse
from pathlib import Path

import torch

from blt.atomic import atomic_json
from blt.bench import val_prompts
from blt.data import verify_supply
from blt.trackA import load_reference
from blt.trackA_bench import benchmark_bolmo, benchmark_pair, run_bolmo


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired Track A stock and forced-layout inference timings")
    parser.add_argument("--bolmo", type=Path, required=True)
    parser.add_argument("--olmo", type=Path, required=True)
    parser.add_argument("--bolmo-subword-tokenizer", type=Path)
    parser.add_argument("--supply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("prefill", "decode", "all"), default="all")
    parser.add_argument("--prompt-kib", type=int, nargs="+", help="override sizes for the selected mode")
    parser.add_argument("--batches", type=int, nargs="+", help="override batch sizes")
    parser.add_argument("--forced-p", nargs="+", default=["stock", "4.4", "6", "8", "10", "12", "16"])
    parser.add_argument("--output-bytes", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.warmup < 10 or args.repetitions < 30 or args.output_bytes < 1:
        parser.error("protocol requires at least 10 warm-ups and 30 repetitions")
    try:
        patch_lengths = [None if x == "stock" else float(x) for x in args.forced_p]
    except ValueError:
        parser.error("forced-p values must be stock, 4.4, 6, 8, 10, 12 or 16")
    if any(x not in (None, 4.4, 6, 8, 10, 12, 16) for x in patch_lengths):
        parser.error("unsupported forced-p value")
    verify_supply(args.supply)
    bolmo = load_reference("bolmo", args.bolmo, bolmo_subword_tokenizer=str(args.bolmo_subword_tokenizer) if args.bolmo_subword_tokenizer else None)
    olmo = load_reference("olmo", args.olmo)
    result = {"protocol": {"paired_order": "alternating", "olmo_prefill_logits_to_keep": 1,
                           "bolmo_prefill": "stock generate through replay of the rolled-back last prompt byte",
                           "prefill_variants": "gpu_resident and end_to_end; stock Bolmo suffix reconstruction remains timed in both",
                           "generation": "greedy raw byte outputs with specials suppressed; OLMo byte overshoot reported",
                           "forced_4.4_gaps": [4, 4, 5, 4, 5],
                           "event_timings": "separate instrumented run; inclusive and overlapping, not additive"}, "results": []}
    failures = False
    for mode in (("prefill", "decode") if args.mode == "all" else (args.mode,)):
        for kib in (args.prompt_kib or ([2, 4, 8, 16] if mode == "prefill" else [2])):
            if kib < 1:
                parser.error("prompt sizes must be positive")
            prompts = val_prompts(str(args.supply), 1024 * kib, args.warmup + args.repetitions, args.seed + kib)
            for batch in (args.batches or ([1, 4] if mode == "prefill" else [1, 8])):
                if batch < 1:
                    parser.error("batch sizes must be positive")
                for p in patch_lengths:
                    generated = 1 if mode == "prefill" else args.output_bytes
                    try:
                        row = {"warmup": args.warmup, "repetitions": args.repetitions,
                               "requested_output_bytes_per_row": generated, "variants": {}}
                        variant_names = ("gpu_resident", "end_to_end") if mode == "prefill" else ("decode",)
                        for variant in variant_names:
                            resident = variant != "end_to_end"
                            try:
                                sample = benchmark_pair(bolmo, olmo, prompts, batch=batch, output_bytes=generated,
                                                        warmup=args.warmup, repetitions=args.repetitions, forced_p=p,
                                                        seed=args.seed, resident=resident)
                            except ValueError as exc:
                                if "position limit" not in str(exc):
                                    raise
                                sample = benchmark_bolmo(bolmo, prompts, batch=batch, output_bytes=generated,
                                                         warmup=args.warmup, repetitions=args.repetitions,
                                                         forced_p=p, resident=resident)
                                sample["variants"][mode]["olmo_error"] = str(exc)
                                failures = True
                            row["variants"][variant] = sample["variants"][mode]
                        profile = run_bolmo(bolmo, prompts[0], batch, generated, p=p, profile=True)
                        row["bolmo_event_profile"] = profile["inclusive_cuda_event_ms"]
                    except Exception as exc:
                        row = {"error": f"{type(exc).__name__}: {exc}"}
                        failures = True
                        torch.cuda.empty_cache()
                    row.update({"mode": mode, "prompt_kib": kib, "batch": batch, "forced_p": p})
                    result["results"].append(row)
                    atomic_json(args.output, result)
                    print(f"{mode} {kib} KiB batch={batch} P={p or 'stock'}: {'error' if 'error' in row else 'done'}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
