from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from blt.atomic import atomic_json
from blt.bench import (FAST_ENCODERS, MemoryBudgetExceeded, benchmark, compile_inference, register_fast_encoder,
                       val_prompts)
from blt.config import BPEConfig
from blt.data import verify_supply
from blt.eval_io import load_evaluation_model
from blt.fast_tokenizer import gpt2_numpy_encoder, hf_byte_level_numpy_encoder


CACHE_SIZES = (1, 2, 4, 8)


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _check_fast_encoders(bundles, prompts: list[bytes]) -> None:
    """Outside timing: a registered encoder must reproduce the HF token IDs of every benchmark prompt."""
    for bundle in bundles:
        fast = FAST_ENCODERS.get(getattr(bundle, "identity", {}).get("tokenizer_sha256"))
        if fast is None:
            continue
        for prompt in prompts:
            text = prompt.decode("utf-8")
            if list(fast[1](text)) != bundle.tokenizer.encode(text, add_special_tokens=False).ids:
                raise ValueError(f"{fast[0]} differs from the HF tokenizer on a benchmark prompt")


def _run_cell(bundles, prompts, settings: dict, cell: str, args, policy: dict) -> dict:
    """Graph cache size per cell (Sol round 9): a frozen policy entry if present; otherwise the largest size in
    CACHE_SIZES whose calibration fits the memory budget. A timed run over budget is discarded and repeated with the
    next smaller size."""
    if settings["execution"] != "graph" or settings["mode"] != "prefill":
        return benchmark(bundles, prompts, graph_cache_size=args.graph_cache_size or 8, **settings)
    attempts = []
    if args.graph_cache_size is not None:
        size, source = args.graph_cache_size, "fixed"
    elif cell in policy:
        size, source = policy[cell], "frozen policy"
    else:
        size, source = None, "calibrated"
        for candidate in CACHE_SIZES:
            _release()
            try:
                calibration = benchmark(bundles, prompts, graph_cache_size=candidate, calibrate_only=True, **settings)
            except MemoryBudgetExceeded as error:
                attempts.append({"size": candidate, "phase": "calibration", "peak_reserved_bytes": error.peak})
                break
            attempts.append({"size": candidate, "phase": "calibration", "peak_reserved_bytes":
                             max(v["peak_reserved_bytes"] for v in calibration["variants"].values())})
            size = candidate
        if size is None:
            raise MemoryBudgetExceeded(attempts[0]["peak_reserved_bytes"], settings["memory_budget_bytes"],
                                       "calibration with one cached graph")
    while True:
        _release()
        try:
            result = benchmark(bundles, prompts, graph_cache_size=size, **settings)
            break
        except MemoryBudgetExceeded as error:
            attempts.append({"size": size, "phase": error.phase, "peak_reserved_bytes": error.peak})
            smaller = [value for value in CACHE_SIZES if value < size]
            if not smaller or source == "fixed":
                raise
            size, source = smaller[-1], "reduced after a timed run over budget"
    if args.cache_policy is not None and source != "fixed":
        policy[cell] = size
        atomic_json(args.cache_policy, policy)
    result["graph_cache_policy"] = {"cell": cell, "size": size, "source": source, "attempts": attempts}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure paired Track B inference latency")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--run", type=Path)
    choice.add_argument("--preset")
    choice.add_argument("--compare", type=Path, nargs=2, metavar=("RUN_A", "RUN_B"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-a", type=Path)
    parser.add_argument("--checkpoint-b", type=Path)
    parser.add_argument("--tokenizer", type=Path, help="tokenizer for a single model or comparison A")
    parser.add_argument("--tokenizer-b", type=Path)
    parser.add_argument("--min-len", type=int)
    parser.add_argument("--supply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("prefill", "decode", "e2e", "all"), default="all")
    parser.add_argument("--prompt-kib", type=int, nargs="+")
    parser.add_argument("--batches", type=int, nargs="+")
    parser.add_argument("--output-bytes", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--capacity", "--patch-capacity", dest="capacity",
                        choices=("exact", "bucket64", "upper_bound"), default="bucket64",
                        help="prefill capacity policy (default: bucket64)")
    parser.add_argument("--graph-cache-size", type=int, help="fixed graph cache size; default: calibrate per cell")
    parser.add_argument("--memory-budget-bytes", type=int, default=10_500_000_000,
                        help="whole-process reserved-memory ceiling (Sol round 9)")
    parser.add_argument("--cache-policy", type=Path,
                        help="JSON of frozen per-cell graph cache sizes; missing cells are calibrated and added")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--execution", choices=("eager", "compile", "graph"),
                           help="prefill execution mode; graph capture is excluded from timing")
    execution.add_argument("--compile", action="store_true", help="alias for --execution compile")
    execution.add_argument("--graph", action="store_true", help="alias for --execution graph")
    parser.add_argument("--byte-e2e-patching", choices=("host", "device"), default="device",
                        help="where end-to-end byte patching and hashes run (device: frozen after the provisional timing)")
    parser.add_argument("--bpe-encoder", choices=("tiktoken", "hf"), default="tiktoken",
                        help="timed BPE tokenization; tiktoken is exact for both tokenizers (scripts/tokenizer_check.py)")
    parser.add_argument("--bf16-reduced-precision-reduction", choices=("false", "true"), default="false",
                        help="cuBLAS bf16 GEMM reductions; false (fp32) matches training and scoring and is primary")
    args = parser.parse_args()
    # Set before loading, compiling or capturing anything, for both families.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = args.bf16_reduced_precision_reduction == "true"
    numerics = {"bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                "tf32_matmul": torch.backends.cuda.matmul.allow_tf32}
    execution_mode = args.execution or ("compile" if args.compile else "graph" if args.graph else "eager")
    if args.warmup < 10 or args.repetitions < 30 or args.output_bytes < 1:
        parser.error("benchmark protocol requires at least 10 warm-ups and 30 repetitions")
    if args.graph_cache_size is not None and args.graph_cache_size not in CACHE_SIZES:
        parser.error(f"graph cache size must be one of {CACHE_SIZES}")
    if args.compare and args.checkpoint:
        parser.error("--checkpoint is for a single run; use --checkpoint-a/--checkpoint-b for comparisons")
    if not args.compare and (args.checkpoint_a or args.checkpoint_b):
        parser.error("--checkpoint-a and --checkpoint-b require --compare")
    verify_supply(args.supply)
    if args.compare:
        bundles = [load_evaluation_model(run=run, checkpoint=checkpoint, tokenizer_path=tokenizer, device=args.device, supply=args.supply)
                   for run, checkpoint, tokenizer in zip(args.compare, (args.checkpoint_a, args.checkpoint_b),
                                                          (args.tokenizer, args.tokenizer_b))]
    else:
        bundles = [load_evaluation_model(run=args.run, checkpoint=args.checkpoint, preset_name=args.preset,
                                         min_len=args.min_len, tokenizer_path=args.tokenizer, device=args.device,
                                         supply=args.supply, seed=args.seed)]
    tokenizers = (args.tokenizer, args.tokenizer_b) if args.compare else (args.tokenizer,)
    if args.bpe_encoder == "tiktoken":
        for bundle, tokenizer in zip(bundles, tokenizers):
            if isinstance(getattr(bundle, "config", None), BPEConfig) and tokenizer:
                if bundle.config.vocab_size == 50257:
                    name, encode = "tiktoken-gpt2", gpt2_numpy_encoder(Path(tokenizer).parent)
                else:
                    name, encode = "tiktoken-hf-byte-level", hf_byte_level_numpy_encoder(tokenizer)
                register_fast_encoder(bundle.identity["tokenizer_sha256"], name, encode)
    if execution_mode == "compile":
        for bundle in bundles:
            compile_inference(bundle.model)
    policy = json.loads(args.cache_policy.read_text(encoding="utf-8")) if args.cache_policy and args.cache_policy.exists() else {}
    results = []
    modes = ("prefill", "decode", "e2e") if args.mode == "all" else (args.mode,)
    for mode in modes:
        sizes = args.prompt_kib or ([2, 4, 8, 16, 32] if mode == "prefill" else [2])
        batches = args.batches or ([1, 4, 8] if mode == "prefill" else [1, 8, 32, 64])
        for kib in sizes:
            if kib <= 0:
                parser.error("prompt sizes must be positive")
            if any(batch <= 0 for batch in batches):
                parser.error("batch sizes must be positive")
            prompts = val_prompts(str(args.supply), kib * 1024,
                                  (args.warmup + args.repetitions) * max(batches), args.seed + kib)
            _check_fast_encoders(bundles, prompts)
            for batch in batches:
                settings = dict(mode=mode, batch=batch, output_bytes=args.output_bytes, warmup=args.warmup,
                                repetitions=args.repetitions, seed=args.seed, execution=execution_mode,
                                patch_capacity=args.capacity, memory_budget_bytes=args.memory_budget_bytes,
                                byte_end_to_end_patching=args.byte_e2e_patching)
                try:
                    result = _run_cell(bundles, prompts, settings, f"{mode}|{kib}|{batch}", args, policy)
                    result["prompt_kib"] = kib
                except Exception as error:
                    result = {"mode": mode, "prompt_kib": kib, "batch": batch, "error": str(error)}
                result["numerics"] = numerics
                results.append(result)
                atomic_json(args.output, {"results": results})
    print(json.dumps({"results": results}, indent=2, sort_keys=True))
    if any("error" in result for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
