from __future__ import annotations

"""CUDA-graph lifetime stress check before final timing (GPT-6.1 Sol round 9, item 2).

Under the benchmark's outer autocast, replays trunk graphs from a small LRU (forcing evictions) for prompts of varying
length (forcing bucket changes and misses). Misses run eager and are then captured, so hits, misses and evictions all
occur. Every sampled replay, and every miss, is compared with a fresh eager prefill of the same inputs.
"""

import argparse
import json
from pathlib import Path
import random

import torch

from blt.atomic import atomic_json
from blt.bench import (CUDAGraphPrefill, GraphCache, InferenceCase, _capture_missing, _eager_preprocessed,
                       _trunk_key, autocast_for, val_prompts)
from blt.config import ByteConfig
from blt.data import verify_supply
from blt.eval_io import load_evaluation_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--run", type=Path)
    choice.add_argument("--preset")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--min-len", type=int)
    parser.add_argument("--supply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--cache-size", type=int, default=4)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-2, help="max |graph - eager| / max |eager|")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    verify_supply(args.supply)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    bundle = load_evaluation_model(run=args.run, checkpoint=args.checkpoint, preset_name=args.preset,
                                   min_len=args.min_len, tokenizer_path=args.tokenizer, device="cuda",
                                   supply=args.supply, seed=args.seed)
    rng = random.Random(args.seed)
    # Three prompt lengths give several 64-position buckets for both families, more than the cache holds.
    pools = {kib: val_prompts(str(args.supply), kib * 1024, (args.replays + 10) * args.batch, args.seed + kib)
             for kib in (4, 8, 12)}
    is_byte = isinstance(bundle.config, ByteConfig)
    cache = GraphCache(args.cache_size)
    device = bundle.device
    records = []
    failures = 0
    with torch.inference_mode(), autocast_for(device):
        for trial in range(args.replays + 10):
            kib = rng.choice(sorted(pools))
            group = pools[kib][trial * args.batch:(trial + 1) * args.batch]
            case = InferenceCase(bundle, group, args.batch, prepare_preprocessed=True)
            inputs = case.resident_inputs(preprocessed=True)
            key = _trunk_key(inputs)
            source = inputs["bytes"] if is_byte else inputs["tokens"]
            runner = cache.get(key)
            calibration = trial < 10
            if runner is not None:
                _, logits = runner.replay(inputs)
                outcome = "hit"
            else:
                _, logits = _eager_preprocessed(case, inputs)
                outcome = "miss"
                cache.record_trial(False)
            graph_logits = logits.float().clone()
            checked = error = None
            # Every replay is checked; eager misses are sampled.
            if outcome == "hit" or trial % args.sample_every == 0:
                _, reference = _eager_preprocessed(case, inputs)
                reference = reference.float()
                error = float((graph_logits - reference).abs().max() / reference.abs().max().clamp_min(1e-6))
                checked = error <= args.tolerance
                failures += int(not checked)
            evicted_before = sum(1 for capture in cache.captures if capture.get("evicted"))
            if outcome == "miss":
                _capture_missing(cache, key, lambda: CUDAGraphPrefill(
                    bundle.model, inputs, max_len=source.shape[1], patching="preprocessed" if is_byte else "device"),
                    device, phase="stress", trial=trial)
            evicted = sum(1 for capture in cache.captures if capture.get("evicted")) - evicted_before
            records.append({"trial": trial, "calibration": calibration, "kib": kib, "key": list(key),
                            "outcome": outcome, "evicted": evicted, "checked": checked, "relative_error": error})
    timed = [record for record in records if not record["calibration"]]
    summary = {"model": bundle.identity, "replays": len(timed),
               "hits": sum(record["outcome"] == "hit" for record in timed),
               "misses": sum(record["outcome"] == "miss" for record in timed),
               "evictions": sum(record["evicted"] for record in timed),
               "checked": sum(record["checked"] is not None for record in timed),
               "checked_hits": sum(record["checked"] is not None and record["outcome"] == "hit" for record in timed),
               "failures": failures, "max_relative_error": max((r["relative_error"] or 0.0) for r in records),
               "tolerance": args.tolerance, "cache_size": args.cache_size, "batch": args.batch}
    atomic_json(args.output, {"summary": summary, "records": records})
    print(json.dumps(summary, indent=2, default=str))
    if failures or summary["hits"] == 0 or summary["misses"] == 0 or summary["evictions"] == 0:
        raise SystemExit("stress check failed or did not exercise hits, misses and evictions")


if __name__ == "__main__":
    main()
