from __future__ import annotations

import argparse
from pathlib import Path

from blt.atomic import atomic_json
from blt.data import iter_eval_chunks, verify_supply
from blt.eval_io import load_evaluation_model
from blt.score import configure_matmul_precision
from blt.tasks import evaluate_candidates, evaluate_retrieval, evaluate_robustness, load_local_examples, retrieval_cases, robustness_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Track B retrieval, robustness, PIQA, or HellaSwag")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--supply", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--task", choices=("retrieval", "robustness", "piqa", "hellaswag"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--long-context", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.limit <= 0 or (args.task == "retrieval" and args.trials < 200):
        parser.error("limit must be positive and retrieval requires at least 200 trials per cell")
    if args.task in {"retrieval", "robustness"} and args.supply is None:
        parser.error("retrieval and robustness require a held-out --supply")
    if args.task in {"piqa", "hellaswag"} and args.dataset is None:
        parser.error("PIQA and HellaSwag require a local --dataset")
    if args.supply is not None:
        verify_supply(args.supply)
    # Candidate log-likelihoods are scored under bf16 autocast, so use the same frozen policy as BPB scoring.
    configure_matmul_precision()
    bundle = load_evaluation_model(run=args.run, checkpoint=args.checkpoint, tokenizer_path=args.tokenizer,
                                   device=args.device, supply=args.supply)
    if args.task in {"piqa", "hellaswag"}:
        result = evaluate_candidates(bundle.model, load_local_examples(args.dataset), args.task,
                                     tokenizer=bundle.tokenizer, device=bundle.device, limit=args.limit, seed=args.seed)
    else:
        chunks = []
        size = 0
        for chunk in iter_eval_chunks(args.supply, "val"):
            chunks.append(chunk.data)
            size += len(chunk.data)
            if size >= 1024 * 1024:
                break
        filler = b"\n".join(chunks)
        if args.task == "retrieval":
            cells = []
            sizes = [2, 4, 8] + ([16, 24, 32] if args.long_context else [])
            for kib in sizes:
                for depth in (0.1, 0.5, 0.9):
                    seed = args.seed + kib * 1000 + int(depth * 100)
                    cases = retrieval_cases(filler, kib * 1024, depth, args.trials, seed)
                    cell = evaluate_retrieval(bundle.model, cases, tokenizer=bundle.tokenizer, device=bundle.device)
                    cell.update(context_kib=kib, depth=depth, seed=seed)
                    cells.append(cell)
                    atomic_json(args.output, {"task": args.task, "checkpoint": bundle.identity, "cells": cells})
            result = {"task": args.task, "cells": cells}
        else:
            pairs = robustness_pairs(filler.decode("utf-8"), args.limit)
            result = {"task": args.task, **evaluate_robustness(bundle.model, pairs, tokenizer=bundle.tokenizer, device=bundle.device)}
    result["checkpoint"] = bundle.identity
    atomic_json(args.output, result)
    print(f"Wrote {args.task} results to {args.output}")


if __name__ == "__main__":
    main()
