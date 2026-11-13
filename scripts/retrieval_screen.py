"""Pass-key retrieval with prompt format v2 (blt.retrieval); otherwise the same cells as scripts/eval_tasks.py
--task retrieval: 1 MiB of validation-split filler, depths 0.1/0.5/0.9, cell seed = seed + KiB*1000 + depth*100,
at least 200 trials per cell. Writes the result after every cell (atomic)."""
from __future__ import annotations

import argparse
from pathlib import Path

from blt.atomic import atomic_json
from blt.data import iter_eval_chunks, verify_supply
from blt.eval_io import load_evaluation_model
from blt.retrieval import PROMPT_FORMAT, evaluate_retrieval
from blt.score import configure_matmul_precision
from blt.tasks import retrieval_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--supply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kib", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.trials < 200 or any(k <= 0 for k in args.kib):
        parser.error("retrieval requires at least 200 trials per cell and positive sizes")
    verify_supply(args.supply)
    configure_matmul_precision()
    bundle = load_evaluation_model(run=args.run, checkpoint=args.checkpoint, tokenizer_path=args.tokenizer,
                                   device=args.device, supply=args.supply)
    chunks, size = [], 0
    for chunk in iter_eval_chunks(args.supply, "val"):
        chunks.append(chunk.data)
        size += len(chunk.data)
        if size >= 1024 * 1024:
            break
    filler = b"\n".join(chunks)
    cells = []
    result = {"task": "retrieval", "prompt_format": PROMPT_FORMAT, "checkpoint": bundle.identity, "cells": cells}
    for kib in args.kib:
        for depth in (0.1, 0.5, 0.9):
            seed = args.seed + kib * 1000 + int(depth * 100)
            cases = retrieval_cases(filler, kib * 1024, depth, args.trials, seed)
            cell = evaluate_retrieval(bundle.model, cases, tokenizer=bundle.tokenizer, device=bundle.device)
            cell.update(context_kib=kib, depth=depth, seed=seed)
            cells.append(cell)
            atomic_json(args.output, result)
    print(f"Wrote retrieval results to {args.output}")


if __name__ == "__main__":
    main()
