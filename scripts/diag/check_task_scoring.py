"""Independent check of the PIQA/HellaSwag candidate scorer before the task evaluation (Sol round 16; the plan's
reference-harness cross-check, which cannot run here because the models are not HF models).

For a fixed sample of items per task, every candidate's NLL from src/blt/tasks.py continuation_nlls (one padded batch
per item) is compared, in FP32 on the CPU, with an independent value: the NLL of the whole sequence (context plus
candidate, one unpadded forward pass) minus the NLL of the context alone (a second forward pass). The same token
sequences and BOS convention are used. Also checked: each candidate scored alone and the candidates in reverse order
give the same values. Agreement must be within 1e-3 nats per candidate, with the same predicted answer.

usage (CPU): python scripts/diag/check_task_scoring.py --run RUN [--tokenizer T] --output OUT.json [--items 50]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from blt.atomic import atomic_json
from blt.data import tokenize_window
from blt.eval_io import load_evaluation_model
from blt.model import BPELM, ByteLatentLM
from blt.tasks import continuation_nlls, format_candidates, load_local_examples

TOLERANCE = 1e-3


@torch.inference_mode()
def sequence_nll(model, units: list[int], first: int) -> float:
    """Summed NLL of units[first:] given everything before them, from one unpadded forward pass."""
    if isinstance(model, BPELM):
        tokens = torch.tensor([units], dtype=torch.long)
        logits = model(tokens)[0, :-1].float()
        targets = tokens[0, 1:]
        start = first - 1
    elif isinstance(model, ByteLatentLM):
        data = torch.tensor([units], dtype=torch.long)
        logits = model(data)[0].float()  # position i predicts byte i from the bytes before it
        targets = data[0]
        start = first
    else:
        raise ValueError("unknown model family")
    losses = F.cross_entropy(logits, targets, reduction="none")
    return float(losses[start:].sum())


def independent_nll(model, tokenizer, context: str, candidate: str) -> float:
    prefix, suffix = context.encode("utf-8"), candidate.encode("utf-8")
    if isinstance(model, BPELM):
        context_units = [model.config.bos_id, *tokenize_window(tokenizer, prefix)]
        full = context_units + tokenize_window(tokenizer, suffix)
        return sequence_nll(model, full, 1) - sequence_nll(model, context_units, 1)
    full = list(prefix) + list(suffix)
    return sequence_nll(model, full, 0) - sequence_nll(model, list(prefix), 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path.home() / "blt/data/eval")
    parser.add_argument("--items", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = load_evaluation_model(run=args.run, checkpoint=args.checkpoint, tokenizer_path=args.tokenizer, device="cpu")
    model, tokenizer = bundle.model.float().eval(), bundle.tokenizer
    report = {"checkpoint": bundle.identity, "tolerance_nats": TOLERANCE, "tasks": {}}
    failures = 0
    for task in ("piqa", "hellaswag"):
        records = load_local_examples(args.data_root / task)
        indices = sorted(random.Random(args.seed).sample(range(len(records)), args.items))
        worst, mismatched, rows = 0.0, 0, []
        for index in indices:
            context, candidates, label = format_candidates(task, records[index])
            batched = continuation_nlls(model, context, candidates, tokenizer=tokenizer, device="cpu")
            alone = [continuation_nlls(model, context, [c], tokenizer=tokenizer, device="cpu")[0] for c in candidates]
            reverse = continuation_nlls(model, context, candidates[::-1], tokenizer=tokenizer, device="cpu")[::-1]
            independent = [independent_nll(model, tokenizer, context, c) for c in candidates]
            gap = max(max(abs(a - b), abs(a - c), abs(a - d))
                      for a, b, c, d in zip(independent, batched, alone, reverse))
            same_answer = min(range(len(batched)), key=batched.__getitem__) == min(range(len(independent)),
                                                                                  key=independent.__getitem__)
            worst = max(worst, gap)
            mismatched += not same_answer
            rows.append({"index": index, "label": label, "batched": batched, "alone": alone, "reverse": reverse,
                         "independent": independent, "max_abs_difference": gap, "same_prediction": same_answer})
        ok = worst <= TOLERANCE and mismatched == 0
        failures += not ok
        report["tasks"][task] = {"items": indices, "max_abs_difference": worst, "prediction_mismatches": mismatched,
                                 "passed": ok, "rows": rows}
        print(f"{task}: {args.items} items, max |difference| {worst:.2e} nats, prediction mismatches {mismatched}, "
              f"{'PASS' if ok else 'FAIL'}")
    report["passed"] = failures == 0
    atomic_json(args.output, report)
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
