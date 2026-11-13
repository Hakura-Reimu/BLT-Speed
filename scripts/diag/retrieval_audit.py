"""Retrieval scoring audit (Sol round 18; diagnostic, not registered). CPU, FP32.
1. Cached teacher-forced scoring: prefill the prompt with the decoding path (init_state/prefill/step), score each
   candidate byte or token from the cached next-unit distribution, teacher-force it, and compare the summed NLL with
   the full-forward continuation_nlls used by the screen (tolerance 1e-3 nats per candidate).
2. Counterfactual key substitution: the same context with the embedded key A replaced by B (same length and
   position), candidates fixed to {A, B}. If the model reads the key from context, the preference for A over B must
   move towards B: delta = [NLL(A|ctx A) - NLL(B|ctx A)] - [NLL(A|ctx B) - NLL(B|ctx B)] < 0.
usage: PYTHONPATH=src python scripts/diag/retrieval_audit.py OUTPUT.json [--cases 20] [--bytes 1024]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from blt.config import BPEConfig
from blt.data import iter_eval_chunks
from blt.eval_io import load_evaluation_model
from blt.data import tokenize_window
from blt.retrieval import retrieval_prompt
from blt.tasks import continuation_nlls, retrieval_cases

HOME = Path.home()
SUPPLY = HOME / "blt/data/supply-v1"
MODELS = {"m-p6": None, "m-bpe16k": HOME / "blt/data/tokenizer16k-v1/tokenizer.json"}


@torch.inference_mode()
def cached_nlls(bundle, prompt: str, candidates: list[str]) -> list[float]:
    model, config = bundle.model, bundle.model.config
    out = []
    for candidate in candidates:
        if isinstance(config, BPEConfig):
            prefix = [config.bos_id, *tokenize_window(bundle.tokenizer, prompt.encode("utf-8"))]
            units = tokenize_window(bundle.tokenizer, candidate.encode("utf-8"))
        else:
            prefix, units = list(prompt.encode("utf-8")), list(candidate.encode("utf-8"))
        state = model.init_state(1, len(prefix) + len(units) + 8, "cpu", torch.float32)
        logits = model.prefill(state, torch.tensor([prefix], dtype=torch.long))
        total = 0.0
        for unit in units:
            row = logits.reshape(-1, logits.shape[-1])[-1].float()
            total -= float(F.log_softmax(row, dim=-1)[unit])
            logits = model.step(state, torch.tensor([unit], dtype=torch.long))
        out.append(total)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--bytes", type=int, default=1024)
    args = parser.parse_args()
    torch.set_num_threads(4)
    chunks, size = [], 0
    for chunk in iter_eval_chunks(SUPPLY, "val"):
        chunks.append(chunk.data)
        size += len(chunk.data)
        if size >= 1024 * 1024:
            break
    filler = b"\n".join(chunks)
    cases = retrieval_cases(filler, args.bytes, 0.5, args.cases, 31337)
    report = {"bytes": args.bytes, "cases": args.cases, "depth": 0.5, "seed": 31337, "models": {}}
    for run, tokenizer in MODELS.items():
        bundle = load_evaluation_model(run=HOME / f"blt/runs/{run}-seed0",
                                       checkpoint=HOME / f"blt/runs/{run}-seed0/ckpt/bytes-3000000000.pt",
                                       tokenizer_path=tokenizer, device="cpu", supply=SUPPLY)
        rows = []
        for case in cases:
            a, b = case["key"], case["distractors"][0]
            marker_a, marker_b = f"The pass key is {a}.", f"The pass key is {b}."
            if case["context"].count(marker_a) != 1:
                raise ValueError("marker not unique")
            case_b = dict(case, context=case["context"].replace(marker_a, marker_b))
            candidates = [" " + a, " " + b]
            full_a = continuation_nlls(bundle.model, retrieval_prompt(case), candidates, tokenizer=bundle.tokenizer)
            full_b = continuation_nlls(bundle.model, retrieval_prompt(case_b), candidates, tokenizer=bundle.tokenizer)
            cached_a = cached_nlls(bundle, retrieval_prompt(case), candidates)
            delta = (full_a[0] - full_a[1]) - (full_b[0] - full_b[1])
            rows.append({"key": a, "other": b, "full_ctx_a": full_a, "full_ctx_b": full_b, "cached_ctx_a": cached_a,
                         "max_cached_full_gap": max(abs(x - y) for x, y in zip(cached_a, full_a)), "delta": delta})
        deltas = [row["delta"] for row in rows]
        summary = {"max_cached_full_gap": max(row["max_cached_full_gap"] for row in rows),
                   "mean_delta": sum(deltas) / len(deltas),
                   "fraction_delta_negative": sum(d < 0 for d in deltas) / len(deltas),
                   "mean_abs_delta": sum(abs(d) for d in deltas) / len(deltas)}
        report["models"][run] = {"summary": summary, "rows": rows}
        print(run, json.dumps(summary), flush=True)
    args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
