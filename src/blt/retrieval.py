"""Pass-key retrieval, prompt format v2 (2026-11-11 23:30), used by scripts/retrieval_screen.py and report.py.

The cases are blt.tasks.retrieval_cases, unchanged. The v1 evaluation in blt.tasks.evaluate_retrieval ended the
prompt with a space ("... The pass key is ") and scored the bare key. That leaves GPT-2-style BPE a lone space token
followed by digit tokens with no leading space, a segmentation it rarely sees; a three-case CPU check, run before any
retrieval measurement, showed BPE greedy output breaking on it. v2 ends the prompt at "is" and scores, and generates,
the answer " 12345" with its leading space, as in ordinary text, for both model families. blt.tasks is left as it
was because its hash is part of the frozen analysis (PIQA/HellaSwag scoring).
"""
from __future__ import annotations

import hashlib
from typing import Any, Iterable

import torch

from .config import BPEConfig
from .eval_tokens import TokenBytes
from .tasks import continuation_nlls, greedy_bytes

PROMPT_FORMAT = "v2: prompt ends at 'is'; answers and greedy output start with the space"
RETRIEVAL_ANSWER_BYTES = 6  # the space and the five key digits


def retrieval_prompt(case: dict[str, Any]) -> str:
    return case["context"] + case["query"]


def evaluate_retrieval(model: torch.nn.Module, cases: Iterable[dict[str, Any]], *, tokenizer: Any = None,
                       device: str | torch.device = "cpu") -> dict[str, Any]:
    results = []
    vocabulary = TokenBytes(tokenizer, model.config.vocab_size) if isinstance(model.config, BPEConfig) else None
    for case in cases:
        prompt = retrieval_prompt(case)
        answer = " " + case["key"]
        candidates = [answer, *(" " + value for value in case["distractors"])]
        nll = continuation_nlls(model, prompt, candidates, tokenizer=tokenizer, device=device)
        rank = 1 + sum(value < nll[0] for value in nll[1:]) + 0.5 * sum(value == nll[0] for value in nll[1:])
        generated = greedy_bytes(model, prompt, RETRIEVAL_ANSWER_BYTES, tokenizer=tokenizer, device=device,
                                 vocabulary=vocabulary)
        results.append({"key": case["key"], "distractors": case["distractors"],
                        "context_sha256": hashlib.sha256(case["context"].encode("utf-8")).hexdigest(),
                        "insertion_byte": case["insertion_byte"], "depth": case["depth"],
                        "context_bytes": case["context_bytes"],
                        "greedy_answer": generated[:RETRIEVAL_ANSWER_BYTES].decode("ascii", errors="replace"),
                        "greedy_exact": generated[:RETRIEVAL_ANSWER_BYTES] == answer.encode("ascii"),
                        "generated_hex": generated.hex(),
                        "overshoot_bytes": max(0, len(generated) - RETRIEVAL_ANSWER_BYTES),
                        "true_key_rank": rank, "candidate_nll": nll})
    if not results:
        raise ValueError("no retrieval cases")
    return {"count": len(results), "greedy_accuracy": sum(row["greedy_exact"] for row in results) / len(results),
            "rank1_accuracy": sum(row["true_key_rank"] == 1 for row in results) / len(results),
            "mean_true_key_rank": sum(row["true_key_rank"] for row in results) / len(results),
            "prompt_format": PROMPT_FORMAT, "trials": results}
