"""Diagnostic (not registered): is the byte models' chance-level pass-key ranking real or a scoring bug?
A. the same cases at short contexts (256 B, 512 B, 1 KiB; depth 0.5);
B. the first key digit alone (10 one-digit candidates) at 2 KiB, depth 0.9;
C. a pure copy prompt with no filler: "Key: 77596. Key: 77596. Key: 77596. Key:".
CPU, FP32 (autocast is disabled on the CPU)."""
import json, random, sys
from pathlib import Path

import torch

from blt.data import iter_eval_chunks
from blt.eval_io import load_evaluation_model
from blt.retrieval import evaluate_retrieval, retrieval_prompt
from blt.tasks import continuation_nlls, retrieval_cases

SUPPLY = Path.home() / "blt/data/supply-v1"
MODELS = {"m-p6": (None,), "m-bpe16k": (Path.home() / "blt/data/tokenizer16k-v1/tokenizer.json",)}
torch.set_num_threads(4)
chunks, size = [], 0
for chunk in iter_eval_chunks(SUPPLY, "val"):
    chunks.append(chunk.data); size += len(chunk.data)
    if size >= 1024 * 1024:
        break
filler = b"\n".join(chunks)
out = {}
for run, (tokenizer,) in MODELS.items():
    bundle = load_evaluation_model(run=Path.home() / f"blt/runs/{run}-seed0",
                                   checkpoint=Path.home() / f"blt/runs/{run}-seed0/ckpt/bytes-3000000000.pt",
                                   tokenizer_path=tokenizer, device="cpu", supply=SUPPLY)
    res = {}
    for size_bytes in (256, 512, 1024):
        cases = retrieval_cases(filler, size_bytes, 0.5, 40, 7 + size_bytes)
        cell = evaluate_retrieval(bundle.model, cases, tokenizer=bundle.tokenizer, device="cpu")
        res[f"A_{size_bytes}"] = {"rank1": cell["rank1_accuracy"], "mean_rank": cell["mean_true_key_rank"]}
        print(run, size_bytes, res[f"A_{size_bytes}"], flush=True)
    cases = retrieval_cases(filler, 2048, 0.9, 40, 2090)
    hits, ranks = 0, []
    for case in cases:
        nll = continuation_nlls(bundle.model, retrieval_prompt(case), [f" {d}" for d in range(10)],
                                tokenizer=bundle.tokenizer, device="cpu")
        true = int(case["key"][0])
        rank = 1 + sum(v < nll[true] for i, v in enumerate(nll) if i != true)
        ranks.append(rank); hits += rank == 1
    res["B_first_digit_2k"] = {"rank1": hits / len(cases), "mean_rank": sum(ranks) / len(ranks)}
    print(run, "first digit", res["B_first_digit_2k"], flush=True)
    rng = random.Random(3)
    hits, ranks = 0, []
    for _ in range(40):
        key = f"{rng.randrange(100000):05d}"
        others = set()
        while len(others) < 9:
            value = f"{rng.randrange(100000):05d}"
            if value != key:
                others.add(value)
        prompt = f"Key: {key}. Key: {key}. Key: {key}. Key:"
        nll = continuation_nlls(bundle.model, prompt, [" " + key, *(" " + v for v in sorted(others))],
                                tokenizer=bundle.tokenizer, device="cpu")
        rank = 1 + sum(v < nll[0] for v in nll[1:])
        ranks.append(rank); hits += rank == 1
    res["C_pure_copy"] = {"rank1": hits / 40, "mean_rank": sum(ranks) / len(ranks)}
    print(run, "pure copy", res["C_pure_copy"], flush=True)
    out[run] = res
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
