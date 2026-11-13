"""Paired BPB gain of each candidate over a reference on the same validation chunks, with document-bootstrap intervals.

Inputs are score files carrying per-chunk records (scripts/score.py, scripts/diag/decoder_capacity.py). A positive gain
means the candidate has lower BPB.

usage: python paired_gain.py REFERENCE.json CANDIDATE.json [CANDIDATE.json ...] [--bootstrap 2000]
"""
import argparse
import json
import math

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("reference")
parser.add_argument("candidates", nargs="+")
parser.add_argument("--bootstrap", type=int, default=2000)
args = parser.parse_args()


def load(path: str) -> tuple[list[tuple], np.ndarray, np.ndarray, list[str]]:
    chunks = json.load(open(path, encoding="utf-8"))["chunks"]
    keys = [(chunk["document"], chunk["start"], chunk["end"]) for chunk in chunks]
    return (keys, np.array([chunk["nats"] for chunk in chunks]), np.array([chunk["bytes"] for chunk in chunks]),
            [chunk["document"] for chunk in chunks])


keys, reference, sizes, documents = load(args.reference)
groups = {}
for i, document in enumerate(documents):
    groups.setdefault(document, []).append(i)
members = list(groups.values())
group_bytes = np.array([sizes[m].sum() for m in members])
draws = np.random.default_rng(0).integers(0, len(members), (args.bootstrap, len(members)))
print(f"reference {args.reference}: {reference.sum() / (math.log(2) * sizes.sum()):.5f} BPB over {sizes.sum()} bytes, "
      f"{len(keys)} chunks, {len(members)} documents")
for path in args.candidates:
    candidate_keys, candidate, candidate_sizes, _ = load(path)
    if candidate_keys != keys or not np.array_equal(candidate_sizes, sizes):
        raise SystemExit(f"{path} does not score the same chunks as the reference")
    difference = reference - candidate
    by_group = np.array([difference[m].sum() for m in members])
    boot = by_group[draws].sum(1) / (math.log(2) * group_bytes[draws].sum(1))
    low, high = np.percentile(boot, [2.5, 97.5])
    print(f"{path}: {candidate.sum() / (math.log(2) * sizes.sum()):.5f} BPB, gain "
          f"{difference.sum() / (math.log(2) * sizes.sum()):+.5f} [{low:+.5f}, {high:+.5f}]")
