"""Track A2, bounded router programming on Bolmo-1B (design frozen in STATUS 2026-11-12 04:50 before any run;
Sol round 18 recommendation, with my changes recorded there). Weight and log-odds edits only, no gradients.

Conditions, all scored on the same first >= 2 MiB of the validation split (8 KiB windows, folded BPB under Bolmo's
prefill layout; the test split stays closed):
  stock    unedited model (also checked against the Track A1 per-chunk results);
  scalar   prefill threshold offset beta_in only, calibrated to P_in = 6.0;
  arm_w    Q/K interpolation gamma = 0.5 (src/blt/router.py symmetrise_qk), then beta_in calibrated to 6.0;
  arm_p    fixed byte-pair prior, strength 1: +1 when the current byte is ASCII whitespace or punctuation,
           -1 when the current and previous bytes are both ASCII digits, 0 otherwise; then beta_in calibrated to 6.0.
Calibration uses training-split documents only (the first 32 documents with >= 2048 bytes, their first 2048 bytes),
bisection over beta_in in [-20, 20], 24 steps. Decode-side routing (beta_out) is not calibrated: folded BPB under the
prefill layout does not depend on it (adjust_joint_logits preserves each byte's folded mass), so every claim here is
about the prefill layout only. Every edit is audited and restored exactly; a failed restore aborts the run.
A smoke gate runs first and aborts on failure: stock scores reproduce Track A1 per-chunk nats on 8 chunks (relative
1e-4), a +3 prefill offset changes the patch count, and Q/K edit plus restore leaves no changed tensor.
usage: python scripts/router_bounded.py --bolmo DIR --supply DIR --trackA-bpb FILE --output FILE
"""
from __future__ import annotations

import argparse
import json
import math
import string
import time
from pathlib import Path

import numpy as np
import torch

from blt.atomic import atomic_json
from blt.data import iter_eval_chunks, verify_supply
from blt.router import RouterProgram, _prefill_boundaries, audit, restore, snapshot, symmetrise_qk
from blt.trackA import load_reference, score_bolmo_chunk

TARGET_P = 6.0
TOLERANCE = 0.03
GAMMA = 0.5
EVAL_BYTES = 2 * 1024 * 1024
CAL_DOCS, CAL_BYTES = 32, 2048
SENTINELS = {
    "numeric_identifier": ["0123456789", "user_id_123", "order 4417 shipped on 2024-03-09", "x = 3.14159",
                           "call 0800 555 0199 before 17:45"],
    "delimiter_structure": ["{[()]}", "name=value;path/to/file", ":: -> != <=", '{"key": [1, 2, 3], "ok": true}',
                            "C:\\Users\\data\\file.txt"],
}
REGRESSION_NATS_PER_BYTE = 0.02


def prior_table() -> torch.Tensor:
    table = torch.zeros(256, 256)
    delimiters = [ord(c) for c in string.whitespace + string.punctuation]
    digits = list(range(ord("0"), ord("9") + 1))
    table[:, delimiters] = 1.0
    for a in digits:
        table[a, digits] = -1.0
    return table


def p_in(model, docs, table, beta, strength):
    with RouterProgram(model, table, beta_in=beta, strength=strength):
        boundaries = [sum(_prefill_boundaries(model, raw)[1:-1]) for raw in docs]
    return sum(map(len, docs)) / (len(docs) + sum(boundaries))


def calibrate_in(model, docs, table=None, strength=0.0):
    left, right, trials = -20.0, 20.0, []
    for _ in range(24):
        middle = (left + right) / 2
        value = p_in(model, docs, table, middle, strength)
        trials.append({"beta_in": middle, "p_in": value})
        if value > TARGET_P:
            left = middle
        else:
            right = middle
    best = min(trials, key=lambda t: abs(t["p_in"] / TARGET_P - 1))
    return best, trials


def evaluate(bundle, chunks, table=None, beta=0.0, strength=0.0):
    rows = []
    with RouterProgram(bundle.model, table, beta_in=beta, strength=strength):
        for chunk in chunks:
            row = score_bolmo_chunk(bundle, chunk.data)
            rows.append({"document": chunk.document, "start": chunk.start, "bytes": row["bytes"], "nats": row["nats"],
                         "patches": row["patches"]})
        sentinels = {name: [score_bolmo_chunk(bundle, text.encode("utf-8")) for text in texts]
                     for name, texts in SENTINELS.items()}
    size, nats, patches = (sum(r[k] for r in rows) for k in ("bytes", "nats", "patches"))
    return {"bpb": nats / (size * math.log(2)), "bytes": size, "bytes_per_patch": size / patches, "rows": rows,
            "sentinels": {name: {"nats_per_byte": sum(r["nats"] for r in items) / sum(r["bytes"] for r in items),
                                 "per_text_nats": [r["nats"] for r in items]} for name, items in sentinels.items()}}


def paired_gain(base, arm, seed=0, draws=2000):
    """BPB(base) - BPB(arm); positive means the arm is better. Chunk bootstrap, byte-weighted."""
    b = np.array([r["nats"] for r in base["rows"]]); a = np.array([r["nats"] for r in arm["rows"]])
    n = np.array([r["bytes"] for r in base["rows"]], dtype=float)
    rng = np.random.default_rng(seed)
    point = (b.sum() - a.sum()) / (n.sum() * math.log(2))
    idx = rng.integers(0, len(n), size=(draws, len(n)))
    samples = (b[idx].sum(1) - a[idx].sum(1)) / (n[idx].sum(1) * math.log(2))
    return {"value": float(point), "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bolmo", type=Path, required=True)
    parser.add_argument("--supply", type=Path, required=True)
    parser.add_argument("--trackA-bpb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    verify_supply(args.supply)
    bundle = load_reference("bolmo", args.bolmo)
    model = bundle.model
    reference = snapshot(model)
    chunks, size = [], 0
    for chunk in iter_eval_chunks(args.supply, "val"):
        chunks.append(chunk); size += len(chunk.data)
        if size >= EVAL_BYTES:
            break
    docs, seen = [], set()
    for chunk in iter_eval_chunks(args.supply, "train"):
        if chunk.document not in seen and len(chunk.data) >= CAL_BYTES:
            seen.add(chunk.document); docs.append(chunk.data[:CAL_BYTES])
        if len(docs) == CAL_DOCS:
            break
    if seen & {c.document for c in chunks}:
        raise ValueError("calibration documents overlap the evaluation chunks")
    report = {"design": __doc__, "target_p_in": TARGET_P, "tolerance": TOLERANCE, "gamma": GAMMA,
              "eval_chunks": len(chunks), "calibration_documents": len(docs), "checkpoint": bundle.identity,
              "regression_nats_per_byte": REGRESSION_NATS_PER_BYTE, "conditions": {}}

    # Smoke gate.
    trackA = {(r["document"], r["start"]): r for r in json.load(open(args.trackA_bpb))["models"]["bolmo"]["chunks"]}
    smoke = []
    for chunk in chunks[:8]:
        row = score_bolmo_chunk(bundle, chunk.data)
        old = trackA[(chunk.document, chunk.start)]
        smoke.append(abs(row["nats"] - old["nats"]) / old["nats"])
    with RouterProgram(model, beta_in=3.0):
        shifted = score_bolmo_chunk(bundle, chunks[0].data)
    plain = score_bolmo_chunk(bundle, chunks[0].data)
    calibration_plain, calibration_shifted = p_in(model, docs[:2], None, 0.0, 0.0), p_in(model, docs[:2], None, 3.0, 0.0)
    symmetrise_qk(model, GAMMA)
    changed = [t["name"] for t in audit(model, reference)]
    restore(model, reference)
    gate = {"max_relative_nats_gap_vs_trackA": max(smoke), "patches_plain": plain["patches"],
            "patches_beta_plus3": shifted["patches"], "qk_changed": changed,
            "calibration_p_in_plain": calibration_plain, "calibration_p_in_beta_plus3": calibration_shifted,
            "changes_after_restore": len(audit(model, reference))}
    report["smoke_gate"] = gate
    if (gate["max_relative_nats_gap_vs_trackA"] > 1e-4 or gate["patches_beta_plus3"] == gate["patches_plain"]
            or not calibration_shifted < calibration_plain or not changed or gate["changes_after_restore"]):
        atomic_json(args.output, report)
        raise SystemExit(f"smoke gate failed: {gate}")
    print("smoke gate ok", json.dumps(gate), flush=True)

    table = prior_table()
    plans = [("stock", None, None, 0.0, False), ("scalar", None, None, 0.0, True),
             ("arm_w", GAMMA, None, 0.0, True), ("arm_p", None, table, 1.0, True)]
    for name, gamma, prior, strength, calibrated in plans:
        if gamma is not None:
            symmetrise_qk(model, gamma)
        cell = {"gamma": gamma, "strength": strength, "changed_tensors": audit(model, reference)}
        beta = 0.0
        if calibrated:
            best, trials = calibrate_in(model, docs, prior, strength)
            beta = best["beta_in"]
            cell.update(calibration=best, calibration_trials=trials,
                        calibration_matched=abs(best["p_in"] / TARGET_P - 1) <= TOLERANCE)
        cell["beta_in"] = beta
        cell["evaluation"] = evaluate(bundle, chunks, prior, beta, strength)
        restore(model, reference)
        if audit(model, reference):
            raise RuntimeError("model was not restored exactly")
        cell["elapsed_seconds"] = time.time() - started
        report["conditions"][name] = cell
        atomic_json(args.output, report)
        e = cell["evaluation"]
        print(f"{name}: bpb {e['bpb']:.4f} bytes/patch {e['bytes_per_patch']:.3f} beta_in {beta:+.3f}", flush=True)

    scalar = report["conditions"]["scalar"]["evaluation"]
    verdicts = {}
    for name in ("arm_w", "arm_p"):
        cell = report["conditions"][name]
        e = cell["evaluation"]
        gain = paired_gain(scalar, e)
        p_match = abs(e["bytes_per_patch"] / scalar["bytes_per_patch"] - 1) <= TOLERANCE
        regressions = [k for k in SENTINELS
                       if e["sentinels"][k]["nats_per_byte"] - scalar["sentinels"][k]["nats_per_byte"] > REGRESSION_NATS_PER_BYTE]
        status = ("inconclusive" if not (cell["calibration_matched"] and p_match and
                                          report["conditions"]["scalar"]["calibration_matched"])
                  else "pass" if gain["value"] >= 0.01 and not regressions else "fail")
        verdicts[name] = {"gain_bpb_vs_scalar": gain, "eval_p_matched": p_match, "sentinel_regressions": regressions,
                          "status": status}
    report["verdicts"] = verdicts
    report["elapsed_seconds"] = time.time() - started
    atomic_json(args.output, report)
    print(json.dumps(verdicts), flush=True)


if __name__ == "__main__":
    main()
