from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from blt.atomic import atomic_json
from blt.data import iter_eval_chunks, sha256_file, verify_supply
from blt.router import RouterProgram, audit, calibrate, edit_paired_rows, evaluate_edited, restore, snapshot, symmetrise_qk
from blt.trackA import load_reference


def tensor_file(path: Path | None) -> torch.Tensor | None:
    return None if path is None else torch.from_numpy(np.load(path, allow_pickle=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit, evaluate, and restore in-memory Bolmo router edits")
    parser.add_argument("--bolmo", type=Path, required=True)
    parser.add_argument("--bolmo-subword-tokenizer", type=Path)
    parser.add_argument("--supply", type=Path, required=True, help="evaluation supply")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--window-bytes", type=int, default=8192)
    parser.add_argument("--calibration-supply", type=Path, help="calibration uses this supply's val split only")
    parser.add_argument("--calibration-chunks", type=int, default=8)
    parser.add_argument("--calibration-bytes", type=int, default=256)
    parser.add_argument("--target-p-in", type=float)
    parser.add_argument("--target-p-out", type=float)
    parser.add_argument("--gamma", type=float, help="Arm W Q/K interpolation")
    parser.add_argument("--direction", type=Path, help="NumPy [hidden_size] array")
    parser.add_argument("--eta", type=Path, help="NumPy [256] array")
    parser.add_argument("--table", type=Path, help="NumPy [256,256] byte-pair prior")
    parser.add_argument("--strength", type=float, default=0.0)
    parser.add_argument("--beta-in", type=float, default=0.0)
    parser.add_argument("--beta-out", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.window_bytes <= 8192 or args.calibration_chunks < 1 or args.calibration_bytes < 3:
        parser.error("invalid window or calibration size")
    if (args.direction is None) != (args.eta is None):
        parser.error("--direction and --eta must be supplied together")
    if (args.target_p_in is None) != (args.target_p_out is None):
        parser.error("--target-p-in and --target-p-out must be supplied together")
    if args.target_p_in is not None and args.calibration_supply is None:
        parser.error("calibration targets require --calibration-supply")
    if args.target_p_in is not None and (args.beta_in != 0 or args.beta_out != 0):
        parser.error("calibrated offsets cannot be combined with explicit beta offsets")
    verify_supply(args.supply)
    if args.calibration_supply:
        verify_supply(args.calibration_supply)
    bundle = load_reference("bolmo", args.bolmo,
                            bolmo_subword_tokenizer=str(args.bolmo_subword_tokenizer) if args.bolmo_subword_tokenizer else None)
    reference = snapshot(bundle.model)
    table = tensor_file(args.table)
    report = {"checkpoint": bundle.identity, "split": args.split, "gamma": args.gamma,
              "strength": args.strength, "calibration": None,
              "input_sha256": {name: sha256_file(path) for name, path in
                               (("table", args.table), ("direction", args.direction), ("eta", args.eta)) if path is not None}}
    try:
        if args.gamma is not None:
            symmetrise_qk(bundle.model, args.gamma)
        if args.direction is not None:
            edit_paired_rows(bundle.model, tensor_file(args.direction), tensor_file(args.eta))
        beta_in, beta_out = args.beta_in, args.beta_out
        if args.target_p_in is not None:
            docs = []
            calibration_docs = set()
            samples = []
            for chunk in iter_eval_chunks(args.calibration_supply, "val", args.window_bytes):
                if len(chunk.data) >= 3 and chunk.document not in calibration_docs:
                    raw = chunk.data[:args.calibration_bytes]
                    docs.append(raw)
                    calibration_docs.add(chunk.document)
                    samples.append({"document": chunk.document, "start": chunk.start, "bytes": len(raw),
                                    "sha256": hashlib.sha256(raw).hexdigest()})
                if len(docs) == args.calibration_chunks:
                    break
            with (args.supply / f"{args.split}.idx").open(encoding="utf-8") as handle:
                evaluation_docs = {json.loads(line)["sha256"] for line in handle}
            if calibration_docs & evaluation_docs:
                raise ValueError("calibration documents overlap the evaluation split")
            calibration = calibrate(bundle.model, docs, args.target_p_in, args.target_p_out, table=table, strength=args.strength)
            report["calibration"] = calibration
            report["calibration_samples"] = samples
            beta_in, beta_out = calibration["beta_in"], calibration["beta_out"]
        report["beta_in"], report["beta_out"] = beta_in, beta_out
        report["changed_tensors"] = audit(bundle.model, reference)
        with RouterProgram(bundle.model, table, beta_in=beta_in, beta_out=beta_out, strength=args.strength):
            report["evaluation"] = evaluate_edited(bundle, iter_eval_chunks(args.supply, args.split, args.window_bytes))
    finally:
        restore(bundle.model, reference)
    report["changes_after_restore"] = audit(bundle.model, reference)
    if report["changes_after_restore"]:
        raise RuntimeError("model was not restored exactly")
    atomic_json(args.output, report)
    print(f"edited folded BPB: {report['evaluation']['bpb']:.6f}; model restored exactly")


if __name__ == "__main__":
    main()
