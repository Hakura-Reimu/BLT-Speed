from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import torch

from blt.atomic import atomic_json
from blt.data import iter_eval_chunks, sha256_file, verify_supply
from blt.thin_bolmo import dense_local_cost, evaluate_pair, load_student_checkpoint
from blt.trackA import ReferenceModel, load_reference


def aggregate(rows: list[dict], layout: str) -> dict:
    count = sum(row[layout]["bytes"] for row in rows)
    teacher_nats = sum(row[layout]["teacher_nats"] for row in rows)
    student_nats = sum(row[layout]["student_nats"] for row in rows)
    teacher_patches = sum(row[layout]["teacher_patches"] for row in rows)
    student_patches = sum(row[layout]["student_patches"] for row in rows)
    return {"bytes": count, "teacher_nats": teacher_nats, "student_nats": student_nats,
            "teacher_bpb": teacher_nats / (math.log(2) * count),
            "student_bpb": student_nats / (math.log(2) * count),
            "teacher_patches": teacher_patches, "student_patches": student_patches,
            "teacher_bytes_per_patch": count / teacher_patches, "student_bytes_per_patch": count / student_patches}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare a Thin-Bolmo student under teacher and student routing")
    parser.add_argument("--bolmo", type=Path, required=True)
    parser.add_argument("--bolmo-subword-tokenizer", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True, help="distillation delta .pt")
    parser.add_argument("--supply", type=Path, required=True)
    parser.add_argument("--window-bytes", type=int, choices=(2048, 4096), default=2048)
    parser.add_argument("--val-windows", type=int, default=16, help="0 evaluates all validation chunks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true", help="also time student stock generation via Track A")
    parser.add_argument("--benchmark-warmup", type=int, default=10)
    parser.add_argument("--benchmark-repetitions", type=int, default=30)
    parser.add_argument("--benchmark-output-bytes", type=int, default=64)
    args = parser.parse_args()
    if args.val_windows < 0 or args.benchmark_warmup < 0 or args.benchmark_repetitions <= 0:
        parser.error("invalid validation or benchmark count")
    verify_supply(args.supply)
    manifest_hash = sha256_file(args.supply / "manifest.json")
    bundle = load_reference("bolmo", args.bolmo,
                            bolmo_subword_tokenizer=str(args.bolmo_subword_tokenizer) if args.bolmo_subword_tokenizer else None)
    student, checkpoint = load_student_checkpoint(bundle.model, args.checkpoint, bundle.identity)
    if checkpoint["data_manifest_hash"] != manifest_hash:
        raise ValueError("evaluation supply does not match the training supply")
    chunks = iter_eval_chunks(args.supply, "val", args.window_bytes)
    if args.val_windows:
        chunks = itertools.islice(chunks, args.val_windows)
    rows, prompts = [], []
    for chunk in chunks:
        common = evaluate_pair(bundle.model, student, chunk.data)
        own = evaluate_pair(bundle.model, student, chunk.data, own_routing=True)
        rows.append({"document": chunk.document, "start": chunk.start, "end": chunk.end,
                     "teacher_layout": common, "student_layout": own})
        if len(chunk.data) >= 2:
            prompts.append(chunk.data)
    if not rows:
        raise ValueError("validation split is empty")
    result = {"metric": "folded BPB under the named Bolmo prefill layout; special mass retained",
              "routing_note": "prefill has one-byte lookahead; P includes the initial BOS patch",
              "checkpoint": checkpoint, "data_manifest_hash": manifest_hash, "window_bytes": args.window_bytes,
              "dense_local_arithmetic_estimate": dense_local_cost(student),
              "chunks": rows, "teacher_layout": aggregate(rows, "teacher_layout"),
              "student_layout": aggregate(rows, "student_layout")}
    # Preserve quality results if an optional generation benchmark fails.
    atomic_json(args.output, result)
    if args.benchmark:
        from blt.trackA_bench import benchmark_bolmo
        if not prompts:
            raise ValueError("benchmark needs a validation prompt with at least two bytes")
        trials = args.benchmark_warmup + args.benchmark_repetitions
        student_bundle = ReferenceModel("bolmo", student, bundle.tokenizer,
                                        {"teacher": bundle.identity, "student_checkpoint": str(args.checkpoint)},
                                        bundle.device, bundle.bos_id)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result["student_benchmark"] = benchmark_bolmo(student_bundle, list(itertools.islice(itertools.cycle(prompts), trials)),
                                                           batch=1, output_bytes=args.benchmark_output_bytes,
                                                           warmup=args.benchmark_warmup, repetitions=args.benchmark_repetitions,
                                                           forced_p=None)
        atomic_json(args.output, result)
    for name in ("teacher_layout", "student_layout"):
        item = result[name]
        print(f"{name}: teacher={item['teacher_bpb']:.6f}, student={item['student_bpb']:.6f} BPB; "
              f"student P={item['student_bytes_per_patch']:.3f}", flush=True)


if __name__ == "__main__":
    main()
