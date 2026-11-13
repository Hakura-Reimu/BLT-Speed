from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from blt.atomic import atomic_json
from blt.data import iter_eval_chunks, verify_supply
from blt.trackA import load_reference, score_reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Bolmo and OLMo on the same Track B byte chunks")
    parser.add_argument("--bolmo", type=Path, required=True)
    parser.add_argument("--olmo", type=Path, required=True)
    parser.add_argument("--bolmo-subword-tokenizer", type=Path)
    parser.add_argument("--supply", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--window-bytes", type=int, default=8192)
    parser.add_argument("--domains", type=Path, help="JSON object mapping document SHA-256 to domain")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.window_bytes <= 8192:
        parser.error("window-bytes must be between 1 and 8192")
    verify_supply(args.supply)
    domains = json.loads(args.domains.read_text(encoding="utf-8")) if args.domains else None
    if domains is not None and (not isinstance(domains, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in domains.items())):
        parser.error("domains must be a JSON object mapping document hashes to domain names")
    output = {"split": args.split, "window_bytes": args.window_bytes, "models": {}}
    for family, path in (("bolmo", args.bolmo), ("olmo", args.olmo)):
        bundle = load_reference(family, path, bolmo_subword_tokenizer=str(args.bolmo_subword_tokenizer) if args.bolmo_subword_tokenizer else None)
        result = score_reference(bundle, iter_eval_chunks(args.supply, args.split, args.window_bytes), split=args.split, domains=domains)
        output["models"][family] = result
        atomic_json(args.output, output)
        print(f"{family}: {result['bpb']:.6f} BPB over {result['bytes']} bytes", flush=True)
        del bundle
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
