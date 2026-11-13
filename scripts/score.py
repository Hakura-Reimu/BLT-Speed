from __future__ import annotations

import argparse
from pathlib import Path

from blt.atomic import atomic_json
from blt.data import verify_supply
from blt.eval_io import load_evaluation_model
from blt.score import score_supply


def main() -> None:
    parser = argparse.ArgumentParser(description="Score document-disjoint Track B chunks in bits per byte")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--supply", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-bytes", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    verify_supply(args.supply)
    bundle = load_evaluation_model(run=args.run, checkpoint=args.checkpoint, tokenizer_path=args.tokenizer,
                                   device=args.device, supply=args.supply)
    result = score_supply(bundle.model, str(args.supply), split=args.split, window_bytes=args.window_bytes,
                          tokenizer=bundle.tokenizer, device=bundle.device, checkpoint=bundle.identity)
    atomic_json(args.output, result)
    print(f"{result['split']}: {result['bpb']:.6f} BPB over {result['bytes']} bytes")


if __name__ == "__main__":
    main()
