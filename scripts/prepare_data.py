from __future__ import annotations

import argparse
from pathlib import Path

from blt.data import prepare_supply


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an immutable byte supply from local parquet files")
    parser.add_argument("source", type=Path, help="parquet file or directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-bytes", type=int, default=10_000_000_000)
    parser.add_argument("--val-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--test-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sources = [args.source] if args.source.is_file() else sorted(args.source.rglob("*.parquet"))
    print(prepare_supply(sources, args.output, seed=args.seed, train_bytes=args.train_bytes,
                         val_bytes=args.val_bytes, test_bytes=args.test_bytes))


if __name__ == "__main__":
    main()
