from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
from tokenizers import Tokenizer

from blt.token_cache import build_token_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-tokenize all fixed training windows")
    parser.add_argument("supply", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("--window-bytes", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--sample-seed", type=int, default=0)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    print(build_token_cache(args.supply, args.tokenizer, tokenizer, window_bytes=args.window_bytes,
                            batch_size=args.batch_size, sample_size=args.sample_size, sample_seed=args.sample_seed))


if __name__ == "__main__":
    main()
