from __future__ import annotations

"""Exactness and speed of a faster GPT-2 tokenizer for the 50K baselines (GPT-6.1 Sol round 9, item 4a).

tiktoken is built from the local GPT-2 merges and vocabulary (no download). It may replace HF `tokenizers` in timed
preprocessing only if token IDs are identical on every 8 KiB window of the full validation split and on adversarial
Unicode, whitespace and literal-special-token cases. Speed is the median per-window encode time, single process.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

import tiktoken
from tokenizers import Tokenizer

from blt.atomic import atomic_json
from blt.data import iter_eval_chunks
from blt.fast_tokenizer import gpt2_tiktoken, hf_byte_level_tiktoken

ADVERSARIAL = [
    "", " ", "  ", "\n", "\n\n\n", "\t \t", "a  b   c    ", "trailing   \n", "\r\n\r\n", " nbsp ",
    "<|endoftext|>", "text<|endoftext|>more", " <|endoftext|> ", "<|endoftext", "|endoftext|>", "<|endoftext|><|endoftext|>",
    "<|bos|>", "a<|bos|>b<|eos|>c<|pad|>", " <|bos|> ",
    "It's we'll they're I'd you've", "IT'S WE'LL", "'s 's'", "don’t", "é café é", "क्ष",
    "中文字符 日本語 한국어", "مرحبا שלום",
    "\U0001f600\U0001f468‍\U0001f469‍\U0001f467 \U0001f1ec\U0001f1e7", "​‌‍﻿",
    "1234567890 3.14159 1,000,000 0x1F", "$$$ !!! ??? ... --- ___", "a\u0000b", "\U0010ffff",
    "mixed\tspaces 　ideographic", "line sep para", "    indented code\n        more",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer-dir", type=Path, default=Path.home() / "blt/models/gpt2-tokenizer")
    parser.add_argument("--tokenizer-json", type=Path, help="a non-GPT-2 HF byte-level BPE (e.g. the 16K tokenizer)")
    parser.add_argument("--supply", type=Path, default=Path.home() / "blt/data/supply-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-windows", type=int, default=300)
    args = parser.parse_args()
    hf = Tokenizer.from_file(str(args.tokenizer_json or args.tokenizer_dir / "tokenizer.json"))
    fast = hf_byte_level_tiktoken(args.tokenizer_json) if args.tokenizer_json else gpt2_tiktoken(args.tokenizer_dir)
    hf_encode = lambda text: hf.encode(text, add_special_tokens=False).ids
    modes = {"encode_ordinary": fast.encode_ordinary, "encode_all_special": lambda text: fast.encode(text, allowed_special="all")}
    mismatches = {mode: [] for mode in modes}
    for index, text in enumerate(ADVERSARIAL):
        reference = hf_encode(text)
        for mode, encode in modes.items():
            if encode(text) != reference:
                mismatches[mode].append({"case": index, "text": text, "hf": reference, "tiktoken": encode(text)})
    windows = [chunk.data.decode("utf-8") for chunk in iter_eval_chunks(str(args.supply), "val", 8192)]
    window_mismatch = {mode: 0 for mode in modes}
    for text in windows:
        reference = hf_encode(text)
        for mode, encode in modes.items():
            window_mismatch[mode] += encode(text) != reference
    timing = {}
    sample = windows[:args.timing_windows]
    for name, encode in (("hf", hf_encode), *modes.items()):
        for text in sample[:20]:
            encode(text)
        seconds = []
        for text in sample:
            started = time.perf_counter()
            encode(text)
            seconds.append(time.perf_counter() - started)
        timing[name] = {"median_ms_per_window": 1e3 * statistics.median(seconds),
                        "mean_ms_per_window": 1e3 * statistics.fmean(seconds)}
    exact = {mode: not mismatches[mode] and not window_mismatch[mode] for mode in modes}
    result = {"windows": len(windows), "adversarial_cases": len(ADVERSARIAL), "adversarial_mismatches": mismatches,
              "window_mismatches": window_mismatch, "exact": exact, "timing": timing,
              "tiktoken_version": tiktoken.__version__}
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "adversarial_mismatches"}, indent=2))
    for mode, rows in mismatches.items():
        for row in rows[:5]:
            print(mode, "mismatch:", repr(row["text"]), row["hf"][:12], row["tiktoken"][:12])


if __name__ == "__main__":
    main()
