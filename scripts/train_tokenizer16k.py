from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from blt.data import WindowIndex, sample_window_index, sha256_file, verify_supply


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the shared GPT-2-style byte BPE tokenizer")
    parser.add_argument("supply", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sample-bytes", type=int, default=200_000_000)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.sample_bytes <= 0 or args.vocab_size < 259:
        parser.error("sample size must be positive and vocabulary must fit the byte alphabet and specials")
    verify_supply(args.supply)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=["<|bos|>", "<|eos|>", "<|pad|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
    sample_hash = hashlib.sha256()
    used = 0
    selected = 0
    with WindowIndex(args.supply / "train.bin", drop_short_tail=False) as windows:
        def samples():
            nonlocal used, selected
            for offset in range(len(windows)):
                index = sample_window_index(args.seed, offset, 0, 1, len(windows))
                raw = windows[index].data
                sample_hash.update(len(raw).to_bytes(8, "little"))
                sample_hash.update(raw)
                used += len(raw)
                selected += 1
                yield raw.decode("utf-8")
                if used >= args.sample_bytes:
                    return
            raise ValueError("training supply is smaller than the tokenizer sample")

        tokenizer.train_from_iterator(samples(), trainer=trainer)
    with tempfile.TemporaryDirectory(prefix=".blt-tokenizer-", dir=args.output.parent) as directory:
        work = Path(directory)
        path = work / "tokenizer.json"
        tokenizer.save(str(path))
        metadata = {"vocab_size": tokenizer.get_vocab_size(), "bos_id": tokenizer.token_to_id("<|bos|>"),
                    "seed": args.seed, "sample_bytes": used, "sample_windows": selected,
                    "sample_sha256": sample_hash.hexdigest(), "tokenizer_sha256": sha256_file(path),
                    "train_manifest_sha256": sha256_file(args.supply / "manifest.json"),
                    "pretokenizer": "GPT-2 ByteLevel regex; no normalization or prefix space"}
        (work / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.rename(work, args.output)


if __name__ == "__main__":
    main()
