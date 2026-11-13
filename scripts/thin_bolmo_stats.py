from __future__ import annotations

import argparse
from pathlib import Path

from blt.atomic import atomic_json
from blt.data import WindowIndex, sample_window_index, sha256_file, verify_supply
from blt.thin_bolmo import importance_stats
from blt.trackA import load_reference


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect local importance statistics for manual Thin-Bolmo selection")
    parser.add_argument("--bolmo", type=Path, required=True)
    parser.add_argument("--bolmo-subword-tokenizer", type=Path)
    parser.add_argument("--supply", type=Path, required=True)
    parser.add_argument("--window-bytes", type=int, choices=(2048, 4096), default=2048)
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.windows <= 0:
        parser.error("windows must be positive")
    verify_supply(args.supply)
    bundle = load_reference("bolmo", args.bolmo,
                            bolmo_subword_tokenizer=str(args.bolmo_subword_tokenizer) if args.bolmo_subword_tokenizer else None)
    with WindowIndex(args.supply / "train.bin", args.window_bytes) as windows:
        count = min(args.windows, len(windows))
        indices = [sample_window_index(args.seed, i, 0, 1, len(windows)) for i in range(count)]
        result = importance_stats(bundle.model, (windows[index] for index in indices))
        result.update({"teacher_identity": bundle.identity, "data_manifest_hash": sha256_file(args.supply / "manifest.json"),
                       "window_index_sha256": windows.index_sha256, "window_indices": indices, "data_seed": args.seed})
        atomic_json(args.output, result)
    print(f"Wrote statistics for {count} training windows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
