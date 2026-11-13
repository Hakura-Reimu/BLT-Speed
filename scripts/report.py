from __future__ import annotations

import argparse
import json
from pathlib import Path

from blt.report import measure_retrieval_cache, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize measured campaign JSON without filling missing results")
    parser.add_argument("--results", nargs="*", type=Path, default=[], help="result JSON files or directories")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selection", type=Path, help="frozen validation selection and source-labelled extra evidence JSON")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--endpoint-bytes", type=int, default=3_000_000_000)
    parser.add_argument("--prefill-batch", type=int, default=1)
    parser.add_argument("--measure-retrieval-cache", type=Path, help="annotate one existing retrieval JSON with full-prompt cache measurements")
    parser.add_argument("--run", type=Path, help="run directory for retrieval cache measurement")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint used for the recorded retrieval result")
    parser.add_argument("--tokenizer", type=Path, help="tokenizer for a BPE retrieval model")
    parser.add_argument("--supply", type=Path, help="same verified supply used by the retrieval evaluation")
    parser.add_argument("--cache-output", type=Path, help="new annotated retrieval JSON")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.measure_retrieval_cache:
        if args.run is None or args.supply is None or args.cache_output is None:
            parser.error("cache measurement requires --run, --supply and --cache-output")
        if args.cache_output.resolve() == args.measure_retrieval_cache.resolve():
            parser.error("--cache-output must differ from the original retrieval JSON")
        from blt.atomic import atomic_json
        from blt.eval_io import load_evaluation_model
        original = json.loads(args.measure_retrieval_cache.read_text(encoding="utf-8"))
        bundle = load_evaluation_model(run=args.run, checkpoint=args.checkpoint, tokenizer_path=args.tokenizer,
                                       supply=args.supply, device=args.device)
        annotated = measure_retrieval_cache(original, bundle, args.supply)
        atomic_json(args.cache_output, annotated)
        print(f"Wrote retrieval cache measurements to {args.cache_output}.")
        return
    if args.output_dir is None:
        parser.error("--output-dir is required for report generation")
    selection = json.loads(args.selection.read_text(encoding="utf-8")) if args.selection else {}
    summary = write_report(args.results, args.output_dir, selection=selection, split=args.split,
                           endpoint=args.endpoint_bytes, batch=args.prefill_batch)
    print(f"Wrote {args.output_dir / 'report.md'} with {len(summary['points'])} measured frontier points.")


if __name__ == "__main__":
    main()
