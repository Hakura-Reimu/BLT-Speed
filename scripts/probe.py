from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from blt.atomic import atomic_json
from blt.data import verify_supply
from blt.eval_io import load_evaluation_model
from blt.model import BPELM, ByteLatentLM
from blt.probe import (ExtractionBudget, cache_fingerprint, collect_features, fit_probes, load_features, perturbation_analysis,
                       prepare_examples, probe_criteria, save_features)
from blt.train import make_model


def _training_seed(run: Path) -> int | None:
    frozen = json.loads((run / "config.json").read_text(encoding="utf-8"))
    return frozen.get("training", {}).get("init_seed")


def _untrained(config, seed: int, device: str):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = make_model(config)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit held-out, causally aligned representation probes")
    parser.add_argument("--byte-run", type=Path, required=True)
    parser.add_argument("--bpe-run", type=Path, required=True)
    parser.add_argument("--byte-checkpoint", type=Path)
    parser.add_argument("--bpe-checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--supply", type=Path)
    parser.add_argument("--examples", type=Path, required=True, help="annotated numeric and suffix examples as JSONL")
    parser.add_argument("--robustness-text", type=Path, help="held-out text for repository robustness pairs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-cache", type=Path, help="reuse or create identity-checked NPZ features")
    parser.add_argument("--replicate", type=Path, help="result JSON from the independently trained seed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--control-seed", type=int, default=9173)
    parser.add_argument("--test-fraction", type=float, default=.25)
    parser.add_argument("--max-prefix-bytes", type=int, default=512)
    parser.add_argument("--robustness-limit", type=int, default=1000)
    parser.add_argument("--extraction-budget-seconds", type=float, default=3600,
                        help="conservative wall-time cap for model calls, at most one hour")
    parser.add_argument("--allow-forced-patch", action="store_true",
                        help="permit an explicitly flagged analysis-only closure at an otherwise aligned anchor")
    args = parser.parse_args()
    if (args.max_prefix_bytes <= 0 or args.robustness_limit <= 0 or not 0 < args.test_fraction < 1
            or not 0 < args.extraction_budget_seconds <= 3600):
        parser.error("prefix length, robustness limit and test fraction must be valid")
    with args.examples.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    examples = prepare_examples(records)
    if not examples:
        parser.error("examples are empty")
    if args.supply is not None:
        verify_supply(args.supply)
    byte = load_evaluation_model(run=args.byte_run, checkpoint=args.byte_checkpoint,
                                 device=args.device, supply=args.supply)
    bpe = load_evaluation_model(run=args.bpe_run, checkpoint=args.bpe_checkpoint, tokenizer_path=args.tokenizer,
                                device=args.device, supply=args.supply)
    if not isinstance(byte.model, ByteLatentLM) or not isinstance(bpe.model, BPELM):
        parser.error("--byte-run and --bpe-run must name the corresponding model families")
    if byte.identity.get("data_manifest_sha256") != bpe.identity.get("data_manifest_sha256"):
        parser.error("checkpoints must use the same data manifest")
    identities = {"byte": byte.identity, "bpe": bpe.identity}
    budget = ExtractionBudget(args.extraction_budget_seconds)
    fingerprint = cache_fingerprint(examples, identities, seed=args.control_seed, forced=args.allow_forced_patch,
                                    max_prefix_bytes=args.max_prefix_bytes)
    if args.state_cache and args.state_cache.exists():
        kept, features, alignment = load_features(args.state_cache, fingerprint, examples)
    else:
        untrained_byte = _untrained(byte.config, args.control_seed, args.device)
        untrained_bpe = _untrained(bpe.config, args.control_seed + 1, args.device)
        kept, features, alignment = collect_features(examples, byte.model, bpe.model, untrained_byte, untrained_bpe,
                                                     bpe.tokenizer, allow_forced_patch=args.allow_forced_patch,
                                                     max_prefix_bytes=args.max_prefix_bytes, budget=budget)
        del untrained_byte, untrained_bpe
        if args.state_cache:
            args.state_cache.parent.mkdir(parents=True, exist_ok=True)
            save_features(args.state_cache, fingerprint, features, alignment)
    probes, fitted = fit_probes(kept, features, seed=args.split_seed, test_fraction=args.test_fraction)
    corpus = (args.robustness_text.read_text(encoding="utf-8") if args.robustness_text
              else "\n".join(dict.fromkeys(row["text"] for row in kept)))
    robustness = perturbation_analysis(byte.model, bpe.model, bpe.tokenizer, corpus, kept, probes, fitted,
                                       limit=args.robustness_limit, allow_forced_patch=args.allow_forced_patch,
                                       max_prefix_bytes=args.max_prefix_bytes, budget=budget)
    dataset_sha = hashlib.sha256(json.dumps(kept, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    output = {"type": "representation_probes_v1", "dataset_sha256": dataset_sha,
              "examples_source": str(args.examples), "checkpoints": identities,
              "model_names": {"byte": byte.config.name, "bpe": bpe.config.name},
              "training_seeds": {"byte": _training_seed(args.byte_run), "bpe": _training_seed(args.bpe_run)},
              "split_seed": args.split_seed, "test_fraction": args.test_fraction,
              "control_seed": args.control_seed, "alignment": alignment,
              "extraction_budget": {"limit_seconds": budget.limit, "elapsed_model_call_seconds": budget.elapsed,
                                    "policy": "conservative wall time; stops between model calls"},
              "state_definition": {"byte": "projected completed-patch global path only; local decoder excluded",
                                   "bpe": "token-end state on the identical causal byte prefix",
                                   "layers": "global input, middle block output, final normalized output",
                                   "forced_patch_closure": "analysis-only; individually flagged"},
              "probes": probes, "robustness": robustness}
    replicate = json.loads(args.replicate.read_text(encoding="utf-8")) if args.replicate else None
    output["replicate_source"] = str(args.replicate) if args.replicate else None
    output["criteria"] = probe_criteria(output, replicate)
    atomic_json(args.output, output)
    print(f"Wrote {args.output}; representation verdict: {output['criteria']['overall']['status']}.")


if __name__ == "__main__":
    main()
