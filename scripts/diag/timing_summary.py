from __future__ import annotations

"""Check and summarise scripts/final_timing.sh outputs (protocol of Sol rounds 9-10 and 14).

  check FILE --kib K... --warmup W --repetitions R --runs RUN [RUN]
      Exit 0 if every requested cell is present with the expected settings, models, trial counts and identical
      prompts for both models, or is a memory-budget error (deterministic, kept as a result); otherwise print the
      problems and exit 1.
  summary [RESULTS_DIR] [--variant V]
      Per pair (A = byte model, B = BPE model, whatever order the sweep used), cell and prefill variant: each sweep's
      A/B ratio of median bytes/s, and the three sweeps combined with a crossed bootstrap that resamples sweeps and
      prompts separately (the sweeps time the same prompts, so concatenating them would understate the uncertainty).
      A pair is complete only with three valid sweeps whose checkpoints, prompts and frozen cache sizes agree.
      Also: solo throughput per model, the anchor pair over time, the 2/4/32 KiB curve, and which byte points are
      dominated (a BPE point with lower 3 GB BPB whose combined throughput interval lies above the byte model's).
      Writes RESULTS_DIR/summary.json and prints Markdown for one variant (default end_to_end).
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from blt.atomic import atomic_json

VARIANTS = ("end_to_end", "gpu_resident", "gpu_resident_preprocessed")
SWEEPS = ("s1", "s2", "s3")
SAMPLES = 2000
SCORES = Path.home() / "blt/results/main-scores"


def _run(identity: dict) -> str:
    """Run name of a checkpoint identity: .../runs/<run>-seed0/ckpt/bytes-3000000000.pt -> <run>."""
    return Path(identity["path"]).parents[1].name.removesuffix("-seed0")


def problems(path: Path, kibs: list[int], warmup: int, repetitions: int, runs: list[str]) -> tuple[list[str], list[str]]:
    """Fatal problems and permanent (memory-budget) cell errors of one bench_infer output."""
    try:
        results = json.loads(path.read_text(encoding="utf-8"))["results"]
    except (OSError, ValueError, KeyError) as error:
        return [f"unreadable: {error}"], []
    fatal, permanent = [], []
    for kib in kibs:
        cells = [r for r in results if r.get("prompt_kib") == kib and r.get("batch") == 1 and r.get("mode") == "prefill"]
        if len(cells) != 1:
            fatal.append(f"{kib} KiB: {len(cells)} results")
            continue
        cell = cells[0]
        if "error" in cell:
            (permanent if cell["error"].startswith("reserved memory") else fatal).append(f"{kib} KiB: {cell['error']}")
            continue
        if (cell.get("execution"), cell.get("warmup"), cell.get("repetitions")) != ("graph", warmup, repetitions):
            fatal.append(f"{kib} KiB: settings {cell.get('execution')}, {cell.get('warmup')}, {cell.get('repetitions')}")
        if cell.get("numerics", {}).get("bf16_reduced_precision_reduction") is not False:
            fatal.append(f"{kib} KiB: reduced-precision reductions not off")
        if set(cell.get("variants", {})) != set(VARIANTS):
            fatal.append(f"{kib} KiB: variants {sorted(cell.get('variants', {}))}")
            continue
        for variant, value in cell["variants"].items():
            models = value["models"]
            if [_run(m["identity"]) for m in models] != runs:
                fatal.append(f"{kib} KiB {variant}: models {[_run(m['identity']) for m in models]}")
                continue
            if any(not m["identity"]["path"].endswith("ckpt/bytes-3000000000.pt") for m in models):
                fatal.append(f"{kib} KiB {variant}: not the 3 GB checkpoints")
            for m in models:
                if {len(m[key]) for key in ("seconds", "bytes", "graph_hits", "prompt_sha256")} != {repetitions}:
                    fatal.append(f"{kib} KiB {variant} {_run(m['identity'])}: trial counts")
            if len(models) == 2 and models[0]["prompt_sha256"] != models[1]["prompt_sha256"]:
                fatal.append(f"{kib} KiB {variant}: the two models timed different prompts")
    return fatal, permanent


def _speeds(model: dict) -> np.ndarray:
    return np.asarray(model["bytes"], dtype=float) / np.asarray(model["seconds"], dtype=float)


def _crossed(a: np.ndarray, b: np.ndarray | None = None, seed: int = 0) -> dict:
    """Geometric mean over sweeps of each sweep's median of a (or ratio of medians a/b), from [sweep, prompt] arrays,
    with a sweep-and-prompt crossed bootstrap. Within-sweep ratios keep the back-to-back pairing; medians pooled over
    sweeps can fall outside every sweep's ratio when the sweeps differ in overall speed."""
    def stat(x: np.ndarray, y: np.ndarray | None) -> float:
        per_sweep = np.median(x, axis=1) if y is None else np.median(x, axis=1) / np.median(y, axis=1)
        return float(np.exp(np.mean(np.log(per_sweep))))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(SAMPLES):
        index = np.ix_(rng.integers(0, a.shape[0], a.shape[0]), rng.integers(0, a.shape[1], a.shape[1]))
        draws.append(stat(a[index], None if b is None else b[index]))
    return {"value": float(stat(a, b)), "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)), "sweeps": a.shape[0]}


def _cells(path: Path) -> dict[tuple[int, str], dict]:
    cells = {}
    for result in json.loads(path.read_text(encoding="utf-8"))["results"]:
        if "error" in result:
            cells[(result["prompt_kib"], "error")] = {"error": result["error"]}
            continue
        for variant, value in result["variants"].items():
            cells[(result["prompt_kib"], variant)] = {**value, "cache_size": result.get("graph_cache_policy", {}).get("size")}
    return cells


def _combine(records: dict[str, dict], order: list[str]) -> dict:
    """records: sweep -> variant record of one cell; order: runs as [A] or [A, B]."""
    notes, sweeps = [], []
    rows = {}
    for sweep, value in records.items():
        models = {_run(m["identity"]): m for m in value["models"]}
        rows[sweep] = [models[run] for run in order]
        a = rows[sweep][0]
        entry = {"sweep": sweep, "cache_size": value["cache_size"], "peak_reserved_bytes":
                 value["peak_reserved_bytes_whole_variant"], "bytes_per_second": [m["bytes_per_second"] for m in rows[sweep]],
                 "p95_seconds": [m["p95_seconds"] for m in rows[sweep]],
                 "miss_rate": [m["graph_cache"]["miss_rate"] for m in rows[sweep]]}
        if len(order) == 2:
            b = rows[sweep][1]
            entry["ratio"] = float(np.median(_speeds(a)) / np.median(_speeds(b)))
            hits = np.asarray(a["graph_hits"], dtype=bool) & np.asarray(b["graph_hits"], dtype=bool)
            entry["both_hit_trials"] = int(hits.sum())
            entry["both_hit_ratio"] = (float(np.median(_speeds(a)[hits]) / np.median(_speeds(b)[hits]))
                                       if hits.any() else None)
        sweeps.append(entry)
    first = next(iter(rows.values()))
    for sweep, models in rows.items():
        for i, run in enumerate(order):
            if models[i]["identity"]["sha256"] != first[i]["identity"]["sha256"]:
                notes.append(f"{run}: checkpoint differs in {sweep}")
            if models[i]["prompt_sha256"] != first[i]["prompt_sha256"]:
                notes.append(f"{run}: prompts differ in {sweep}")
    if len({entry["cache_size"] for entry in sweeps}) > 1:
        notes.append("frozen cache size differs between sweeps")
    out = {"sweeps": sweeps, "notes": notes, "complete": len(records) == len(SWEEPS) and not notes}
    if not any("prompts differ" in note for note in notes):
        a = np.stack([_speeds(rows[s][0]) for s in rows])
        out["combined"] = _crossed(a, np.stack([_speeds(rows[s][1]) for s in rows]) if len(order) == 2 else None)
    return out


def _bpb(run: str) -> float | None:
    path = SCORES / f"{run}-seed0-bytes-3000000000.json"
    return json.loads(path.read_text(encoding="utf-8"))["bpb"] if path.exists() else None


def summarise(root: Path) -> dict:
    summary: dict = {"pairs": {}, "solo": {}, "anchor": {}, "curve": {}, "errors": {}, "dominated": {}}
    names = sorted({p.stem for s in SWEEPS for p in (root / s).glob("*.json")})
    for name in names:
        group = "solo" if name.startswith("solo-") else "pairs"
        order = [name.removeprefix("solo-")] if group == "solo" else name.split("--")
        per_sweep = {s: _cells(root / s / f"{name}.json") for s in SWEEPS if (root / s / f"{name}.json").exists()}
        keys = sorted({key for cells in per_sweep.values() for key in cells})
        for kib, variant in keys:
            records = {s: cells[(kib, variant)] for s, cells in per_sweep.items() if (kib, variant) in cells}
            if variant == "error":
                summary["errors"][f"{name}/{kib}"] = {s: r["error"] for s, r in records.items()}
                continue
            summary[group].setdefault(name, {})[f"{kib}/{variant}"] = _combine(records, order)
    for path in sorted((root / "anchor").glob("*.json")) + sorted((root / "curve").glob("*.json")):
        group = path.parent.name
        order = ["s-p6", "s-bpe50k"] if group == "anchor" else path.stem.split("--")
        for (kib, variant), value in _cells(path).items():
            if variant in ("end_to_end", "gpu_resident_preprocessed"):
                models = {_run(m["identity"]): m for m in value["models"]}
                summary[group].setdefault(path.stem, {})[f"{kib}/{variant}"] = {
                    "ratio": float(np.median(_speeds(models[order[0]])) / np.median(_speeds(models[order[1]]))),
                    "prompt_bootstrap": value["paired_bootstrap"] if order == [_run(m["identity"]) for m in value["models"]]
                    else None}
    for name, cells in summary["pairs"].items():
        a, b = name.split("--")
        bpb_a, bpb_b = _bpb(a), _bpb(b)
        for cell, value in cells.items():
            combined = value.get("combined")
            if "bpe" in b and bpb_a and bpb_b and bpb_b < bpb_a and combined and combined["ci95_high"] < 1:
                summary["dominated"].setdefault(a, {}).setdefault(cell, []).append(
                    {"by": b, "bpb_gap": bpb_a - bpb_b, "complete": value["complete"], **combined})
    return summary


def markdown(summary: dict, variant: str) -> str:
    lines = [f"## Pairs, {variant} (A/B speed; above 1 means the byte model A is faster)", "",
             "| A--B | KiB | complete | combined A/B [95% CI] | per sweep | A, B kB/s | miss A, B | both-hit A/B | cache |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, cells in summary["pairs"].items():
        for cell, value in cells.items():
            kib, v = cell.split("/")
            if v != variant:
                continue
            c, s = value.get("combined"), value["sweeps"]
            lines.append(
                f"| {name} | {kib} | {'yes' if value['complete'] else 'NO ' + '; '.join(value['notes'])} | "
                + (f"{c['value']:.3f} [{c['ci95_low']:.3f}, {c['ci95_high']:.3f}]" if c else "-") + " | "
                + ", ".join(f"{x['ratio']:.3f}" for x in s) + " | "
                + f"{np.median([x['bytes_per_second'][0] for x in s]) / 1e3:.0f}, "
                + f"{np.median([x['bytes_per_second'][1] for x in s]) / 1e3:.0f} | "
                + f"{max(x['miss_rate'][0] or 0 for x in s):.2f}, {max(x['miss_rate'][1] or 0 for x in s):.2f} | "
                + ", ".join("-" if x["both_hit_ratio"] is None else f"{x['both_hit_ratio']:.3f}" for x in s) + " | "
                + ",".join(str(x["cache_size"]) for x in s) + " |")
    lines += ["", f"## Solo throughput, {variant} (median kB/s)", "", "| model | KiB | complete | combined [95% CI] | per sweep |",
              "|---|---|---|---|---|"]
    for name, cells in summary["solo"].items():
        for cell, value in cells.items():
            kib, v = cell.split("/")
            if v == variant and value.get("combined"):
                c = value["combined"]
                lines.append(f"| {name.removeprefix('solo-')} | {kib} | {'yes' if value['complete'] else 'NO'} | "
                             f"{c['value'] / 1e3:.0f} [{c['ci95_low'] / 1e3:.0f}, {c['ci95_high'] / 1e3:.0f}] | "
                             + ", ".join(f"{x['bytes_per_second'][0] / 1e3:.0f}" for x in value["sweeps"]) + " |")
    lines += ["", "## Anchor S-P6/S-BPE50K over the run", ""]
    for name, cells in summary["anchor"].items():
        lines.append(f"- {name}: " + ", ".join(f"{cell} {value['ratio']:.3f}" for cell, value in cells.items()))
    lines += ["", "## Curve (one sweep)", ""]
    for name, cells in summary["curve"].items():
        lines.append(f"- {name}: " + ", ".join(f"{cell} {value['ratio']:.3f}" for cell, value in cells.items()
                                               if cell.endswith(variant)))
    lines += ["", "## Dominated byte points (lower-BPB BPE with combined A/B interval below 1)", ""]
    for name, cells in summary["dominated"].items():
        for cell, entries in cells.items():
            if cell.endswith(variant):
                lines.append(f"- {name} {cell}: " + ", ".join(
                    f"{e['by']} (BPB {e['bpb_gap']:+.3f}, A/B {e['value']:.3f} [{e['ci95_low']:.3f}, {e['ci95_high']:.3f}]"
                    f"{'' if e['complete'] else ', incomplete'})" for e in entries))
    if summary["errors"]:
        lines += ["", "## Cell errors", ""] + [f"- {k}: {v}" for k, v in summary["errors"].items()]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("file", type=Path)
    check.add_argument("--kib", type=int, nargs="+", required=True)
    check.add_argument("--warmup", type=int, required=True)
    check.add_argument("--repetitions", type=int, required=True)
    check.add_argument("--runs", nargs="+", required=True)
    summ = sub.add_parser("summary")
    summ.add_argument("root", type=Path, nargs="?", default=Path.home() / "blt/results/final-timing")
    summ.add_argument("--variant", choices=VARIANTS, default="end_to_end")
    args = parser.parse_args()
    if args.command == "check":
        fatal, permanent = problems(args.file, args.kib, args.warmup, args.repetitions, args.runs)
        for line in fatal + [f"kept (memory budget): {p}" for p in permanent]:
            print(line)
        sys.exit(1 if fatal else 0)
    summary = summarise(args.root)
    atomic_json(args.root / "summary.json", summary)
    print(markdown(summary, args.variant))


if __name__ == "__main__":
    main()
