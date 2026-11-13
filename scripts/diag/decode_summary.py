from __future__ import annotations

"""Validate decode outputs and combine three paired sessions.

  check FILE --runs A B --batches 1 8 --kib 8 --repetitions 30
  summary ROOT

ROOT contains s1/, s2/, s3/; each may contain one or more JSON files.
All sessions must time the same ordered prompt batches and checkpoints.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys

import numpy as np

if __package__:
    from .timing_summary import _crossed
else:  # Direct execution, including isolated Python distributions.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from timing_summary import _crossed


SESSIONS = ("s1", "s2", "s3")
POLL_INTERVAL = 8
MODES = {"eager", "device", "device-conditional", "graph", "graph-split"}


def _run(identity: dict) -> str:
    path = PurePosixPath(identity["path"].replace("\\", "/"))
    return path.parent.parent.name.removesuffix("-seed0")


def _hex_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


TIE_ULPS = 4  # must match scripts/bench_decode.py


def _bf16_ulp(value: float) -> float:
    magnitude = abs(value)
    if not math.isfinite(magnitude):
        return math.inf
    return 2.0 ** -133 if magnitude == 0 else max(2.0 ** (math.floor(math.log2(magnitude)) - 7), 2.0 ** -133)


def _valid_tie(record: object, batch: int) -> bool:
    """Recompute the tie tolerance and check both margins; token ids are checked when recorded (from f32dc9a+1)."""
    if not isinstance(record, dict):
        return False
    numbers = [record.get(key) for key in ("reference_margin", "tested_margin", "tolerance", "reference_top_logit")]
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) for value in numbers):
        return False
    reference_margin, tested_margin, tolerance, top = (float(value) for value in numbers)
    if abs(tolerance - TIE_ULPS * _bf16_ulp(top)) > 1e-12 * max(1.0, tolerance):
        return False
    if not (0 <= reference_margin <= tolerance and 0 <= tested_margin <= tolerance):
        return False
    row, step = record.get("row"), record.get("step")
    if not isinstance(row, int) or not 0 <= row < batch or not isinstance(step, int) or step < 0:
        return False
    if "expected" in record or "selected" in record:
        expected, selected = record.get("expected"), record.get("selected")
        if not isinstance(expected, int) or not isinstance(selected, int) or expected == selected or min(expected, selected) < 0:
            return False
    return True


def _batch_digest(row_hashes: list[str]) -> str:
    values = [bytes.fromhex(value) for value in row_hashes]
    return (values[0] if len(values) == 1 else hashlib.sha256(b"".join(values)).digest()).hex()


def _cell_problems(cell: dict, runs: list[str], batch: int, kib: int, repetitions: int,
                   *, output_bytes: int = 256, warmup: int = 10, calibration_requests: int = 10,
                   verify_bytes: int = 512, memory_budget: int = 10_500_000_000) -> list[str]:
    errors = []
    prefix = f"{kib} KiB batch {batch}"
    if "error" in cell:
        return [f"{prefix}: {cell['error']}"]
    try:
        if (cell["mode"], cell["batch"], cell["prompt_kib"], cell["repetitions"], cell["warmup"],
            cell["requested_output_bytes_per_row"], cell["completion_poll_interval_native_steps"]) != (
                "decode", batch, kib, repetitions, warmup, output_bytes, POLL_INTERVAL):
            errors.append(f"{prefix}: incorrect benchmark settings")
        if cell.get("decoder_variant") != "selected":
            errors.append(f"{prefix}: missing calibrated/fixed selected-mode result")
        if cell["verify_bytes_per_row"] < verify_bytes:
            errors.append(f"{prefix}: requested parity length below {verify_bytes} bytes per row")
        if (cell["memory_budget_bytes"] != memory_budget or
                cell["peak_reserved_bytes_whole_cell"] > memory_budget):
            errors.append(f"{prefix}: memory budget or peak is invalid")
        if cell["numerics"].get("bf16_reduced_precision_reduction") is not False:
            errors.append(f"{prefix}: BF16 reduced-precision reductions are not disabled")
        models = cell["variants"]["decode"]["models"]
        if len(models) != len(runs) or [_run(model["identity"]) for model in models] != [
                PurePosixPath(run.replace("\\", "/")).name.removesuffix("-seed0") for run in runs]:
            return errors + [f"{prefix}: checkpoint runs differ from requested runs"]
        cal = cell["calibration"]
        modes = cal["requested_modes"]
        if (cal["requests"] != calibration_requests or not modes or len(set(modes)) != len(modes) or
                not set(modes) <= MODES):
            errors.append(f"{prefix}: incorrect calibration settings")
        if len(cal["models"]) != len(models):
            return errors + [f"{prefix}: calibration model count"]
        warm_hashes = cell["warmup_prompt_sha256_per_row"]
        warm_sizes = cell["warmup_prompt_bytes_per_row"]
        if (len(warm_hashes) != warmup or any(len(group) != batch or len(set(group)) != batch or
                                               any(not _hex_digest(v) for v in group) for group in warm_hashes)):
            errors.append(f"{prefix}: warm-up prompt count")
        if len(warm_sizes) != warmup or any(len(group) != batch or any(v != kib * 1024 for v in group)
                                             for group in warm_sizes):
            errors.append(f"{prefix}: warm-up prompt lengths")
        for index, (model, model_cal) in enumerate(zip(models, cal["models"])):
            name = runs[index]
            if "error" in model:
                errors.append(f"{prefix} {name}: {model['error']}")
                continue
            identity = model["identity"]
            if identity.get("kind") != "checkpoint" or not _hex_digest(identity.get("sha256")):
                errors.append(f"{prefix} {name}: missing checkpoint hash")
            keys = ("seconds", "bytes", "prompt_sha256", "prompt_sha256_per_row", "prompt_bytes_per_row",
                    "output_bytes_per_row", "overshoot_bytes", "generated_units_per_row",
                    "generated_patch_closures_per_row", "polls", "native_steps")
            if any(len(model[key]) != repetitions for key in keys):
                errors.append(f"{prefix} {name}: timed trial counts")
                continue
            if any(not math.isfinite(value) or value <= 0 for value in model["seconds"]):
                errors.append(f"{prefix} {name}: invalid elapsed time")
            if any(value != batch * output_bytes for value in model["bytes"]):
                errors.append(f"{prefix} {name}: throughput does not use requested bytes")
            if model["decoder_mode"] != model_cal["selected_mode"] or model["decoder_mode"] not in modes:
                errors.append(f"{prefix} {name}: selected mode differs from calibration")
            if set(model_cal["modes"]) != set(modes):
                errors.append(f"{prefix} {name}: missing calibration mode")
                continue
            calibration_hashes = None
            median_times = {}
            for mode in modes:
                sample = model_cal["modes"][mode]
                if (len(sample["seconds"]) != calibration_requests or
                        len(sample["prompt_sha256"]) != calibration_requests or
                        len(sample["prompt_sha256_per_row"]) != calibration_requests or
                        len(sample["prompt_bytes_per_row"]) != calibration_requests or
                        any(not math.isfinite(v) or v <= 0 for v in sample["seconds"])):
                    errors.append(f"{prefix} {name} {mode}: calibration trial counts or times")
                    continue
                median_times[mode] = float(np.median(sample["seconds"]))
                if not math.isclose(sample["median_seconds"], median_times[mode], rel_tol=1e-10):
                    errors.append(f"{prefix} {name} {mode}: calibration median is inconsistent")
                if calibration_hashes is None:
                    calibration_hashes = sample["prompt_sha256_per_row"]
                elif sample["prompt_sha256_per_row"] != calibration_hashes:
                    errors.append(f"{prefix} {name}: calibration prompts differ between modes")
                if any(len(group) != batch or len(set(group)) != batch or
                       any(not _hex_digest(v) for v in group) or _batch_digest(group) != digest
                       for group, digest in zip(sample["prompt_sha256_per_row"], sample["prompt_sha256"])):
                    errors.append(f"{prefix} {name} {mode}: invalid calibration prompt hashes")
                if any(len(group) != batch or any(value != kib * 1024 for value in group)
                       for group in sample["prompt_bytes_per_row"]):
                    errors.append(f"{prefix} {name} {mode}: invalid calibration prompt lengths")
            if len(median_times) == len(modes) and model["decoder_mode"] != min(modes, key=median_times.get):
                errors.append(f"{prefix} {name}: selected mode is not fastest on calibration")
            for trial in range(repetitions):
                hashes = model["prompt_sha256_per_row"][trial]
                sizes = model["prompt_bytes_per_row"][trial]
                produced = model["output_bytes_per_row"][trial]
                overshoot = model["overshoot_bytes"][trial]
                units = model["generated_units_per_row"][trial]
                steps, polls = model["native_steps"][trial], model["polls"][trial]
                if (len(hashes) != batch or len(set(hashes)) != batch or
                        any(not _hex_digest(v) for v in hashes) or _batch_digest(hashes) != model["prompt_sha256"][trial] or
                        len(sizes) != batch or any(v != 1024 * kib for v in sizes)):
                    errors.append(f"{prefix} {name} trial {trial}: prompt hashes or lengths")
                if (len(produced) != batch or len(overshoot) != batch or len(units) != batch or
                        any(n < output_bytes or n - output_bytes != o for n, o in zip(produced, overshoot)) or
                        not isinstance(steps, int) or steps <= 0 or steps % POLL_INTERVAL or
                        polls != steps // POLL_INTERVAL or any(u != steps for u in units)):
                    errors.append(f"{prefix} {name} trial {trial}: generation or polling counters")
                closures = model["generated_patch_closures_per_row"][trial]
                if closures is not None and (len(closures) != batch or
                                              any(not isinstance(v, int) or v < 0 for v in closures)):
                    errors.append(f"{prefix} {name} trial {trial}: invalid generated patch counts")
            timed = model["prompt_sha256_per_row"]
            checks = [v for v in cell["verification"] if v["identity"].get("sha256") == identity["sha256"]]
            for mode in modes:
                matches = [v for v in checks if v["decoder_mode"] == mode and v["greedy_mismatches"] == 0]
                if len(matches) != 1:
                    errors.append(f"{prefix} {name} {mode}: missing successful parity")
                    continue
                parity = matches[0]
                per_row = parity["checked_output_bytes_per_row"]
                if (len(per_row) != 1 or len(per_row[0]) != batch or
                        min(per_row[0]) < max(verify_bytes, cell["verify_bytes_per_row"]) or
                        len(parity["prompt_sha256"]) != batch or len(set(parity["prompt_sha256"])) != batch or
                        any(not _hex_digest(v) for v in parity["prompt_sha256"]) or
                        parity["prompt_bytes_per_row"] != [[kib * 1024] * batch]):
                    errors.append(f"{prefix} {name} {mode}: parity was too short or rows were duplicated")
                if checks and parity["prompt_sha256"] != checks[0]["prompt_sha256"]:
                    errors.append(f"{prefix} {name} {mode}: parity prompts differ between modes")
                ties = parity.get("tolerated_ties", 0)
                if (not isinstance(ties, int) or ties < 0 or ties > 0.01 * parity.get("checked_units", 0)
                        or len(parity.get("tie_records", [])) != ties):
                    errors.append(f"{prefix} {name} {mode}: numerical ties missing or above 1% of decisions")
                for record in parity.get("tie_records", []):
                    if not _valid_tie(record, batch):
                        errors.append(f"{prefix} {name} {mode}: invalid numerical-tie record {record}")
                        break
                closures = parity.get("generated_patch_closures_per_row")
                if closures is not None:
                    resets = parity.get("local_block_resets_per_row")
                    if (len(closures) != 1 or len(closures[0]) != batch or min(closures[0]) < 2 or
                            not resets or len(resets) != 1 or len(resets[0]) != batch or min(resets[0]) < 2):
                        errors.append(f"{prefix} {name} {mode}: parity lacks patch or local-block coverage")
            if checks and calibration_hashes is not None:
                phases = [sum(warm_hashes, []), sum(calibration_hashes, []), sum(timed, []),
                          checks[0]["prompt_sha256"]]
                if any(len(set(phase)) != len(phase) for phase in phases) or any(
                        set(phases[i]) & set(phases[j]) for i in range(len(phases)) for j in range(i)):
                    errors.append(f"{prefix} {name}: prompts overlap within or between phases")
        if len(models) == 2:
            if (models[0].get("prompt_sha256") != models[1].get("prompt_sha256") or
                    models[0].get("prompt_sha256_per_row") != models[1].get("prompt_sha256_per_row")):
                errors.append(f"{prefix}: paired timed prompts differ")
            if cal["models"][0]["modes"][modes[0]]["prompt_sha256_per_row"] != cal["models"][1]["modes"][modes[0]]["prompt_sha256_per_row"]:
                errors.append(f"{prefix}: paired calibration prompts differ")
            verification = cell["verification"]
            if verification and any(v["prompt_sha256"] != verification[0]["prompt_sha256"] for v in verification):
                errors.append(f"{prefix}: paired parity prompts differ")
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as error:
        errors.append(f"{prefix}: malformed or missing data: {error}")
    return errors


def problems(path: Path, runs: list[str], batches: list[int], kibs: list[int], repetitions: int,
             **settings) -> list[str]:
    try:
        results = json.loads(path.read_text(encoding="utf-8"))["results"]
        if not isinstance(results, list) or any(not isinstance(result, dict) for result in results):
            raise ValueError("results must be a list")
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [f"{path}: unreadable: {error}"]
    errors = []
    for kib in kibs:
        for batch in batches:
            cells = [v for v in results if v.get("prompt_kib") == kib and v.get("batch") == batch]
            if len(cells) != 1:
                errors.append(f"{kib} KiB batch {batch}: expected one result, found {len(cells)}")
            else:
                errors.extend(_cell_problems(cells[0], runs, batch, kib, repetitions, **settings))
    return errors


def summarise(root: Path) -> list[dict]:
    cells = {}
    for session in SESSIONS:
        directory = root / session
        paths = sorted(directory.rglob("*.json")) if directory.is_dir() else []
        if not paths:
            raise ValueError(f"{session}: no session JSON files in {directory}")
        for path in paths:
            try:
                results = json.loads(path.read_text(encoding="utf-8"))["results"]
                for cell in results:
                    rows = cell["variants"]["decode"]["models"]
                    if len(rows) != 2:
                        raise ValueError("summary requires pairs")
                    runs = [_run(row["identity"]) for row in rows]
                    errors = _cell_problems(cell, runs, cell["batch"], cell["prompt_kib"], cell["repetitions"])
                    if errors:
                        raise ValueError("; ".join(errors))
                    key = (*runs, cell["prompt_kib"], cell["batch"])
                    sessions = cells.setdefault(key, {})
                    if session in sessions:
                        raise ValueError(f"duplicate cell {key} in {session}")
                    sessions[session] = cell
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise ValueError(f"{path}: {error}") from error
    output = []
    for key, sessions in sorted(cells.items()):
        if set(sessions) != set(SESSIONS):
            raise ValueError(f"{key}: expected s1, s2 and s3")
        ordered = [sessions[session] for session in SESSIONS]
        if len({cell["repetitions"] for cell in ordered}) != 1:
            raise ValueError(f"{key}: trial counts differ across sessions")
        if key[2] == 8 and key[3] in (1, 8) and ordered[0]["repetitions"] != 30:
            raise ValueError(f"{key}: primary 8 KiB cells require 30 trials per session")
        rows = [cell["variants"]["decode"]["models"] for cell in ordered]
        for pair in rows[1:]:
            for first, row in zip(rows[0], pair):
                if (first["identity"]["sha256"] != row["identity"]["sha256"] or
                        first["prompt_sha256_per_row"] != row["prompt_sha256_per_row"]):
                    raise ValueError(f"{key}: checkpoints or ordered prompt batches differ across sessions")
        speeds = [np.asarray([[n / s for n, s in zip(pair[i]["bytes"], pair[i]["seconds"])]
                              for pair in rows], dtype=float) for i in range(2)]
        statistics = []
        for index in range(2):
            model_rows = [pair[index] for pair in rows]
            total_bytes = sum(sum(sum(trial) for trial in row["output_bytes_per_row"]) for row in model_rows)
            is_byte = model_rows[0]["generated_patch_closures_per_row"][0] is not None
            counts = ("generated_patch_closures_per_row" if is_byte else "generated_units_per_row")
            total = sum(sum(sum(trial) for trial in row[counts]) for row in model_rows)
            statistics.append({"unit": "patches/byte" if is_byte else "tokens/byte", "value": total / total_bytes})
        output.append({"runs": list(key[:2]), "kib": key[2], "batch": key[3],
                       "speed_a_over_b": _crossed(speeds[0], speeds[1]), "unit_statistics": statistics,
                       "selected_modes": [[model["decoder_mode"] for model in pair] for pair in rows]})
    return output


def markdown(rows: list[dict]) -> str:
    lines = ["| A | B | KiB | Batch | A/B speed [95% CI] | A units/byte | B units/byte |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        ratio, (a, b) = row["speed_a_over_b"], row["unit_statistics"]
        lines.append(f"| {row['runs'][0]} | {row['runs'][1]} | {row['kib']} | {row['batch']} | "
                     f"{ratio['value']:.3f} [{ratio['ci95_low']:.3f}, {ratio['ci95_high']:.3f}] | "
                     f"{a['value']:.4f} {a['unit']} | {b['value']:.4f} {b['unit']} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("file", type=Path)
    check.add_argument("--runs", nargs=2, required=True)
    check.add_argument("--batches", type=int, nargs="+", required=True)
    check.add_argument("--kib", type=int, nargs="+", required=True)
    check.add_argument("--repetitions", type=int, required=True)
    check.add_argument("--warmup", type=int, default=10)
    check.add_argument("--output-bytes", type=int, default=256)
    check.add_argument("--calibration-requests", type=int, default=10)
    check.add_argument("--verify-bytes", type=int, default=512)
    check.add_argument("--memory-budget-bytes", type=int, default=10_500_000_000)
    summary = sub.add_parser("summary")
    summary.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "check":
        errors = problems(args.file, args.runs, args.batches, args.kib, args.repetitions,
                          warmup=args.warmup, output_bytes=args.output_bytes,
                          calibration_requests=args.calibration_requests, verify_bytes=args.verify_bytes,
                          memory_budget=args.memory_budget_bytes)
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1 if errors else 0)
    try:
        print(markdown(summarise(args.root)))
    except ValueError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
