from __future__ import annotations

"""Conservative, reproducible summaries of measured campaign results."""

import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable


CLAIMS = (
    "iso_quality_prefill", "task_retention", "batched_decode", "batch_one_latency",
    "longer_usable_context", "representation", "thin_bolmo_wsp", "router_programming",
)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _sha(identity: Any) -> str | None:
    return identity.get("sha256") if isinstance(identity, dict) and identity.get("kind") == "checkpoint" else None


def _initialization(identity: dict) -> dict | None:
    path = identity.get("path")
    if not path:
        return None
    checkpoint = Path(path)
    metadata = checkpoint.parent.parent / "initialization.json"
    if checkpoint.parent.name == "ckpt" and metadata.is_file():
        value = json.loads(metadata.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    return None


def _at_endpoint(identity: dict, endpoint: int) -> bool:
    # Milestones are saved at the first whole optimizer step crossing the target;
    # their actual byte counters may therefore be slightly above the nominal target.
    path = str(identity.get("path", "")).replace("\\", "/").split("/")[-1]
    if identity.get("nominal_training_bytes") == endpoint:
        return True
    origin = _initialization(identity)
    if origin is not None:
        nominal = origin.get("source_nominal_total_bytes")
        milestone = path[6:-3] if path.startswith("bytes-") and path.endswith(".pt") else ""
        if isinstance(nominal, int) and milestone.isdigit():
            return nominal + int(milestone) == endpoint
        actual = origin.get("source_total_consumed_bytes")
        return _number(actual) and _number(identity.get("consumed_bytes")) and actual + identity["consumed_bytes"] == endpoint
    return identity.get("consumed_bytes") == endpoint or path == f"bytes-{endpoint}.pt"


def _run_metadata(identity: dict) -> dict:
    config = identity.get("config") or {}
    training = {}
    counts = {}
    path = identity.get("path")
    if path:
        checkpoint = Path(path)
        config_path = checkpoint.parent.parent / "config.json"
        if checkpoint.parent.name == "ckpt" and config_path.is_file():
            frozen = json.loads(config_path.read_text(encoding="utf-8"))
            training = frozen.get("training", {})
            config = training.get("model", config)
            counts = frozen.get("parameter_counts", {})
    origin = _initialization(identity)
    actual = identity.get("consumed_bytes")
    previous = origin.get("source_total_consumed_bytes") if origin is not None else None
    total = actual + previous if _number(actual) and _number(previous) else actual
    return {"model": config.get("name", identity.get("preset", "unknown")),
            "family": config.get("family"), "seed": training.get("init_seed", identity.get("seed")),
            "parameters": counts.get("total"), "hash_enabled": config.get("hash_enabled"),
            "consumed_bytes": total, "run_consumed_bytes": actual, "checkpoint": identity.get("path"),
            "gpu_hours": identity.get("gpu_hours") if _number(identity.get("gpu_hours")) and identity["gpu_hours"] >= 0 else None}


def load_results(inputs: Iterable[str | Path]) -> dict[str, list[dict]]:
    """Read native score/bench/task/pilot/Track A/probe JSON without guessing missing data."""
    files = set()
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.update(path.rglob("*.json"))
        elif path.is_file():
            files.add(path)
        else:
            raise FileNotFoundError(path)
    out: dict[str, list[dict]] = {key: [] for key in ("scores", "benches", "tasks", "pilots", "track_a", "probes")}
    for path in sorted(files):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read result JSON {path}: {error}") from error
        if not isinstance(obj, dict):
            continue
        tagged = {"path": str(path), "data": obj}
        if obj.get("type") == "representation_probes_v1":
            out["probes"].append(tagged)
        elif "bpb" in obj and isinstance(obj.get("checkpoint"), dict) and "chunks" in obj:
            out["scores"].append(tagged)
        elif obj.get("task") in {"piqa", "hellaswag", "retrieval", "robustness"}:
            out["tasks"].append(tagged)
        elif "results" in obj and isinstance(obj["results"], list):
            if "model" in obj and "global_batch_bytes" in obj:
                out["pilots"].append(tagged)
            elif "protocol" in obj and "bolmo_prefill" in obj.get("protocol", {}):
                out["track_a"].append(tagged)
            elif any(isinstance(row, dict) and "mode" in row for row in obj["results"]):
                out["benches"].append(tagged)
        elif "models" in obj or "evaluation" in obj:
            out["track_a"].append(tagged)
    return out


def _score_index(results: dict, split: str, endpoint: int) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = {}
    for item in results["scores"]:
        data = item["data"]
        ident = data.get("checkpoint", {})
        key = _sha(ident)
        if key and data.get("split") == split and _at_endpoint(ident, endpoint) and _number(data.get("bpb")) and data["bpb"] > 0:
            grouped.setdefault(key, []).append(item)
    index = {}
    for key, candidates in grouped.items():
        first = candidates[0]["data"]
        if any((c["data"].get("bpb"), c["data"].get("chunks")) != (first.get("bpb"), first.get("chunks"))
               for c in candidates[1:]):
            raise ValueError(f"conflicting {split} scores for checkpoint {key}")
        index[key] = candidates[0]
    return index


def _rows(results: dict, *, mode: str, batch: int, variant: str) -> list[dict]:
    rows = []
    for document in results["benches"]:
        for case in document["data"].get("results", []):
            if case.get("mode") != mode or case.get("batch") != batch or "error" in case:
                continue
            variant_data = case.get("variants", {}).get(variant, {})
            if not isinstance(variant_data, dict):
                continue
            for row in variant_data.get("models", []):
                if _sha(row.get("identity")):
                    rows.append({"path": document["path"], "case": case, "variant": variant_data, "row": row})
    return rows


def points(results: dict, *, split: str, endpoint: int, batch: int = 1) -> list[dict]:
    scores = _score_index(results, split, endpoint)
    found = []
    for entry in _rows(results, mode="prefill", batch=batch, variant="end_to_end"):
        case, row = entry["case"], entry["row"]
        identity = row["identity"]
        key = _sha(identity)
        if case.get("prompt_kib") != 8 or key not in scores or not _number(row.get("bytes_per_second")) or row["bytes_per_second"] <= 0:
            continue
        if not _at_endpoint(identity, endpoint):
            continue
        score = scores[key]["data"]
        found.append({"sha256": key, "bpb": score["bpb"], "bytes_per_second": row["bytes_per_second"],
                      "score_source": scores[key]["path"], "bench_source": entry["path"],
                      **_run_metadata(identity)})
    # Repeated benchmark files are retained as separate measurements, never averaged silently.
    return found


def pareto_frontier(values: list[dict]) -> list[dict]:
    return sorted((p for p in values if not any(
        q["bytes_per_second"] >= p["bytes_per_second"] and q["bpb"] <= p["bpb"]
        and (q["bytes_per_second"] > p["bytes_per_second"] or q["bpb"] < p["bpb"])
        for q in values)), key=lambda p: p["bytes_per_second"])


def _score_pair(scores: dict[str, dict], byte_id: str, bpe_id: str) -> tuple[dict | None, str]:
    if byte_id not in scores or bpe_id not in scores:
        return None, "both held-out scores at the common endpoint are required"
    a, b = scores[byte_id]["data"], scores[bpe_id]["data"]
    ia, ib = a["checkpoint"], b["checkpoint"]
    if not ia.get("data_manifest_sha256") or ia.get("data_manifest_sha256") != ib.get("data_manifest_sha256"):
        return None, "the checkpoint data manifests are absent or different"
    def chunks(value):
        return [(r.get("document"), r.get("start"), r.get("end"), r.get("bytes")) for r in value.get("chunks", [])]
    if (not chunks(a) or chunks(a) != chunks(b)
            or any(not isinstance(doc, str) or not doc or not isinstance(start, int) or not isinstance(end, int)
                   or not isinstance(size, int) or start < 0 or end - start != size or size <= 0
                   for doc, start, end, size in chunks(a))):
        return None, "held-out document windows are missing or not identical"
    difference = abs(a["bpb"] - b["bpb"])
    relative = difference / b["bpb"] if b["bpb"] > 0 else None
    return {"byte_bpb": a["bpb"], "bpe_bpb": b["bpb"], "absolute_bpb_gap": difference,
            "relative_bpb_gap": relative,
            "approximate_iso_quality_005_5pct": (difference <= .05 + 1e-12 and relative <= .05 + 1e-12)
            if relative is not None else None}, ""


def _paired(results: dict, byte_id: str, bpe_id: str, *, mode: str, batch: int, prompt_kib: int,
            variant: str) -> dict | None:
    candidates = []
    for item in results["benches"]:
        for case in item["data"].get("results", []):
            if case.get("mode") != mode or case.get("batch") != batch or case.get("prompt_kib") != prompt_kib:
                continue
            if case.get("warmup", 0) < 10 or case.get("repetitions", 0) < 30:
                continue
            data = case.get("variants", {}).get(variant, {})
            rows, interval = data.get("models", []), data.get("paired_bootstrap", {})
            if len(rows) != 2 or {_sha(row.get("identity")) for row in rows} != {byte_id, bpe_id}:
                continue
            prompts_a, prompts_b = rows[0].get("prompt_sha256", []), rows[1].get("prompt_sha256", [])
            if (len(prompts_a) != case["repetitions"] or prompts_a != prompts_b
                    or any(not isinstance(digest, str) or not digest for digest in prompts_a)
                    or any(any(count > 0 for count in row.get("compile_new_graphs", [])) for row in rows)):
                continue
            ratio, low, high = (interval.get(k) for k in ("speed_a_over_b", "ci95_low", "ci95_high"))
            if not all(_number(x) and x > 0 for x in (ratio, low, high)) or low > high:
                continue
            measured = []
            for row in rows:
                seconds, sizes = row.get("seconds", []), row.get("bytes", [])
                if (len(seconds) != case["repetitions"] or len(sizes) != len(seconds)
                        or any(not _number(value) or value <= 0 for value in [*seconds, *sizes])):
                    break
                measured.append(median(size / elapsed for size, elapsed in zip(sizes, seconds)))
            if len(measured) != 2 or not math.isclose(measured[0] / measured[1], ratio, rel_tol=1e-6, abs_tol=1e-12):
                continue
            if _sha(rows[0].get("identity")) == bpe_id:
                ratio, low, high = 1 / ratio, 1 / high, 1 / low
            candidates.append({"byte_over_bpe": ratio, "ci95_low": low, "ci95_high": high,
                               "source": item["path"], "prompt_kib": prompt_kib, "batch": batch})
    # Multiple measurements require an explicit selection instead of choosing a winner.
    return candidates[0] if len(candidates) == 1 else None


def _result(status: str, reason: str, **numbers: Any) -> dict:
    return {"status": status, "reason": reason, "numbers": numbers}


def _supported_evidence(value: Any, keys: tuple[str, ...]) -> bool:
    return (isinstance(value, dict) and isinstance(value.get("sources"), list) and bool(value["sources"])
            and all(isinstance(s, str) and s for s in value["sources"]) and all(_number(value.get(k)) for k in keys))


def _context_verdict(results: dict, byte_id: str, bpe_id: str, budget: Any,
                     byte_source: str | None = None, bpe_source: str | None = None, *, split: str = "val") -> dict | None:
    if not _number(budget) or budget <= 0:
        return None
    chosen = {}
    manifests = set()
    for family, key, source in (("byte", byte_id, byte_source), ("bpe", bpe_id, bpe_source)):
        retrievals = [item for item in results["tasks"] if item["data"].get("task") == "retrieval"
                      and _sha(item["data"].get("checkpoint")) == key
                      and (source is None or Path(item["path"]).resolve() == Path(source).resolve())]
        if len(retrievals) != 1:
            return None
        document = retrievals[0]["data"]
        measurement = document.get("cache_measurement", {})
        if (measurement.get("checkpoint_sha256") != key or measurement.get("filler_split") != split
                or not measurement.get("supply_manifest_sha256")
                or measurement["supply_manifest_sha256"] != document.get("checkpoint", {}).get("data_manifest_sha256")):
            return None
        manifests.add(measurement["supply_manifest_sha256"])
        eligible = {}
        for size in (8, 16, 24, 32):
            cells = [cell for cell in document.get("cells", []) if cell.get("context_kib") == size]
            by_depth = {cell.get("depth"): cell for cell in cells}
            if set(by_depth) != {.1, .5, .9} or len(cells) != 3:
                continue
            if any(not isinstance(cell.get("count"), int) or cell["count"] < 200
                   or len(cell.get("trials", [])) != cell["count"]
                   or not _number(cell.get("greedy_accuracy")) or not 0 <= cell["greedy_accuracy"] <= 1
                   or any(not isinstance(trial.get("greedy_exact"), bool)
                          or trial.get("context_bytes") != size * 1024 for trial in cell["trials"])
                   or abs(sum(trial["greedy_exact"] for trial in cell["trials"]) / cell["count"]
                          - cell["greedy_accuracy"]) > 1e-12
                   or cell.get("cache_includes_query") is not True
                   or not _number(cell.get("max_physical_cache_bytes")) or cell["max_physical_cache_bytes"] <= 0
                   or cell.get("cache_generation_headroom_bytes", 0) < 5
                   or not isinstance(cell.get("cache_device"), str) or not cell["cache_device"]
                   or cell.get("cache_dtype") not in {"bfloat16", "float32"}
                   or len(cell.get("physical_cache_bytes", [])) != cell["count"]
                   or any(not _number(value) or value <= 0 for value in cell["physical_cache_bytes"])
                   or max(cell["physical_cache_bytes"]) != cell["max_physical_cache_bytes"]
                   for cell in cells):
                continue
            if len({(cell["cache_device"], cell["cache_dtype"]) for cell in cells}) != 1:
                continue
            maximum_cache = max(cell["max_physical_cache_bytes"] for cell in cells)
            if maximum_cache <= budget:
                eligible[size] = (by_depth, maximum_cache)
        if not eligible:
            return None
        size = max(eligible)
        by_depth, maximum_cache = eligible[size]
        chosen[family] = {"context_bytes": size * 1024, "max_physical_cache_bytes": maximum_cache,
                          "cache_device": by_depth[.1]["cache_device"], "cache_dtype": by_depth[.1]["cache_dtype"],
                          "retrieval_by_depth": {str(depth): by_depth[depth]["greedy_accuracy"] for depth in (.1, .5, .9)},
                          "retrieval_source": retrievals[0]["path"]}
    if len(manifests) != 1 or ((chosen["byte"]["cache_device"], chosen["byte"]["cache_dtype"]) !=
            (chosen["bpe"]["cache_device"], chosen["bpe"]["cache_dtype"])):
        return None
    ratio = chosen["byte"]["context_bytes"] / chosen["bpe"]["context_bytes"]
    byte_accuracy, bpe_accuracy = chosen["byte"]["retrieval_by_depth"], chosen["bpe"]["retrieval_by_depth"]
    worst_loss = max(bpe_accuracy[depth] - byte_accuracy[depth] for depth in byte_accuracy)
    passed = ratio >= 1.25 - 1e-12 and min(byte_accuracy.values()) >= .9 - 1e-12 and worst_loss <= .05 + 1e-12
    return _result("pass" if passed else "fail", "longest measured contexts within the same physical-cache budget; all three depths", budget_bytes=budget, ratio=ratio, worst_retrieval_loss=worst_loss, models=chosen)


def _router_verdict(results: dict, baseline_source: Any, edited_source: Any, *, split: str = "val") -> dict | None:
    if not isinstance(baseline_source, str) or not isinstance(edited_source, str):
        return None
    def find(source):
        matches = [item for item in results["track_a"] if Path(item["path"]).resolve() == Path(source).resolve()]
        return matches[0]["data"] if len(matches) == 1 else None
    before, after = find(baseline_source), find(edited_source)
    if before is None or after is None or before.get("split") != split or after.get("split") != split:
        return None
    a_id, b_id = before.get("checkpoint", {}), after.get("checkpoint", {})
    if not a_id.get("files_sha256") or a_id.get("files_sha256") != b_id.get("files_sha256"):
        return None
    if not before.get("calibration_samples") or before["calibration_samples"] != after.get("calibration_samples"):
        return None
    a, b = before.get("evaluation", {}), after.get("evaluation", {})
    def layout(value):
        return [(row.get("document"), row.get("start"), row.get("end"), row.get("bytes")) for row in value.get("chunks", [])]
    if not layout(a) or layout(a) != layout(b):
        return None
    values = (a.get("bpb"), b.get("bpb"), a.get("bytes_per_patch"), b.get("bytes_per_patch"),
              (before.get("calibration") or {}).get("p_out"), (after.get("calibration") or {}).get("p_out"))
    if not all(_number(value) and value > 0 for value in values):
        return None
    def sentinels(value):
        return {row.get("text"): row.get("nats") for row in value.get("sentinels", [])}
    old, new = sentinels(a), sentinels(b)
    if (not old or len(old) != len(a.get("sentinels", [])) or len(new) != len(b.get("sentinels", []))
            or set(old) != set(new) or any(not isinstance(key, str) or not _number(old[key])
            or not _number(new[key]) for key in old)):
        return None
    old_bpb, new_bpb, old_p, new_p, old_out, new_out = values
    p_gap, decode_p_gap = abs(new_p / old_p - 1), abs(new_out / old_out - 1)
    improvement = old_bpb - new_bpb
    passed = (p_gap <= .03 + 1e-12 and decode_p_gap <= .03 + 1e-12 and improvement >= .01 - 1e-12
              and all(new[key] <= old[key] + 1e-12 for key in old))
    return _result("pass" if passed else "fail", "matched evaluation P and decode calibration P, ≥0.01 BPB improvement, no sentinel degradation",
                   baseline_bpb=old_bpb, edited_bpb=new_bpb, improvement=improvement, prefill_p_relative_gap=p_gap,
                   decode_p_relative_gap=decode_p_gap, baseline_sentinels=old, edited_sentinels=new,
                   sources=[baseline_source, edited_source])


def _check_extra(evidence: dict, verdicts: dict) -> None:
    """Evaluate measured, source-labelled results for analyses with separate tools."""
    value = evidence.get("batch_one_latency")
    keys = ("byte_bytes_per_second", "strongest_bpe_bytes_per_second")
    if _supported_evidence(value, keys) and value.get("all_decoder_variants_compared") is True and value[keys[0]] > 0 and value[keys[1]] > 0:
        ratio = value[keys[0]] / value[keys[1]]
        verdicts["batch_one_latency"] = _result("pass" if ratio >= 1.10 - 1e-12 else "fail", "requires ≥1.10× the strongest tested BPE decoder", ratio=ratio, **value)
    value = evidence.get("longer_usable_context")
    keys = ("byte_context_bytes", "bpe_context_bytes", "byte_cache_bytes", "bpe_cache_bytes", "budget_bytes", "byte_retrieval", "bpe_retrieval")
    if (_supported_evidence(value, keys) and min(value[k] for k in keys[:5]) > 0
            and 0 <= value["byte_retrieval"] <= 1 and 0 <= value["bpe_retrieval"] <= 1):
        ratio = value["byte_context_bytes"] / value["bpe_context_bytes"]
        passed = (value["byte_cache_bytes"] <= value["budget_bytes"] and value["bpe_cache_bytes"] <= value["budget_bytes"]
                  and ratio >= 1.25 - 1e-12 and value["byte_retrieval"] >= 0.90 - 1e-12
                  and value["bpe_retrieval"] - value["byte_retrieval"] <= 0.05 + 1e-12)
        verdicts["longer_usable_context"] = _result("pass" if passed else "fail", "equal physical-cache budget, ≥1.25× bytes, ≥90% retrieval and ≤5-point loss", ratio=ratio, **value)
    value = evidence.get("representation")
    keys = ("attributes_with_gain", "repeat_attributes_with_gain", "byte_harmless_excess_nll", "bpe_harmless_excess_nll", "byte_digit_sensitivity", "bpe_digit_sensitivity")
    if (_supported_evidence(value, keys) and value["bpe_harmless_excess_nll"] > 0
            and 0 <= value["byte_digit_sensitivity"] <= 1 and 0 <= value["bpe_digit_sensitivity"] <= 1
            and value["attributes_with_gain"] >= 0 and value["repeat_attributes_with_gain"] >= 0
            and value.get("controls_passed") is True):
        ratio = value["byte_harmless_excess_nll"] / value["bpe_harmless_excess_nll"]
        passed = (value["attributes_with_gain"] >= 2 and value["repeat_attributes_with_gain"] >= 2
                  and ratio <= 0.80 + 1e-12 and value["byte_digit_sensitivity"] > 0
                  and value["byte_digit_sensitivity"] >= value["bpe_digit_sensitivity"])
        verdicts["representation"] = _result("pass" if passed else "fail", "two replicated controlled gains and ≥20% lower harmless excess NLL without losing digit sensitivity", excess_nll_ratio=ratio, **value)
    value = evidence.get("thin_bolmo_wsp")
    keys = ("teacher_bpb", "student_bpb", "programmed_recovery_bytes", "random_recovery_bytes")
    if _supported_evidence(value, keys) and min(value[k] for k in keys) > 0:
        gap = value["student_bpb"] - value["teacher_bpb"]
        fraction = value["programmed_recovery_bytes"] / value["random_recovery_bytes"]
        verdicts["thin_bolmo_wsp"] = _result("pass" if gap <= 0.03 + 1e-12 and fraction <= 0.75 + 1e-12 else "fail", "teacher gap ≤0.03 BPB and ≥25% less recovery data", bpb_gap=gap, recovery_fraction=fraction, **value)
    value = evidence.get("router_programming")
    keys = ("baseline_bpb", "edited_bpb", "baseline_p", "edited_p")
    if _supported_evidence(value, keys) and min(value[k] for k in keys) > 0:
        before, after = value.get("baseline_sentinels"), value.get("edited_sentinels")
        if isinstance(before, dict) and before and isinstance(after, dict) and set(before) == set(after) and all(
                _number(before[k]) and before[k] >= 0 and _number(after[k]) and after[k] >= 0 for k in before):
            p_gap = abs(value["edited_p"] - value["baseline_p"]) / value["baseline_p"]
            improvement = value["baseline_bpb"] - value["edited_bpb"]
            passed = p_gap <= 0.03 + 1e-12 and improvement >= 0.01 - 1e-12 and all(after[k] <= before[k] for k in before)
            verdicts["router_programming"] = _result("pass" if passed else "fail", "P matched within 3%, ≥0.01 BPB improvement, no sentinel NLL increase", improvement=improvement, p_relative_gap=p_gap, **value)


def evaluate_verdicts(results: dict, selection: dict | None = None, *, split: str = "val",
                      endpoint: int = 3_000_000_000) -> dict[str, dict]:
    selection = selection or {}
    verdicts = {name: _result("not run", "required selected checkpoints or measured evidence are missing") for name in CLAIMS}
    scores = _score_index(results, split, endpoint)
    byte_id, bpe_id = selection.get("byte_checkpoint_sha256"), selection.get("bpe_checkpoint_sha256")
    if byte_id and bpe_id and byte_id != bpe_id:
        repeat = (selection.get("repeat_byte_checkpoint_sha256"), selection.get("repeat_bpe_checkpoint_sha256"))
        batch = selection.get("prefill_batch", 1)
        checks, reasons, numbers = [], [], []
        if (selection.get("timing_hardware_verified") is not True
                or not isinstance(selection.get("timing_hardware_source"), str)
                or not selection["timing_hardware_source"]):
            reasons.append("benchmark JSON does not record the device; a source-labelled hardware verification is required")
        selection_frozen = (selection.get("validation_selection_frozen") is True
                            and isinstance(selection.get("validation_selection_source"), str)
                            and bool(selection["validation_selection_source"]))
        if not selection_frozen:
            reasons.append("a source-labelled validation-only selection must be frozen")
        for label, pair in (("selected", (byte_id, bpe_id)), ("repeat", repeat)):
            a, b = pair
            if not a or not b:
                reasons.append(f"{label}: selected checkpoint pair missing")
                continue
            quality, reason = _score_pair(scores, a, b)
            metrics: dict[str, Any] = {"pair": label}
            if quality is None:
                reasons.append(f"{label}: {reason}")
            else:
                metrics.update(quality)
                if quality["relative_bpb_gap"] is None:
                    reasons.append(f"{label}: invalid baseline BPB")
                else:
                    checks.append(quality["absolute_bpb_gap"] <= 0.03 + 1e-12 and quality["relative_bpb_gap"] <= 0.03 + 1e-12)
            for length in (8, 16):
                timing = _paired(results, a, b, mode="prefill", batch=batch, prompt_kib=length, variant="end_to_end")
                if timing is None:
                    reasons.append(f"{label}: unique paired {length} KiB timing and CI missing")
                else:
                    metrics[f"prefill_{length}kib"] = timing
                    checks.append(timing["byte_over_bpe"] >= 1.20 - 1e-12 and timing["ci95_low"] > 1.0)
            numbers.append(metrics)
        if all(value in scores for value in (byte_id, bpe_id, *repeat)):
            meta = [_run_metadata(scores[value]["data"]["checkpoint"]) for value in (byte_id, bpe_id, *repeat)]
            if (meta[0]["family"] != "byte" or meta[1]["family"] != "bpe"
                    or meta[2]["model"] != meta[0]["model"] or meta[3]["model"] != meta[1]["model"]
                    or [item["seed"] for item in meta] != [0, 0, 1, 1]):
                reasons.append("model family or independent repeat seeds cannot be verified")
        status = "fail" if False in checks else "not run" if reasons else "pass"
        verdicts["iso_quality_prefill"] = _result(status, "; ".join(reasons) if reasons else "quality, both timing CIs and repeat meet the registered thresholds", comparisons=numbers)

        task_metric = selection.get("task_metric")
        margin = selection.get("task_chance_margin")
        task_numbers, task_checks = {}, []
        task_reason = []
        if not selection_frozen:
            task_reason.append("a source-labelled validation-only selection must be frozen")
        if selection.get("task_split") != split:
            task_reason.append("the selected task dataset split is missing or differs from the report split")
        if (selection.get("bpe_task_reference_checked") is not True
                or not isinstance(selection.get("task_reference_source"), str)
                or not selection["task_reference_source"]):
            task_reason.append("source-labelled BPE reference-harness cross-check is required")
        if task_metric not in {"accuracy", "accuracy_byte_normalised"}:
            task_reason.append("preregistered task_metric is missing")
        else:
            for name, chance in (("piqa", 0.5), ("hellaswag", 0.25)):
                candidates = {key: [item for item in results["tasks"] if item["data"].get("task") == name
                                    and _sha(item["data"].get("checkpoint")) == key] for key in (byte_id, bpe_id)}
                if any(len(items) != 1 for items in candidates.values()):
                    task_reason.append(f"{name}: unique paired task results missing")
                    continue
                a, b = candidates[byte_id][0]["data"], candidates[bpe_id][0]["data"]
                def examples(obj):
                    return [(x.get("index"), x.get("context"), x.get("candidates"), x.get("label")) for x in obj.get("examples", [])]
                def valid_accuracy(obj):
                    rows = obj.get("examples", [])
                    prediction = "prediction" if task_metric == "accuracy" else "prediction_byte_normalised"
                    if not rows or obj.get("count") != len(rows):
                        return False
                    if any(not isinstance(row.get("candidates"), list) or len(row["candidates"]) != (2 if name == "piqa" else 4)
                           or not isinstance(row.get("label"), int)
                           or not isinstance(row.get(prediction), int) or not 0 <= row["label"] < len(row["candidates"])
                           or not 0 <= row[prediction] < len(row["candidates"]) for row in rows):
                        return False
                    return abs(sum(row[prediction] == row["label"] for row in rows) / len(rows) - obj[task_metric]) <= 1e-12
                if (not examples(a) or examples(a) != examples(b) or a.get("seed") != b.get("seed")
                        or not _number(a.get(task_metric)) or not _number(b.get(task_metric))
                        or not 0 <= a[task_metric] <= 1 or not 0 <= b[task_metric] <= 1
                        or not valid_accuracy(a) or not valid_accuracy(b)):
                    task_reason.append(f"{name}: frozen examples or accuracy do not match")
                    continue
                task_numbers[name] = {"byte": a[task_metric], "bpe": b[task_metric], "chance": chance,
                                      "byte_raw": a.get("accuracy"), "bpe_raw": b.get("accuracy"),
                                      "byte_normalised": a.get("accuracy_byte_normalised"),
                                      "bpe_normalised": b.get("accuracy_byte_normalised")}
                if _number(margin) and margin > 0:
                    task_checks.append(min(a[task_metric], b[task_metric]) > chance + margin)
                else:
                    task_reason.append("a preregistered positive near-chance margin is required")
        if len(task_numbers) == 2:
            loss = sum(v["bpe"] - v["byte"] for v in task_numbers.values()) / 2
            if loss > 0.02 + 1e-12:
                status = "fail"
            elif task_reason or not all(task_checks):
                status = "not run"
                if task_checks and not all(task_checks):
                    task_reason.append("one or more task accuracies are near chance")
            else:
                status = "pass"
            verdicts["task_retention"] = _result(status, "; ".join(dict.fromkeys(task_reason)) or "mean task accuracy loss ≤2 points", mean_accuracy_loss=loss, tasks=task_numbers, metric=task_metric)
        else:
            verdicts["task_retention"] = _result("not run", "; ".join(dict.fromkeys(task_reason)) or "both task comparisons are required",
                                                  tasks=task_numbers, metric=task_metric)

        decode_batch = selection.get("decode_batch")
        if decode_batch in (8, 32):
            timing = _paired(results, byte_id, bpe_id, mode="decode", batch=decode_batch,
                             prompt_kib=selection.get("decode_prompt_kib", 2), variant="decode")
            if timing is not None:
                verdicts["batched_decode"] = _result("pass" if timing["byte_over_bpe"] >= 1.20 - 1e-12 else "fail", "preregistered batch requires ≥1.20× output bytes/s", **timing)
            else:
                verdicts["batched_decode"] = _result("not run", "unique protocol-compliant paired timing is missing", batch=decode_batch)
    _check_extra(selection.get("evidence", {}), verdicts)
    if byte_id and bpe_id:
        context_result = _context_verdict(results, byte_id, bpe_id, selection.get("context_budget_bytes"),
                                          selection.get("retrieval_byte_source"), selection.get("retrieval_bpe_source"), split=split)
        if context_result is not None:
            verdicts["longer_usable_context"] = context_result
    router_result = _router_verdict(results, selection.get("router_baseline_source"), selection.get("router_edited_source"), split=split)
    if router_result is not None:
        verdicts["router_programming"] = router_result
    probe_source = selection.get("probe_result")
    matching_probes = [item for item in results["probes"]
                       if (not probe_source or item["path"] == probe_source)
                       and _sha(item["data"].get("checkpoints", {}).get("byte")) == byte_id
                       and _sha(item["data"].get("checkpoints", {}).get("bpe")) == bpe_id]
    if byte_id and bpe_id and len(matching_probes) == 1:
        item = matching_probes[0]
        criteria = item["data"].get("criteria", {})
        overall = criteria.get("overall", {})
        if overall.get("status") in {"pass", "fail", "not run"}:
            verdicts["representation"] = _result(overall["status"], overall.get("reason", "probe result"),
                                                  source=item["path"], accessibility=criteria.get("accessibility"),
                                                  robustness=criteria.get("robustness"))
    return verdicts


def _label(meta: dict) -> str:
    seed = "?" if meta.get("seed") is None else str(meta["seed"])
    label = f"{meta.get('model', 'unknown')} / seed {seed}"
    if _number(meta.get("parameters")):
        label += f" / {meta['parameters'] / 1_000_000:.1f}M params"
    if _number(meta.get("gpu_hours")):
        label += f" / {meta['gpu_hours']:.2f} GPU-h"
    return label


def _marker(model: str) -> str:
    if "nohash" in model.lower():
        return "X"
    if "BPE50K" in model:
        return "s"
    if "BPE16K" in model:
        return "D"
    if "P8" in model:
        return "^"
    if "P6" in model:
        return "o"
    return "P"


def measure_retrieval_cache(result: dict, bundle: Any, supply: str | Path) -> dict:
    """Reconstruct recorded retrieval trials and measure their actual tensor caches.

    This adds measured cache fields to a copy; it does not rerun or alter the
    recorded retrieval accuracy. The tokenizer, checkpoint and supply must match.
    """
    import copy
    import hashlib
    import torch

    from .bench import InferenceCase, cache_statistics
    from .config import BPEConfig
    from .data import iter_eval_chunks, sha256_file, verify_supply
    from .eval_tokens import TokenBytes
    from .score import autocast_for
    from .retrieval import RETRIEVAL_ANSWER_BYTES, retrieval_prompt
    from .tasks import retrieval_cases

    if not str(result.get("prompt_format", "")).startswith("v2") or any(
            not str(cell.get("prompt_format", "v2")).startswith("v2") for cell in result.get("cells", [])):
        raise ValueError("cache measurement reconstructs v2 retrieval prompts; this result uses another format")
    root = Path(supply)
    verify_supply(root)
    original_identity = result.get("checkpoint", {})
    if (result.get("task") != "retrieval" or not _sha(original_identity)
            or _sha(original_identity) != _sha(bundle.identity)
            or original_identity.get("data_manifest_sha256") != bundle.identity.get("data_manifest_sha256")
            or original_identity.get("tokenizer_sha256") != bundle.identity.get("tokenizer_sha256")
            or bundle.identity.get("data_manifest_sha256") != sha256_file(root / "manifest.json")):
        raise ValueError("retrieval, checkpoint and supply identities must match")
    chunks, total = [], 0
    for chunk in iter_eval_chunks(root, "val"):
        chunks.append(chunk.data)
        total += len(chunk.data)
        if total >= 1024 * 1024:
            break
    filler = b"\n".join(chunks)
    if not filler:
        raise ValueError("validation filler is empty")
    vocabulary = TokenBytes(bundle.tokenizer, bundle.config.vocab_size) if isinstance(bundle.config, BPEConfig) else None
    annotated = copy.deepcopy(result)
    if not annotated.get("cells"):
        raise ValueError("retrieval result has no cells")
    for cell in annotated["cells"]:
        size, depth, count, seed = cell.get("context_kib"), cell.get("depth"), cell.get("count"), cell.get("seed")
        if (not _number(size) or size <= 0 or not _number(depth) or not isinstance(count, int) or count <= 0
                or not isinstance(seed, int) or len(cell.get("trials", [])) != count):
            raise ValueError("retrieval cell lacks deterministic trial metadata")
        cases = retrieval_cases(filler, int(size * 1024), depth, count, seed)
        cache_bytes = []
        with torch.inference_mode():
            for case, trial in zip(cases, cell["trials"]):
                digest = hashlib.sha256(case["context"].encode("utf-8")).hexdigest()
                if (trial.get("context_sha256") != digest or trial.get("key") != case["key"]
                        or trial.get("distractors") != case["distractors"]
                        or trial.get("insertion_byte") != case["insertion_byte"]
                        or trial.get("context_bytes") != case["context_bytes"]
                        or trial.get("depth") != case["depth"]):
                    raise ValueError("reconstructed retrieval prompt does not match the recorded trial")
                prompt = retrieval_prompt(case).encode("utf-8")
                instance = InferenceCase(bundle, prompt, 1, output_bytes=RETRIEVAL_ANSWER_BYTES, token_bytes=vocabulary,
                                         prepare_resident=False)
                with autocast_for(bundle.device):
                    state, logits, _ = instance.prefill(resident=False, headroom=RETRIEVAL_ANSWER_BYTES)
                measured = cache_statistics(state, len(prompt), logits)["physical_cache_bytes"]
                if not _number(measured) or measured <= 0:
                    raise ValueError("retrieval cache measurement is invalid")
                cache_bytes.append(measured)
        cell.update(max_physical_cache_bytes=max(cache_bytes), physical_cache_bytes=cache_bytes,
                    cache_includes_query=True, cache_generation_headroom_bytes=RETRIEVAL_ANSWER_BYTES,
                    cache_device=str(bundle.device), cache_dtype="bfloat16" if bundle.device.type == "cuda" else "float32")
    annotated["cache_measurement"] = {"definition": "physical KV/state tensors for each reconstructed full retrieval prompt, with six bytes generation headroom (space and key)",
                                      "checkpoint_sha256": _sha(bundle.identity), "filler_split": "val",
                                      "supply_manifest_sha256": sha256_file(root / "manifest.json")}
    return annotated


def _draw_plots(results: dict, values: list[dict], output_dir: Path, *, endpoint: int, batch: int) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.4), constrained_layout=True)
    for item in values:
        ax.scatter(item["bytes_per_second"], item["bpb"], marker=_marker(item["model"]), s=65,
                   label=_label(item))
    frontier = pareto_frontier(values)
    if frontier:
        xs, ys = [p["bytes_per_second"] for p in frontier], [p["bpb"] for p in frontier]
        ax.plot(xs, ys, "k--", linewidth=1, label="observed frontier")
        if len(xs) > 1:
            ax.fill_between(xs, [y - .03 for y in ys], [y + .03 for y in ys], color="grey", alpha=.15,
                            label="±0.03 BPB")
        else:
            ax.axhspan(ys[0] - .03, ys[0] + .03, color="grey", alpha=.15, label="±0.03 BPB")
        ax.legend(fontsize="small")
    else:
        ax.text(.5, .5, "not run", ha="center", va="center", transform=ax.transAxes)
    ax.set(xlabel="8 KiB end-to-end prefill (bytes/s)", ylabel="held-out BPB", title=f"Common {endpoint:,}-byte endpoint")
    ax.grid(alpha=.2)
    fig.savefig(output_dir / "frontier.png", dpi=160)
    plt.close(fig)

    rows = _rows(results, mode="prefill", batch=batch, variant="end_to_end")
    grouped: dict[str, dict] = {}
    for item in rows:
        row, case = item["row"], item["case"]
        identity, length = row["identity"], case.get("prompt_kib")
        if not _at_endpoint(identity, endpoint) or not _number(length) or not _number(row.get("bytes_per_second")):
            continue
        key = _sha(identity)
        group = grouped.setdefault(key, {"label": _label(_run_metadata(identity)), "points": []})
        physical = []
        for cache, raw in zip(row.get("cache", []), row.get("prompt_bytes", [])):
            size = cache.get("physical_cache_bytes") if isinstance(cache, dict) else None
            if _number(size) and _number(raw) and raw > 0:
                physical.append(size / (raw * batch))
        group["points"].append((length, row["bytes_per_second"], median(physical) if physical else None))
    for filename, column, ylabel in (("prefill_lengths.png", 1, "end-to-end prefill (bytes/s)"),
                                     ("cache_per_byte.png", 2, "physical KV/state bytes per context byte")):
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        drawn = False
        for group in grouped.values():
            data = sorted((p for p in group["points"] if p[column] is not None), key=lambda p: p[0])
            if data:
                ax.plot([p[0] for p in data], [p[column] for p in data], marker="o", label=group["label"])
                drawn = True
        if drawn:
            ax.legend(fontsize="small")
        else:
            ax.text(.5, .5, "not run", ha="center", va="center", transform=ax.transAxes)
        ax.set(xlabel="prompt length (KiB)", ylabel=ylabel, title=f"batch {batch}, common endpoint")
        ax.grid(alpha=.2)
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)
    return ["frontier.png", "prefill_lengths.png", "cache_per_byte.png"]


def _display(value: Any, digits: int = 4) -> str:
    if value is None:
        return "not run"
    if isinstance(value, float):
        return f"{value:.{digits}f}" if math.isfinite(value) else "not run"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown(results: dict, values: list[dict], verdicts: dict, *, split: str, endpoint: int, batch: int) -> str:
    lines = ["## 研究结果", "", f"共同训练终点：{endpoint:,} 原始字节；留出集：{split}；prefill batch：{batch}。",
             "里程碑实际字节数可能在完整优化步骤结束时略微超出名义终点。所有缺失数据均标为 not run。",
             "Bootstrap 区间表示计时采样不确定性，不代表训练种子差异。", "",
             "### 留出 BPB 与 8 KiB prefill", "", "![BPB 与实测 prefill 前沿](frontier.png)", "",
             "| 模型 | 种子 | 参数 | 实际训练字节 | BPB | prefill 字节/秒 | GPU 小时 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    if values:
        for item in values:
            lines.append("| " + " | ".join(_display(item.get(key)) for key in
                         ("model", "seed", "parameters", "consumed_bytes", "bpb", "bytes_per_second", "gpu_hours")) + " |")
    else:
        lines.append("| not run | not run | not run | not run | not run | not run | not run |")
    lines += ["", "灰色带为观察前沿周围 ±0.03 BPB；主判定还要求相对差距 ≤3%。", "",
              "### 长度与缓存", "", "![prefill 与提示长度](prefill_lengths.png)", "",
              "![每上下文字节的物理 KV 和状态](cache_per_byte.png)", "",
              "物理缓存包含基准记录中的 KV 和状态张量；不包含 Python 对象开销。", "",
              "### 解码", "", "| 模型 | batch | 提示 KiB | 输出字节/秒 | 中位延迟秒 | 来源 |",
              "|---|---:|---:|---:|---:|---|"]
    decode = []
    for size in (1, 8, 32, 64):
        for entry in _rows(results, mode="decode", batch=size, variant="decode"):
            row, case = entry["row"], entry["case"]
            if _at_endpoint(row["identity"], endpoint):
                decode.append((_label(_run_metadata(row["identity"])), size, case.get("prompt_kib"),
                               row.get("bytes_per_second"), row.get("median_seconds"), entry["path"]))
    if decode:
        lines.extend("| " + " | ".join(_display(value) for value in row) + " |" for row in decode)
    else:
        lines.append("| not run | not run | not run | not run | not run | not run |")
    lines += ["", "### 预注册判定", "", "| 主张 | 判定 | 判定依据及缺项 | 数值 |", "|---|---|---|---|"]
    for claim in CLAIMS:
        item = verdicts[claim]
        details = json.dumps(item["numbers"], ensure_ascii=False, sort_keys=True) if item["numbers"] else "not run"
        lines.append(f"| {_display(claim)} | {_display(item['status'])} | {_display(item['reason'])} | {_display(details)} |")
    lines += ["", "### 训练 pilot", "", "| 模型 | 编译 | micro-batch | 字节/秒 | 峰值 reserved 字节 | 状态 | 来源 |",
              "|---|---|---:|---:|---:|---|---|"]
    pilot_rows = []
    for item in results["pilots"]:
        for row in item["data"].get("results", []):
            pilot_rows.append((item["data"].get("model"), row.get("compile"), row.get("micro_batch_size"),
                               row.get("bytes_per_second"), row.get("peak_reserved_bytes"), row.get("status"), item["path"]))
    if pilot_rows:
        lines.extend("| " + " | ".join(_display(value) for value in row) + " |" for row in pilot_rows)
    else:
        lines.append("| not run | not run | not run | not run | not run | not run | not run |")
    lines += ["", "### Track A", "", "| 模型或编辑 | BPB | 来源 |", "|---|---:|---|"]
    track_rows = []
    for item in results["track_a"]:
        data = item["data"]
        if isinstance(data.get("models"), dict):
            for name, model in data["models"].items():
                if isinstance(model, dict):
                    track_rows.append((name, model.get("bpb"), item["path"]))
        if isinstance(data.get("evaluation"), dict):
            track_rows.append(("router edit", data["evaluation"].get("bpb"), item["path"]))
    if track_rows:
        lines.extend("| " + " | ".join(_display(value) for value in row) + " |" for row in track_rows)
    else:
        lines.append("| not run | not run | not run |")
    lines += ["", "| Track A 模型 | 路径 | P | 提示 KiB | batch | 字节/秒 | 来源 |",
              "|---|---|---|---:|---:|---:|---|"]
    track_timing = []
    for item in results["track_a"]:
        for case in item["data"].get("results", []):
            if not isinstance(case, dict):
                continue
            variant = "end_to_end" if case.get("mode") == "prefill" else "decode"
            for row in case.get("variants", {}).get(variant, {}).get("models", []):
                track_timing.append((row.get("identity", {}).get("family"), case.get("mode"),
                                     case.get("forced_p") if case.get("forced_p") is not None else "stock",
                                     case.get("prompt_kib"), case.get("batch"), row.get("bytes_per_second"), item["path"]))
    if track_timing:
        lines.extend("| " + " | ".join(_display(value) for value in row) + " |" for row in track_timing)
    else:
        lines.append("| not run | not run | not run | not run | not run | not run | not run |")
    lines.append("")
    return "\n".join(lines)


def write_report(inputs: Iterable[str | Path], output_dir: str | Path, *, selection: dict | None = None,
                 split: str = "val", endpoint: int = 3_000_000_000, batch: int = 1) -> dict:
    if split not in {"val", "test"} or endpoint <= 0 or batch <= 0:
        raise ValueError("invalid report split, endpoint or batch")
    data = load_results(inputs)
    selected = selection or {}
    values = points(data, split=split, endpoint=endpoint, batch=batch)
    verdicts = evaluate_verdicts(data, selected, split=split, endpoint=endpoint)
    output = Path(output_dir)
    figures = _draw_plots(data, values, output, endpoint=endpoint, batch=batch)
    summary = {"split": split, "nominal_endpoint_bytes": endpoint, "prefill_batch": batch,
               "points": values, "pareto_frontier": pareto_frontier(values), "verdicts": verdicts,
               "selection": selected, "figures": figures, "input_counts": {key: len(value) for key, value in data.items()}}
    (output / "report.md").write_text(markdown(data, values, verdicts, split=split, endpoint=endpoint, batch=batch), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary
