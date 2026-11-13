import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from blt.config import ByteConfig, TransformerConfig
from blt.report import CLAIMS, evaluate_verdicts, load_results, measure_retrieval_cache, pareto_frontier, points, write_report
from blt.tasks import retrieval_cases
from blt.train import make_model


def _identity(family, seed):
    name = "M-P6" if family == "byte" else "M-BPE50K"
    return {"kind": "checkpoint", "sha256": f"{family}-{seed}", "seed": seed,
            "path": f"/runs/{name}-{seed}/ckpt/bytes-3000000000.pt", "consumed_bytes": 3_000_000_100,
            "data_manifest_sha256": "same-supply", "config": {"name": name, "family": family}}


def _fixture(tmp_path):
    ids = {(family, seed): _identity(family, seed) for family in ("byte", "bpe") for seed in (0, 1)}
    for (family, seed), identity in ids.items():
        score = {"checkpoint": identity, "split": "val", "bpb": 1.02 if family == "byte" else 1.0,
                 "chunks": [{"document": "d", "start": 0, "end": 10, "bytes": 10, "nats": 7.0}]}
        (tmp_path / f"score-{family}-{seed}.json").write_text(json.dumps(score), encoding="utf-8")
    cases = []
    for seed in (0, 1):
        for length in (8, 16):
            # The BPE is first: the byte/BPE ratio and interval must be inverted.
            rows = [{"identity": ids[(family, seed)], "bytes_per_second": speed,
                     "prompt_bytes": [length * 1024] * 30, "prompt_sha256": [f"prompt-{i}" for i in range(30)],
                     "bytes": [length * 1024] * 30, "seconds": [length * 1024 / speed] * 30,
                     "cache": [{"physical_cache_bytes": length * 2048}] * 30}
                    for family, speed in (("bpe", 100.0), ("byte", 130.0))]
            cases.append({"mode": "prefill", "batch": 1, "prompt_kib": length, "warmup": 10, "repetitions": 30,
                          "variants": {"end_to_end": {"models": rows, "paired_bootstrap": {
                              "speed_a_over_b": 1 / 1.3, "ci95_low": 1 / 1.4, "ci95_high": 1 / 1.25}}}})
    (tmp_path / "bench.json").write_text(json.dumps({"results": cases}), encoding="utf-8")
    for task in ("piqa", "hellaswag"):
        for family in ("byte", "bpe"):
            correct = 80 if family == "bpe" else 79
            task_examples = [{"index": index, "context": f"context-{index}",
                              "candidates": [str(i) for i in range(2 if task == "piqa" else 4)], "label": 0,
                              "prediction": 0 if index < correct else 1,
                              "prediction_byte_normalised": 0 if index < correct else 1} for index in range(100)]
            value = {"task": task, "seed": 4, "checkpoint": ids[(family, 0)],
                     "accuracy": .8 if family == "bpe" else .79,
                     "accuracy_byte_normalised": .8 if family == "bpe" else .79,
                     "count": len(task_examples), "examples": task_examples}
            (tmp_path / f"{task}-{family}.json").write_text(json.dumps(value), encoding="utf-8")
    selection = {"byte_checkpoint_sha256": "byte-0", "bpe_checkpoint_sha256": "bpe-0",
                 "repeat_byte_checkpoint_sha256": "byte-1", "repeat_bpe_checkpoint_sha256": "bpe-1",
                 "task_metric": "accuracy_byte_normalised", "task_chance_margin": .05, "task_split": "val",
                 "bpe_task_reference_checked": True, "task_reference_source": "task-cross-check.json",
                 "validation_selection_frozen": True, "validation_selection_source": "validation-selection.json",
                 "timing_hardware_verified": True, "timing_hardware_source": "environment.json"}
    return ids, selection


def test_frontier_and_registered_claims_use_actual_paired_data(tmp_path):
    _, selection = _fixture(tmp_path)
    results = load_results([tmp_path])
    data = points(results, split="val", endpoint=3_000_000_000)
    assert len(data) == 4
    assert all(point["model"] in {"M-P6", "M-BPE50K"} for point in data)
    frontier = pareto_frontier(data)
    assert {point["model"] for point in frontier} == {"M-P6", "M-BPE50K"}
    verdict = evaluate_verdicts(results, selection)
    assert verdict["iso_quality_prefill"]["status"] == "pass"
    comparison = verdict["iso_quality_prefill"]["numbers"]["comparisons"][0]
    assert comparison["prefill_8kib"]["byte_over_bpe"] == pytest.approx(1.3)
    assert comparison["prefill_8kib"]["ci95_low"] == pytest.approx(1.25)
    assert verdict["task_retention"]["status"] == "pass"
    assert verdict["task_retention"]["numbers"]["mean_accuracy_loss"] == pytest.approx(.01)
    assert verdict["batched_decode"]["status"] == "not run"


def test_missing_repeat_changed_windows_and_quality_failure_are_not_fabricated(tmp_path):
    _, selection = _fixture(tmp_path)
    data = load_results([tmp_path])
    no_repeat = dict(selection)
    no_repeat.pop("repeat_byte_checkpoint_sha256")
    assert evaluate_verdicts(data, no_repeat)["iso_quality_prefill"]["status"] == "not run"
    score = next(item["data"] for item in data["scores"] if item["data"]["checkpoint"]["sha256"] == "byte-0")
    score["chunks"][0]["document"] = "different"
    assert evaluate_verdicts(data, selection)["iso_quality_prefill"]["status"] == "not run"
    score["chunks"][0]["document"] = "d"
    score["bpb"] = 1.05
    assert evaluate_verdicts(data, selection)["iso_quality_prefill"]["status"] == "fail"
    no_margin = dict(selection)
    no_margin.pop("task_chance_margin")
    assert evaluate_verdicts(data, no_margin)["task_retention"]["status"] == "not run"


def test_evidence_requires_sources_and_evaluates_each_remaining_claim(tmp_path):
    _, selection = _fixture(tmp_path)
    selection["evidence"] = {
        "batch_one_latency": {"sources": ["decode.json"], "byte_bytes_per_second": 120,
                              "strongest_bpe_bytes_per_second": 100, "all_decoder_variants_compared": True},
        "longer_usable_context": {"sources": ["retrieval.json", "cache.json"], "byte_context_bytes": 32000,
                                  "bpe_context_bytes": 24000, "byte_cache_bytes": 100, "bpe_cache_bytes": 99,
                                  "budget_bytes": 100, "byte_retrieval": .93, "bpe_retrieval": .96},
        "representation": {"sources": ["probes.json"], "attributes_with_gain": 2, "repeat_attributes_with_gain": 2,
                           "byte_harmless_excess_nll": .08, "bpe_harmless_excess_nll": .1,
                           "byte_digit_sensitivity": .2, "bpe_digit_sensitivity": .2, "controls_passed": True},
        "thin_bolmo_wsp": {"sources": ["thin.json"], "teacher_bpb": 1.0, "student_bpb": 1.02,
                            "programmed_recovery_bytes": 70, "random_recovery_bytes": 100},
        "router_programming": {"sources": ["router.json"], "baseline_bpb": 1.2, "edited_bpb": 1.18,
                               "baseline_p": 6, "edited_p": 6.1, "baseline_sentinels": {"digits": 3},
                               "edited_sentinels": {"digits": 2.9}},
    }
    verdict = evaluate_verdicts(load_results([tmp_path]), selection)
    for key in selection["evidence"]:
        assert verdict[key]["status"] == "pass"
    selection["evidence"]["thin_bolmo_wsp"].pop("sources")
    assert evaluate_verdicts(load_results([tmp_path]), selection)["thin_bolmo_wsp"]["status"] == "not run"


def test_empty_report_marks_every_claim_not_run_and_writes_figures(tmp_path):
    summary = write_report([], tmp_path / "report")
    assert set(summary["verdicts"]) == set(CLAIMS)
    assert all(row["status"] == "not run" for row in summary["verdicts"].values())
    for name in ("report.md", "report.json", "frontier.png", "prefill_lengths.png", "cache_per_byte.png"):
        assert (tmp_path / "report" / name).is_file()


def test_continuation_endpoint_includes_recorded_source_bytes(tmp_path):
    run = tmp_path / "run"
    (run / "ckpt").mkdir(parents=True)
    (run / "initialization.json").write_text(json.dumps({"source_nominal_total_bytes": 3_000_000_000,
                                                         "source_total_consumed_bytes": 3_000_000_100}), encoding="utf-8")
    identity = {"kind": "checkpoint", "sha256": "continued", "seed": 0,
                "config": {"name": "M-P6", "family": "byte"},
                "path": str(run / "ckpt" / "bytes-750000000.pt"), "consumed_bytes": 750_000_050}
    score = {"split": "val", "checkpoint": identity, "bpb": 1.1, "chunks": [
        {"document": "d", "start": 0, "end": 5, "bytes": 5, "nats": 3}]}
    bench = {"results": [{"mode": "prefill", "batch": 1, "prompt_kib": 8,
                           "variants": {"end_to_end": {"models": [{"identity": identity, "bytes_per_second": 100}]}}}]}
    score_path, bench_path = tmp_path / "score.json", tmp_path / "bench.json"
    score_path.write_text(json.dumps(score), encoding="utf-8")
    bench_path.write_text(json.dumps(bench), encoding="utf-8")
    results = load_results([score_path, bench_path])
    selected = points(results, split="val", endpoint=3_750_000_000)
    assert len(selected) == 1 and selected[0]["consumed_bytes"] == 3_750_000_150
    assert points(results, split="val", endpoint=750_000_000) == []


def test_context_verdict_uses_measured_cache_and_all_retrieval_depths(tmp_path):
    identities, selection = _fixture(tmp_path)
    document = json.loads((tmp_path / "bench.json").read_text(encoding="utf-8"))
    document["results"].append({"mode": "prefill", "batch": 1, "prompt_kib": 24, "warmup": 10, "repetitions": 30,
                                "variants": {"end_to_end": {"models": [{"identity": identities[("byte", 0)],
                                  "bytes_per_second": 100, "cache": [{"physical_cache_bytes": 36_000}] * 30}]}}})
    (tmp_path / "bench.json").write_text(json.dumps(document), encoding="utf-8")
    for family, size, accuracy in (("byte", 24, .91), ("bpe", 16, .94)):
        cache = 36000 if family == "byte" else 32768
        cells = [{"context_kib": size, "depth": depth, "count": 200, "greedy_accuracy": accuracy,
                  "trials": [{"greedy_exact": index < round(accuracy * 200), "context_bytes": size * 1024}
                             for index in range(200)], "cache_includes_query": True,
                  "max_physical_cache_bytes": cache, "physical_cache_bytes": [cache] * 200,
                  "cache_generation_headroom_bytes": 5, "cache_device": "cuda", "cache_dtype": "bfloat16"}
                 for depth in (.1, .5, .9)]
        (tmp_path / f"retrieval-{family}.json").write_text(json.dumps({"task": "retrieval", "cells": cells,
                                      "checkpoint": identities[(family, 0)],
                                      "cache_measurement": {"checkpoint_sha256": f"{family}-0",
                                                            "supply_manifest_sha256": "same-supply", "filler_split": "val"}}), encoding="utf-8")
    selection["context_budget_bytes"] = 40_000
    verdict = evaluate_verdicts(load_results([tmp_path]), selection)["longer_usable_context"]
    assert verdict["status"] == "pass" and verdict["numbers"]["ratio"] == 1.5
    selection["context_budget_bytes"] = 20_000
    assert evaluate_verdicts(load_results([tmp_path]), selection)["longer_usable_context"]["status"] == "not run"
    selection["context_budget_bytes"] = 40_000
    retrieval_path = tmp_path / "retrieval-byte.json"
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    for cell in retrieval["cells"]:
        cell.pop("cache_includes_query")
    retrieval_path.write_text(json.dumps(retrieval), encoding="utf-8")
    assert evaluate_verdicts(load_results([tmp_path]), selection)["longer_usable_context"]["status"] == "not run"


def test_router_verdict_reads_p_matched_native_track_a_results(tmp_path):
    def output(bpb, p, sentinel):
        return {"checkpoint": {"files_sha256": {"weights": "same"}}, "split": "val",
                "calibration": {"p_out": p}, "calibration_samples": [{"sha256": "same"}],
                "evaluation": {"bpb": bpb, "bytes_per_patch": p,
                    "chunks": [{"document": "d", "start": 0, "end": 10, "bytes": 10}],
                    "sentinels": [{"text": "123", "nats": sentinel}]}}
    baseline, edited = tmp_path / "baseline.json", tmp_path / "edited.json"
    baseline.write_text(json.dumps(output(1.2, 6, 5)), encoding="utf-8")
    edited.write_text(json.dumps(output(1.18, 6.1, 4.9)), encoding="utf-8")
    selection = {"router_baseline_source": str(baseline), "router_edited_source": str(edited)}
    assert evaluate_verdicts(load_results([baseline, edited]), selection)["router_programming"]["status"] == "pass"
    edited.write_text(json.dumps(output(1.18, 6.1, 5.1)), encoding="utf-8")
    assert evaluate_verdicts(load_results([baseline, edited]), selection)["router_programming"]["status"] == "fail"


def test_retrieval_cache_measurement_reconstructs_exact_prompts(tmp_path):
    supply = tmp_path / "supply"
    supply.mkdir()
    raw = b"A held out document with ordinary text and numbers. " * 30
    index = (json.dumps({"start": 0, "end": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}) + "\n").encode()
    files = {}
    for split in ("train", "val", "test"):
        for ext, data in (("bin", raw), ("idx", index)):
            name = f"{split}.{ext}"
            (supply / name).write_bytes(data)
            files[name] = {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = supply / "manifest.json"
    manifest.write_text(json.dumps({"version": 1, "files": files}), encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    identity = {"kind": "checkpoint", "sha256": "checkpoint", "data_manifest_sha256": manifest_hash}
    trunk = TransformerConfig(1, 8, 2, 1, 4, 16, max_positions=256)
    config = ByteConfig("tiny", trunk, 8, 2, 16, 2, 6, max_len=8, block_size=16, hash_rows=8)
    bundle = SimpleNamespace(model=make_model(config).eval(), config=config, tokenizer=None,
                             device=torch.device("cpu"), identity=identity)
    cases = retrieval_cases(raw, 128, .5, 2, 7)
    trials = [{"key": case["key"], "distractors": case["distractors"], "insertion_byte": case["insertion_byte"],
               "context_bytes": case["context_bytes"], "depth": case["depth"],
               "context_sha256": hashlib.sha256(case["context"].encode()).hexdigest()} for case in cases]
    original = {"task": "retrieval", "prompt_format": "v2", "checkpoint": identity, "cells": [{"context_kib": .125, "depth": .5,
                 "seed": 7, "count": 2, "greedy_accuracy": .5, "trials": trials}]}
    annotated = measure_retrieval_cache(original, bundle, supply)
    cell = annotated["cells"][0]
    assert cell["cache_includes_query"] and cell["cache_generation_headroom_bytes"] == 6
    assert len(cell["physical_cache_bytes"]) == 2 and cell["max_physical_cache_bytes"] > 0
    assert "cache_includes_query" not in original["cells"][0]
    original["cells"][0]["trials"][0]["key"] = "wrong"
    with pytest.raises(ValueError, match="does not match"):
        measure_retrieval_cache(original, bundle, supply)


def test_retrieval_cache_measurement_rejects_v1_results():
    with pytest.raises(ValueError, match="v2 retrieval prompts"):
        measure_retrieval_cache({"task": "retrieval", "cells": []}, None, "unused")
