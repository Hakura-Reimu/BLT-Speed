import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


jobs = _module("make_jobs_test", "scripts/make_jobs.py")
train = _module("train_cli_test", "scripts/train.py")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _campaign(tmp_path):
    pilot_dir, supply = tmp_path / "pilots", tmp_path / "supply"
    pilot_dir.mkdir()
    supply.mkdir()
    (supply / "manifest.json").write_text('{"version": 1}', encoding="utf-8")
    t50, t16 = tmp_path / "50k.json", tmp_path / "16k.json"
    t50.write_text("50k", encoding="utf-8")
    t16.write_text("16k", encoding="utf-8")
    for model in jobs.BASE_MODELS:
        token = _sha(t50 if "50K" in model else t16) if "BPE" in model else None
        pilot = {"model": model, "device": "cuda:0", "global_batch_bytes": 65536, "window_bytes": 8192,
                 "manifest_sha256": _sha(supply / "manifest.json"), "tokenizer_sha256": token,
                 "results": [
                     {"status": "ok", "compile": True, "micro_batch_size": 1,
                      "bytes_per_second": 100, "peak_reserved_bytes": 9_000_000_000},
                     {"status": "ok", "compile": True, "micro_batch_size": 2,
                      "bytes_per_second": 200, "peak_reserved_bytes": jobs.MEMORY_LIMIT_BYTES},
                     {"status": "ok", "compile": True, "micro_batch_size": 4,
                      "bytes_per_second": 300, "peak_reserved_bytes": jobs.MEMORY_LIMIT_BYTES + 1},
                     {"status": "ok", "compile": False, "micro_batch_size": 1,
                      "bytes_per_second": 400, "peak_reserved_bytes": 1},
                 ]}
        (pilot_dir / f"{model}.json").write_text(json.dumps(pilot), encoding="utf-8")
    return pilot_dir, supply, t50, t16


LEARNING_RATES = {model: (6e-4 if model.startswith("S-") else 3e-4) * (2 if "BPE" in model else 1)
                  for model in jobs.BASE_MODELS}


def _invoke(tmp_path, extra=()):
    pilot_dir, supply, t50, t16 = _campaign(tmp_path)
    output = tmp_path / "staging"
    rates = tmp_path / "learning_rates.json"
    rates.write_text(json.dumps({"learning_rates": LEARNING_RATES}), encoding="utf-8")
    command = [sys.executable, str(ROOT / "scripts/make_jobs.py"), "--pilot-dir", str(pilot_dir),
               "--supply", str(supply), "--tokenizer-50k", str(t50), "--tokenizer-16k", str(t16),
               "--output-dir", str(output), "--run-root", str(tmp_path / "runs"), "--learning-rates", str(rates), *extra]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output, pilot_dir, supply, t50, t16


def test_main_jobs_use_fastest_safe_compiled_pilot_and_parse(tmp_path):
    output, _, supply, t50, t16 = _invoke(tmp_path)
    staged = [json.loads(path.read_text(encoding="utf-8")) for path in output.glob("*.json")]
    assert len(staged) == len(jobs.BASE_MODELS) == 9
    assert {entry["phase"] for entry in staged} == {"main"}
    assert {entry["pilot"]["micro_batch_size"] for entry in staged} == {2}
    assert all(entry["pilot"]["peak_reserved_bytes"] == jobs.MEMORY_LIMIT_BYTES for entry in staged)
    ordered = sorted(staged, key=lambda entry: -entry["priority"])
    assert [train.build_parser().parse_args(entry["command"][2:]).model for entry in ordered] == list(jobs.BASE_MODELS)
    help_text = subprocess.run([sys.executable, str(ROOT / "scripts/train.py"), "--help"],
                               check=True, capture_output=True, text=True).stdout
    assert "--init-from" in help_text and "--long-context" in help_text
    for entry in staged:
        parsed = train.build_parser().parse_args(entry["command"][2:])
        assert parsed.stop_bytes == parsed.schedule_bytes == 3_000_000_000
        assert parsed.compile and parsed.data_seed == parsed.hash_seed == parsed.init_seed == 0
        assert parsed.global_batch_bytes == 65536 and parsed.micro_batch_size == 2
        assert parsed.expected_manifest_sha256 == _sha(supply / "manifest.json")
        assert parsed.learning_rate == LEARNING_RATES[parsed.model]
        if "BPE" in parsed.model:
            assert parsed.expected_tokenizer_sha256 == _sha(t50 if "50K" in parsed.model else t16)


def test_optional_jobs_are_separate_selected_and_have_matched_targets(tmp_path):
    output, *_ = _invoke(tmp_path, ("--include-repeats", "--include-continuations", "--include-long-context",
                                    "--selected-bpe", "M-BPE16K", "--selected-byte", "M-P8",
                                    "--long-context-bytes", "120000000"))
    staged = [json.loads(path.read_text(encoding="utf-8")) for path in output.glob("*.json")]
    assert len(staged) == 17
    repeats = [entry for entry in staged if entry["phase"] == "repeat"]
    progress = [entry for entry in staged if entry["phase"] == "progress-continuation"]
    long_context = [entry for entry in staged if entry["phase"] == "long-context"]
    assert len(repeats) == 2 and len(progress) == 4 and len(long_context) == 2
    nohash = next(entry for entry in staged if entry["name"] == "m-p6-nohash-seed0")
    assert min(entry["priority"] for entry in repeats) > nohash["priority"] > max(entry["priority"] for entry in progress)
    for entry in repeats + progress + long_context:
        parsed = train.build_parser().parse_args(entry["command"][2:])
        assert parsed.model in {"M-BPE16K", "M-P8"}
        if entry in repeats:
            assert parsed.init_seed == 1 and parsed.init_from is None
        else:
            assert parsed.init_from.name == "bytes-3000000000.pt"
            assert parsed.stop_bytes == (120000000 if entry in long_context else 750000000)
            assert parsed.long_context == (entry in long_context)


def test_missing_or_wrong_pilots_are_rejected_before_writing(tmp_path):
    pilot_dir, supply, t50, t16 = _campaign(tmp_path)
    bad = pilot_dir / "S-P6.json"
    pilot = json.loads(bad.read_text(encoding="utf-8"))
    pilot["manifest_sha256"] = "wrong"
    bad.write_text(json.dumps(pilot), encoding="utf-8")
    with pytest.raises(ValueError, match="S-P6"):
        jobs.select_pilots(pilot_dir, _sha(supply / "manifest.json"), {"50K": _sha(t50), "16K": _sha(t16)})
    assert not (tmp_path / "staging").exists()


def test_learning_rate_map_must_cover_every_base_model(tmp_path):
    partial = {model: rate for model, rate in LEARNING_RATES.items() if model != "M-P6-nohash"}
    pilot_dir, supply, t50, t16 = _campaign(tmp_path)
    rates = tmp_path / "learning_rates.json"
    rates.write_text(json.dumps({"learning_rates": partial}), encoding="utf-8")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/make_jobs.py"), "--pilot-dir", str(pilot_dir),
                             "--supply", str(supply), "--tokenizer-50k", str(t50), "--tokenizer-16k", str(t16),
                             "--output-dir", str(tmp_path / "staging"), "--learning-rates", str(rates)],
                            capture_output=True, text=True)
    assert result.returncode != 0 and "learning-rates" in result.stderr
    assert not (tmp_path / "staging").exists()


def test_legacy_pilots_require_an_explicit_window_size_attestation(tmp_path):
    pilot_dir, supply, t50, t16 = _campaign(tmp_path)
    for path in pilot_dir.glob("*.json"):
        pilot = json.loads(path.read_text(encoding="utf-8"))
        pilot.pop("window_bytes")
        path.write_text(json.dumps(pilot), encoding="utf-8")
    hashes = {"50K": _sha(t50), "16K": _sha(t16)}
    with pytest.raises(ValueError, match="confirm-legacy-pilot-8k"):
        jobs.select_pilots(pilot_dir, _sha(supply / "manifest.json"), hashes)
    selected = jobs.select_pilots(pilot_dir, _sha(supply / "manifest.json"), hashes, confirm_legacy_8k=True)
    assert len(selected) == 9
    assert all(not pilot["pilot_window_bytes_recorded"] for pilot in selected.values())
