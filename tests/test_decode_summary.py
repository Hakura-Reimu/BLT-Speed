import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.diag.decode_summary import _batch_digest, main, markdown, problems, summarise


def _hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _fixture():
    batch, repetitions = 2, 2
    warmup = [[_hash(f"warm-{trial}-{row}") for row in range(batch)] for trial in range(10)]
    calibration = [[_hash(f"cal-{trial}-{row}") for row in range(batch)] for trial in range(10)]
    timed = [[_hash(f"timed-{trial}-{row}") for row in range(batch)] for trial in range(repetitions)]
    parity = [_hash(f"parity-{row}") for row in range(batch)]
    names = ("m-p6", "m-bpe16k")
    identities = [{"kind": "checkpoint", "path": f"/runs/{name}-seed0/ckpt/bytes-3000000000.pt",
                   "sha256": _hash(name)} for name in names]
    modes = ["device", "graph-split"]
    models, calibration_models, verification = [], [], []
    for index, identity in enumerate(identities):
        times = {"device": 1.0 if index == 0 else 2.0, "graph-split": 2.0 if index == 0 else 1.0}
        chosen = min(modes, key=times.get)
        calibration_models.append({"selected_mode": chosen, "modes": {
            mode: {"seconds": [times[mode]] * 10, "median_seconds": times[mode],
                   "prompt_sha256": [_batch_digest(group) for group in calibration],
                   "prompt_sha256_per_row": calibration,
                   "prompt_bytes_per_row": [[8192] * batch] * 10} for mode in modes}})
        models.append({"identity": identity, "decoder_mode": chosen,
                       "seconds": [2.0 if index == 0 else 1.0] * repetitions,
                       "bytes": [batch * 256] * repetitions,
                       "prompt_sha256": [_batch_digest(group) for group in timed],
                       "prompt_sha256_per_row": timed, "prompt_bytes_per_row": [[8192] * batch] * repetitions,
                       "output_bytes_per_row": [[256 if index == 0 else 512] * batch] * repetitions,
                       "overshoot_bytes": [[0 if index == 0 else 256] * batch] * repetitions,
                       "generated_units_per_row": [[256] * batch] * repetitions,
                       "generated_patch_closures_per_row": ([[64] * batch] * repetitions if index == 0
                                                            else [None] * repetitions),
                       "polls": [32] * repetitions, "native_steps": [256] * repetitions})
        for mode in modes:
            verification.append({"identity": identity, "decoder_mode": mode, "greedy_mismatches": 0,
                                 "checked_output_bytes_per_row": [[512] * batch], "prompt_sha256": parity,
                                 "prompt_bytes_per_row": [[8192] * batch],
                                 "generated_patch_closures_per_row": [[128] * batch] if index == 0 else None,
                                 "local_block_resets_per_row": [[8] * batch] if index == 0 else None})
    return {"mode": "decode", "decoder_variant": "selected", "batch": batch, "prompt_kib": 8,
            "warmup": 10, "repetitions": repetitions, "requested_output_bytes_per_row": 256,
            "completion_poll_interval_native_steps": 8, "verify_bytes_per_row": 512,
            "warmup_prompt_sha256_per_row": warmup, "warmup_prompt_bytes_per_row": [[8192] * batch] * 10,
            "memory_budget_bytes": 10_500_000_000,
            "peak_reserved_bytes_whole_cell": 1_000_000_000,
            "numerics": {"bf16_reduced_precision_reduction": False}, "verification": verification,
            "calibration": {"requests": 10, "requested_modes": modes, "models": calibration_models},
            "variants": {"decode": {"models": models}}}


def _write(path, cell):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": [cell]}), encoding="utf-8")


def test_check_accepts_valid_cell_and_rejects_missing_cells(tmp_path):
    path = tmp_path / "session.json"
    _write(path, _fixture())
    assert problems(path, ["m-p6", "m-bpe16k"], [2], [8], 2) == []
    assert any("expected one result" in error for error in
               problems(path, ["m-p6", "m-bpe16k"], [2, 8], [8], 2))


@pytest.mark.parametrize("corrupt", ["unpaired", "short_parity", "polls", "calibration", "overlap", "memory", "error"])
def test_check_rejects_incomplete_or_invalid_output(tmp_path, corrupt):
    cell = _fixture()
    models = cell["variants"]["decode"]["models"]
    if corrupt == "unpaired":
        models[1]["prompt_sha256"][0] = _hash("different")
    elif corrupt == "short_parity":
        cell["verification"][0]["checked_output_bytes_per_row"][0][0] = 511
    elif corrupt == "polls":
        models[0]["polls"][0] = 31
    elif corrupt == "calibration":
        models[0]["decoder_mode"] = "graph-split"
    elif corrupt == "overlap":
        cell["warmup_prompt_sha256_per_row"][0][0] = models[0]["prompt_sha256_per_row"][0][0]
    elif corrupt == "memory":
        cell["peak_reserved_bytes_whole_cell"] = 10_500_000_001
    else:
        models[0]["error"] = "out of memory"
    path = tmp_path / "bad.json"
    _write(path, cell)
    assert problems(path, ["m-p6", "m-bpe16k"], [2], [8], 2)


def test_summary_combines_paired_sessions_and_refuses_different_prompts(tmp_path):
    cell = _fixture()
    for session in ("s1", "s2", "s3"):
        _write(tmp_path / session / "pair.json", cell)
    combined = summarise(tmp_path)
    assert len(combined) == 1
    assert combined[0]["speed_a_over_b"]["value"] == pytest.approx(0.5)
    assert combined[0]["unit_statistics"][0]["value"] == pytest.approx(0.25)
    assert combined[0]["unit_statistics"][1]["value"] == pytest.approx(0.5)
    assert "0.500" in markdown(combined)

    changed = copy.deepcopy(cell)
    for model in changed["variants"]["decode"]["models"]:
        model["prompt_sha256_per_row"][0][0] = _hash("new prompt")
        model["prompt_sha256"][0] = _batch_digest(model["prompt_sha256_per_row"][0])
    _write(tmp_path / "s2" / "pair.json", changed)
    with pytest.raises(ValueError, match="prompt batches differ"):
        summarise(tmp_path)


def test_check_cli_exits_nonzero_for_an_invalid_cell(tmp_path, monkeypatch, capsys):
    path = tmp_path / "session.json"
    cell = _fixture()
    _write(path, cell)
    monkeypatch.setattr(sys, "argv", ["decode_summary.py", "check", str(path), "--runs", "m-p6", "m-bpe16k",
                                     "--batches", "2", "--kib", "8", "--repetitions", "2"])
    with pytest.raises(SystemExit) as success:
        main()
    assert success.value.code == 0
    cell["verification"][0]["greedy_mismatches"] = 1
    _write(path, cell)
    with pytest.raises(SystemExit) as failure:
        main()
    assert failure.value.code == 1
    assert "parity" in capsys.readouterr().err


def test_check_cli_runs_as_a_script(tmp_path):
    path = tmp_path / "session.json"
    _write(path, _fixture())
    script = Path(__file__).resolve().parents[1] / "scripts/diag/decode_summary.py"
    args = [sys.executable, "-B", str(script), "check", str(path), "--runs", "m-p6", "m-bpe16k",
            "--batches", "2", "--kib", "8", "--repetitions", "2"]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    args[-1] = "3"
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 1 and "trial counts" in result.stderr
