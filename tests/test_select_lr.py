import importlib.util
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("select_lr_test", ROOT / "scripts/select_lr.py")
select_lr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(select_lr)

GRIDS = {"S-BPE50K": (3e-4, 6e-4, 1.2e-3), "S-P6": (3e-4, 6e-4, 1.2e-3),
         "M-BPE50K": (1.5e-4, 3e-4, 6e-4), "M-P8": (1.5e-4, 3e-4, 6e-4)}
# Screen BPB by LR index. S-BPE50K: best at the top edge. S-P6: the top LR diverges. M-P8: the best two are not
# adjacent, so the edge winner's neighbour has to be confirmed before an extension.
SCREEN = {"S-BPE50K": (1.30, 1.25, 1.20), "S-P6": (1.32, 1.28, 1.10), "M-BPE50K": (1.20, 1.15, 1.18),
          "M-P8": (1.15, 1.20, 1.17)}


def _train(name, model, lr, stop=150_000_000):
    command = ["python", "scripts/train.py", "--model", model, "--run-name", name, "--supply", "/supply",
               "--global-batch-bytes", "131072", "--micro-batch-size", "2", "--stop-bytes", str(stop),
               "--schedule-bytes", str(stop), "--learning-rate", repr(lr), "--compile"]
    if "BPE" in model:
        command += ["--tokenizer", "/gpt2.json"]
    return {"name": name, "priority": 1, "created": "t", "command": command}


def _run(root, name, bpb, scores, *, loss=1.0, status="complete", score=True, step=10):
    run = root / "runs" / name
    run.mkdir(parents=True)
    (run / "log.jsonl").write_text(json.dumps({"loss_nats_per_byte": loss, "grad_norm": 0.5}) + "\n", encoding="utf-8")
    (run / "status.json").write_text(json.dumps({"status": status, "step": 10}), encoding="utf-8")
    if score:
        scores.mkdir(parents=True, exist_ok=True)
        identity = {"path": str(run / "ckpt" / "rolling-0.pt"), "step": step}
        (scores / f"{name}.json").write_text(json.dumps({"bpb": bpb, "checkpoint": identity}), encoding="utf-8")


def _screen(root, overrides=None):
    jobs = root / "screen-jobs"
    jobs.mkdir()
    for model, grid in GRIDS.items():
        for lr, bpb in zip(grid, SCREEN[model]):
            name = f"lr-{model.lower()}-{lr:.1e}"
            (jobs / f"{name}.json").write_text(json.dumps(_train(name, model, lr)), encoding="utf-8")
            options = {"loss": float("nan")} if (model == "S-P6" and lr == 1.2e-3) else {}
            options.update((overrides or {}).get(name, {}))
            _run(root, name, bpb, root / "screen-scores", **options)


def _cli(root, stage):
    paths = ["--screen-jobs", root / "screen-jobs", "--screen-scores", root / "screen-scores",
             "--confirm-jobs", root / "confirm-jobs", "--confirm-scores", root / "confirm-scores",
             "--runs", root / "runs", "--tokenizer-16k", "/16k.json", "--output", root / "learning_rates.json"]
    return subprocess.run([sys.executable, str(ROOT / "scripts/select_lr.py"), stage, *map(str, paths)],
                          capture_output=True, text=True)


def _staged(root):
    jobs = {}
    for path in (root / "confirm-jobs").glob("confirm-*.json"):
        command = json.loads(path.read_text(encoding="utf-8"))["command"]
        jobs[path.stem] = (select_lr._arg(command, "--model"), float(select_lr._arg(command, "--learning-rate")), command)
    return jobs


def test_screen_confirm_tie_rule_neighbour_extension_and_transfers(tmp_path):
    _screen(tmp_path)
    assert _cli(tmp_path, "screen").returncode == 0
    staged = _staged(tmp_path)
    lrs = {}
    for model, lr, command in staged.values():
        lrs.setdefault(model, set()).add(lr)
        assert select_lr._arg(command, "--schedule-bytes") == "3000000000"
        assert select_lr._arg(command, "--stop-bytes") == ("150000000" if "16K" in model else "300000000")
        if "16K" in model:
            assert select_lr._arg(command, "--tokenizer") == "/16k.json"
            assert select_lr._arg(command, "--micro-batch-size") == str(select_lr.MICRO_BATCH_16K[model])
    assert lrs == {"S-BPE50K": {6e-4, 1.2e-3}, "S-P6": {3e-4, 6e-4}, "M-BPE50K": {3e-4, 6e-4},
                   "M-P8": {1.5e-4, 6e-4}, "S-BPE16K": {6e-4, 1.2e-3}, "M-BPE16K": {3e-4, 6e-4}}
    assert len(list((tmp_path / "confirm-jobs").glob("score-*.json"))) == len(staged) == 12

    confirm = {("S-BPE50K", 6e-4): 1.10, ("S-BPE50K", 1.2e-3): 1.08,  # edge winner leads its neighbour by 0.02
               ("S-P6", 3e-4): 1.004, ("S-P6", 6e-4): 1.000,  # within 0.005: the lower LR wins
               ("M-BPE50K", 3e-4): 1.05, ("M-BPE50K", 6e-4): 1.06,
               ("M-P8", 1.5e-4): 1.00, ("M-P8", 6e-4): 1.04,  # edge winner; neighbour 3e-4 not yet confirmed
               ("S-BPE16K", 6e-4): 1.12, ("S-BPE16K", 1.2e-3): 1.13, ("M-BPE16K", 3e-4): 1.07, ("M-BPE16K", 6e-4): 1.06,
               ("M-P8", 3e-4): 1.03, ("S-BPE50K", 2.4e-3): 1.09, ("M-P8", 7.5e-5): 1.02}

    def score_new():
        for name, (model, lr, _) in _staged(tmp_path).items():
            if not (tmp_path / "runs" / name).exists():
                _run(tmp_path, name, confirm[(model, lr)], tmp_path / "confirm-scores")

    score_new()
    before = set(_staged(tmp_path))
    assert _cli(tmp_path, "confirm").returncode == 0
    added = {_staged(tmp_path)[name][:2] for name in set(_staged(tmp_path)) - before}
    assert added == {("S-BPE50K", 2.4e-3), ("M-P8", 3e-4)}
    assert json.loads((tmp_path / "confirm-jobs/extensions.json").read_text()) == {"S-BPE50K": 2.4e-3}
    assert not (tmp_path / "learning_rates.json").exists()
    score_new()
    assert _cli(tmp_path, "confirm").returncode == 0  # M-P8 leads its neighbour by 0.03: extension to 7.5e-5
    assert json.loads((tmp_path / "confirm-jobs/extensions.json").read_text()) == {"S-BPE50K": 2.4e-3, "M-P8": 7.5e-5}
    score_new()
    result = _cli(tmp_path, "confirm")
    assert result.returncode == 0, result.stderr
    rates = json.loads((tmp_path / "learning_rates.json").read_text(encoding="utf-8"))["learning_rates"]
    assert rates == {"S-BPE50K": 1.2e-3, "S-P6": 3e-4, "M-BPE50K": 3e-4, "M-P8": 1.5e-4, "S-BPE16K": 6e-4,
                     "M-BPE16K": 6e-4, "S-P8": 3e-4, "M-P6": 1.5e-4, "M-P6-nohash": 1.5e-4}


def test_paused_or_unscored_runs_make_selection_pending(tmp_path):
    _screen(tmp_path, {"lr-m-p8-3.0e-04": {"status": "paused", "score": False}})
    result = _cli(tmp_path, "screen")
    assert result.returncode == select_lr.PENDING_EXIT and "lr-m-p8-3.0e-04" in result.stderr
    assert not (tmp_path / "confirm-jobs").exists()


def test_non_finite_scores_are_excluded_and_two_candidates_are_required(tmp_path):
    _screen(tmp_path, {"lr-s-p6-3.0e-04": {"loss": 1.0}, "lr-s-p6-6.0e-04": {"loss": 1.0}})
    nan_score = tmp_path / "screen-scores" / "lr-s-p6-6.0e-04.json"
    score = json.loads(nan_score.read_text())
    score["bpb"] = float("nan")
    nan_score.write_text(json.dumps(score))
    result = _cli(tmp_path, "screen")
    assert result.returncode != 0 and "S-P6: fewer than two eligible" in result.stderr


def test_a_score_from_another_checkpoint_is_rejected(tmp_path):
    _screen(tmp_path, {"lr-s-bpe50k-6.0e-04": {"step": 9}})
    result = _cli(tmp_path, "screen")
    assert result.returncode != 0 and "does not belong to the final checkpoint" in result.stderr
