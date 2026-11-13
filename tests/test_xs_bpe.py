import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def xs(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("xs_bpe", ROOT / "scripts/diag/xs_bpe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    home = tmp_path / "blt"
    monkeypatch.setattr(module, "HOME", home)
    monkeypatch.setattr(module, "DIRS", {key: home / Path(value).relative_to(Path.home() / "blt")
                                          for key, value in module.DIRS.items()})
    for state in ("pending", "running", "done", "failed"):
        (home / "queue" / state).mkdir(parents=True)
    return module


def _finish(home: Path, job: dict, scores: Path, bpb: float) -> None:
    run = home / "runs" / job["name"]
    (run / "ckpt").mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"status": "complete", "step": 10}))
    (run / "log.jsonl").write_text(json.dumps({"loss_nats_per_byte": 1.0, "grad_norm": 0.5}) + "\n")
    scores.mkdir(parents=True, exist_ok=True)
    (scores / f"{job['name']}.json").write_text(json.dumps(
        {"bpb": bpb, "checkpoint": {"path": str(run / "ckpt" / "latest.pt"), "step": 10}}))


def _jobs(directory: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in directory.glob("*.json") if "command" in json.loads(p.read_text())}


def _arg(job: dict, flag: str) -> str:
    return job["command"][job["command"].index(flag) + 1]


def test_xs_procedure_follows_select_lr_rules(xs, monkeypatch):
    home, dirs = xs.HOME, xs.DIRS
    xs.stage_screen(None)
    screen = _jobs(dirs["screen_jobs"])
    trains = [job for job in screen.values() if "scripts/train.py" in job["command"]]
    assert sorted(float(_arg(job, "--learning-rate")) for job in trains) == list(xs.GRID)
    assert {_arg(job, "--model") for job in trains} == {"XS-BPE50K"}
    assert all(_arg(job, "--stop-bytes") == _arg(job, "--schedule-bytes") == "150000000" for job in trains)
    for job, bpb in zip(sorted(trains, key=lambda job: float(_arg(job, "--learning-rate"))), (1.50, 1.45, 1.47)):
        _finish(home, job, dirs["screen_scores"], bpb)

    monkeypatch.setattr("sys.argv", ["x"])
    xs.select(type("A", (), {"stage": "screen"})())
    confirm = _jobs(dirs["confirm_jobs"])
    confirm_trains = {name: job for name, job in confirm.items() if "scripts/train.py" in job["command"]}
    # Anchor: best two screened LRs at 300 MB; control: transferred winner and its better neighbour at 150 MB.
    assert set(confirm_trains) == {"confirm-xs-bpe50k-1.2e-03", "confirm-xs-bpe50k-2.4e-03",
                                   "confirm-xs-bpe16k-1.2e-03", "confirm-xs-bpe16k-2.4e-03"}
    for name, job in confirm_trains.items():
        assert _arg(job, "--schedule-bytes") == "3000000000"
        assert _arg(job, "--stop-bytes") == ("150000000" if "16k" in name else "300000000")
        assert ("tokenizer16k" in _arg(job, "--tokenizer")) == ("16k" in name)
        assert _arg(job, "--micro-batch-size") == str(xs.MICRO_BATCH)
    for name, job in confirm_trains.items():
        _finish(home, job, dirs["confirm_scores"], {"confirm-xs-bpe50k-1.2e-03": 1.392, "confirm-xs-bpe50k-2.4e-03": 1.389,
                                                    "confirm-xs-bpe16k-1.2e-03": 1.41,
                                                    "confirm-xs-bpe16k-2.4e-03": 1.43}[name])
    xs.select(type("A", (), {"stage": "confirm"})())
    # The lower LR wins within 0.005 BPB; it is not a grid edge, so no extension is staged.
    assert json.loads(dirs["rates"].read_text())["learning_rates"] == {"XS-BPE50K": 1.2e-3, "XS-BPE16K": 1.2e-3}

    for source, tokenizer in (("s-bpe50k-seed0", "gpt2"), ("s-bpe16k-seed0", "tok16k")):
        command = ["python", "scripts/train.py", "--model", "S", "--run-name", source, "--micro-batch-size", "2",
                   "--learning-rate", "0.0006", "--tokenizer", tokenizer]
        (home / "queue/done" / f"{source}-main.json").write_text(json.dumps({"command": command}))
    xs.stage_main(None)
    mains = _jobs(dirs["main_jobs"])
    assert set(mains) == {"xs-bpe50k-seed0-main", "xs-bpe16k-seed0-main"}
    assert _arg(mains["xs-bpe16k-seed0-main"], "--tokenizer") == "tok16k"
    assert _arg(mains["xs-bpe16k-seed0-main"], "--model") == "XS-BPE16K"

    xs.enqueue(type("A", (), {"staged": dirs["main_jobs"]})())
    xs.enqueue(type("A", (), {"staged": dirs["confirm_jobs"]})())
    pending = _jobs(home / "queue/pending")
    assert "xs-bpe50k-seed0-main" in pending and pending["xs-bpe50k-seed0-main"]["priority"] == xs.TRAIN_PRIORITY
    score = pending["score-confirm-xs-bpe16k-1.2e-03"]
    assert score["priority"] == xs.SCORE_PRIORITY and score["resume_command"] == score["command"]
    before = len(pending)
    xs.enqueue(type("A", (), {"staged": dirs["main_jobs"]})())
    assert len(_jobs(home / "queue/pending")) == before


def test_xs_outward_extension_then_second_driver_iteration(xs, monkeypatch):
    # drive_xs.sh's unattended branch: the anchor's winner sits on the grid edge and beats its neighbour by more
    # than 0.01, so `select confirm` stages one outward confirmation beside the first ones and writes no rates; the
    # driver enqueues it (the first ones are skipped as already queued) and runs `select confirm` again.
    home, dirs, queue = xs.HOME, xs.DIRS, xs.HOME / "queue"
    xs.stage_screen(None)
    trains = sorted((job for job in _jobs(dirs["screen_jobs"]).values() if "scripts/train.py" in job["command"]),
                    key=lambda job: float(_arg(job, "--learning-rate")))
    for job, bpb in zip(trains, (1.50, 1.47, 1.45)):
        _finish(home, job, dirs["screen_scores"], bpb)
    monkeypatch.setattr("sys.argv", ["x"])
    xs.select(type("A", (), {"stage": "screen"})())
    xs.enqueue(type("A", (), {"staged": dirs["confirm_jobs"]})())
    first = _jobs(queue / "pending")
    for name in first:
        (queue / "pending" / f"{name}.json").rename(queue / "done" / f"{name}.json")
    bpb = {"confirm-xs-bpe50k-2.4e-03": 1.38, "confirm-xs-bpe50k-1.2e-03": 1.40,
           "confirm-xs-bpe16k-2.4e-03": 1.41, "confirm-xs-bpe16k-1.2e-03": 1.42}
    for name, job in first.items():
        if "scripts/train.py" in job["command"]:
            _finish(home, job, dirs["confirm_scores"], bpb[name])
    xs.select(type("A", (), {"stage": "confirm"})())
    assert not dirs["rates"].exists()
    xs.enqueue(type("A", (), {"staged": dirs["confirm_jobs"]})())
    second = _jobs(queue / "pending")
    assert set(second) == {"confirm-xs-bpe50k-4.8e-03", "score-confirm-xs-bpe50k-4.8e-03"}
    extension = second["confirm-xs-bpe50k-4.8e-03"]
    assert extension["priority"] == xs.TRAIN_PRIORITY and _arg(extension, "--stop-bytes") == "300000000"
    assert second["score-confirm-xs-bpe50k-4.8e-03"]["resume_command"] == second["score-confirm-xs-bpe50k-4.8e-03"]["command"]
    _finish(home, extension, dirs["confirm_scores"], 1.39)
    xs.select(type("A", (), {"stage": "confirm"})())
    # One extension per anchor: 4.8e-3 is worse, 2.4e-3 stays, and the 16K control keeps its own confirmed winner.
    assert json.loads(dirs["rates"].read_text())["learning_rates"] == {"XS-BPE50K": 2.4e-3, "XS-BPE16K": 2.4e-3}
