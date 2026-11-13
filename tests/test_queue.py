import json
import os
from pathlib import Path
import sys
import time

import pytest

from blt.campaign_queue import QueueDaemon, _process_identity, daemon_lock


pytestmark = pytest.mark.skipif(os.name != "posix" or not Path("/proc").exists(), reason="Linux queue semantics")


def test_lock_excludes_another_daemon(tmp_path):
    with daemon_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another"):
            with daemon_lock(tmp_path):
                pass
    with daemon_lock(tmp_path):
        pass


def test_pause_signals_running_worker_and_requeues_it(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "src"
    monkeypatch.setenv("PYTHONPATH", str(source) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    root = tmp_path / "queue"
    pause = tmp_path / "PAUSE"
    ready = tmp_path / "ready"
    signalled = tmp_path / "signalled"
    program = ("import pathlib,signal,sys,time\n"
               "ready=pathlib.Path(sys.argv[1]); marker=pathlib.Path(sys.argv[2])\n"
               "def stop(a,b):\n marker.write_text('SIGTERM'); sys.exit(0)\n"
               "signal.signal(signal.SIGTERM,stop)\nready.write_text('ready')\n"
               "while True: time.sleep(0.01)\n")
    daemon = QueueDaemon(root, pause, poll_seconds=0.01, contention_check=lambda _: False)
    job = {"name": "test", "command": [sys.executable, "-c", program, str(ready), str(signalled)],
           "priority": 1, "created": "2026-01-01T00:00:00Z"}
    (root / "pending/test.json").write_text(json.dumps(job))
    daemon.tick()
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        pause.touch()
        daemon.tick()
        while daemon.process is not None and time.monotonic() < deadline:
            time.sleep(0.02)
            daemon.tick()
        assert signalled.read_text() == "SIGTERM"
        assert daemon.process is None
        pending = json.loads((root / "pending/test.json").read_text())
        assert pending["resume"] is True
        assert not (root / "done/test.json").exists()
        daemon.tick()
        assert daemon.process is None
    finally:
        if daemon.process is not None and daemon.process.poll() is None:
            daemon.process.kill()
            daemon.process.wait(timeout=5)


def test_stale_running_job_is_recovered(tmp_path):
    root = tmp_path / "queue"
    daemon = QueueDaemon(root, tmp_path / "PAUSE", contention_check=lambda _: False)
    job = {"name": "stale", "command": [sys.executable, "-c", "pass"], "priority": 1, "created": "2026"}
    path = root / "running/stale.json"
    path.write_text(json.dumps(job))
    # A marker for a vanished process is safe to discard without signaling a reused PID.
    path.with_suffix(".pid").write_text(json.dumps({"pid": os.getpid(), "start_time": "not-this-process"}))
    assert _process_identity(os.getpid()) is not None
    daemon.recover()
    assert json.loads((root / "pending/stale.json").read_text())["resume"]
    assert not path.exists() and not path.with_suffix(".pid").exists()


def test_successful_pause_exit_is_requeued_even_if_flag_is_already_removed(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "src"
    monkeypatch.setenv("PYTHONPATH", str(source) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    root = tmp_path / "queue"
    daemon = QueueDaemon(root, tmp_path / "PAUSE", contention_check=lambda _: False)
    code = "import json,os,pathlib; pathlib.Path(os.environ['BLT_QUEUE_OUTCOME_FILE']).write_text(json.dumps({'status':'paused'}))"
    job = {"name": "paused", "command": [sys.executable, "-c", code], "priority": 1, "created": "2026"}
    (root / "pending/paused.json").write_text(json.dumps(job))
    daemon.tick()
    daemon.process.wait(timeout=10)
    # Keep the requeued job from being immediately launched in the same tick.
    daemon.stopping = True
    daemon.tick()
    assert json.loads((root / "pending/paused.json").read_text())["resume"] is True
