from __future__ import annotations

from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Callable, Iterator

from .atomic import atomic_json


@contextmanager
def daemon_lock(root: str | Path) -> Iterator[None]:
    import fcntl

    path = Path(root) / "daemon.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another queue daemon holds the lock") from error
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _descendants(pid: int) -> set[int]:
    found = {pid}
    parents = {}
    for directory in Path("/proc").iterdir():
        if not directory.name.isdigit():
            continue
        try:
            # The executable name in /proc/<pid>/stat can contain spaces and parentheses.
            rest = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            parents[int(directory.name)] = int(rest[1])
        except (OSError, ValueError, IndexError):
            continue
    while True:
        children = {child for child, parent in parents.items() if parent in found}
        expanded = found | children
        if expanded == found:
            return found
        found = expanded


def gpu_contended(own_pid: int | None = None) -> bool:
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return False
    own = _descendants(own_pid) if own_pid is not None else set()
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            if int(parts[0]) not in own and int(parts[1]) > 1024:
                return True
        except ValueError:
            continue
    return False


def _command(job: dict) -> list[str]:
    command = job["command"]
    command = shlex.split(command) if isinstance(command, str) else list(command)
    if not command or not all(isinstance(part, str) for part in command):
        raise ValueError("job command must be a nonempty argument list or command string")
    if job.get("resume"):
        if job.get("resume_command"):
            resume = job["resume_command"]
            command = shlex.split(resume) if isinstance(resume, str) else list(resume)
        elif "--resume" not in command and "--resume-if-exists" not in command:
            command.append("--resume-if-exists")
    return command


def _process_identity(pid: int) -> tuple[str, str] | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return stat[0], stat[19]
    except (OSError, IndexError):
        return None


def _parent_death_signal(parent: int) -> Callable[[], None]:
    def setup() -> None:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() != parent:
            os._exit(1)
    return setup


class QueueDaemon:
    def __init__(self, root: str | Path, pause_file: str | Path, *, poll_seconds: float = 2.0,
                 heartbeat_seconds: float = 30.0, max_retries: int = 3, backoff_seconds: float = 60.0,
                 contention_check: Callable[[int | None], bool] = gpu_contended) -> None:
        if not 0 < poll_seconds <= 10 or not 0 < heartbeat_seconds <= 30 or max_retries < 0 or backoff_seconds < 0:
            raise ValueError("invalid queue timing or retry settings")
        self.root, self.pause_file = Path(root), Path(pause_file)
        self.poll_seconds, self.heartbeat_seconds = poll_seconds, heartbeat_seconds
        self.max_retries, self.backoff_seconds = max_retries, backoff_seconds
        self.contention_check = contention_check
        self.process: subprocess.Popen | None = None
        self.job_path: Path | None = None
        self.job: dict | None = None
        self.terminating_for_pause = False
        self.last_heartbeat = 0.0
        self.stopping = False
        for name in ("pending", "running", "done", "failed"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def recover(self) -> None:
        for path in sorted((self.root / "running").glob("*.json")):
            marker = path.with_suffix(".pid")
            if marker.exists():
                worker = json.loads(marker.read_text(encoding="utf-8"))
                pid = int(worker["pid"])
                identity = _process_identity(pid)
                if identity is not None and identity[0] != "Z" and identity[1] == worker["start_time"]:
                    if os.getpgid(pid) != pid:
                        raise RuntimeError("stale worker no longer owns its process group")
                    os.killpg(pid, signal.SIGTERM)
                    deadline = time.monotonic() + 60
                    while time.monotonic() < deadline:
                        identity = _process_identity(pid)
                        if identity is None or identity[0] == "Z" or identity[1] != worker["start_time"]:
                            break
                        time.sleep(0.2)
                    else:
                        raise RuntimeError("stale worker did not release its process within 60 seconds")
                marker.unlink(missing_ok=True)
            job = json.loads(path.read_text(encoding="utf-8"))
            job["resume"] = True
            atomic_json(path, job)
            destination = self.root / "pending" / path.name
            if destination.exists():
                raise FileExistsError(f"duplicate queue job: {path.name}")
            os.replace(path, destination)

    def _heartbeat(self, state: str) -> None:
        atomic_json(self.root / "heartbeat.json", {"time": time.time(), "status": state,
                                                   "job": self.job.get("name") if self.job else None,
                                                   "pid": self.process.pid if self.process else None})
        self.last_heartbeat = time.monotonic()

    def _next(self) -> tuple[Path, dict] | None:
        candidates = []
        for path in (self.root / "pending").glob("*.json"):
            job = json.loads(path.read_text(encoding="utf-8"))
            if not all(key in job for key in ("name", "command", "priority", "created")):
                raise ValueError(f"invalid queue job: {path}")
            if float(job.get("not_before", 0)) <= time.time():
                candidates.append((-float(job["priority"]), str(job["created"]), path.name, path, job))
        if not candidates:
            return None
        _, _, _, path, job = min(candidates)
        return path, job

    def _terminate(self) -> None:
        if self.process is not None and self.process.poll() is None and not self.terminating_for_pause:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            self.terminating_for_pause = True

    def _finish(self, returncode: int) -> None:
        assert self.job is not None and self.job_path is not None
        job, path = self.job, self.job_path
        if self.terminating_for_pause:
            job["resume"] = True
            job.pop("not_before", None)
            target = "pending"
        elif returncode == 0:
            target = "done"
        else:
            job["attempt"] = int(job.get("attempt", 0)) + 1
            job["last_returncode"] = returncode
            if job["attempt"] <= self.max_retries:
                job["resume"] = True
                job["not_before"] = time.time() + self.backoff_seconds * 2 ** (job["attempt"] - 1)
                target = "pending"
            else:
                target = "failed"
        atomic_json(path, job)
        destination = self.root / target / path.name
        if destination.exists():
            raise FileExistsError(f"queue destination already exists: {destination}")
        os.replace(path, destination)
        path.with_suffix(".pid").unlink(missing_ok=True)
        path.with_suffix(".outcome").unlink(missing_ok=True)
        self.process = self.job_path = self.job = None
        self.terminating_for_pause = False

    def tick(self) -> None:
        if self.process is not None:
            result = self.process.poll()
            if result is not None:
                assert self.job_path is not None
                outcome_path = self.job_path.with_suffix(".outcome")
                outcome = json.loads(outcome_path.read_text(encoding="utf-8")) if outcome_path.exists() else {}
                if self.pause_file.exists() or outcome.get("status") == "paused":
                    self.terminating_for_pause = True
                self._finish(result)
        paused = self.pause_file.exists() or self.contention_check(self.process.pid if self.process else None)
        if paused or self.stopping:
            self._terminate()
        elif self.process is None:
            candidate = self._next()
            if candidate is not None:
                path, job = candidate
                running = self.root / "running" / path.name
                if running.exists():
                    raise FileExistsError(running)
                os.replace(path, running)
                self.job_path, self.job = running, job
                running.with_suffix(".outcome").unlink(missing_ok=True)
                command = [sys.executable, "-m", "blt.queue_worker", str(running.with_suffix(".pid")), *_command(job)]
                self.process = subprocess.Popen(command, start_new_session=True, preexec_fn=_parent_death_signal(os.getpid()))
        if time.monotonic() - self.last_heartbeat >= self.heartbeat_seconds:
            self._heartbeat("stopping" if self.stopping else "paused" if paused else "running" if self.process else "idle")

    def run(self) -> None:
        previous_handler = signal.getsignal(signal.SIGTERM)
        def terminate(_signum: int, _frame: object) -> None:
            self.stopping = True
        with daemon_lock(self.root):
            signal.signal(signal.SIGTERM, terminate)
            try:
                self.recover()
                while not self.stopping or self.process is not None:
                    self.tick()
                    time.sleep(self.poll_seconds)
                self._heartbeat("stopped")
            except BaseException:
                self.pause_file.parent.mkdir(parents=True, exist_ok=True)
                self.pause_file.touch(exist_ok=True)
                self._terminate()
                if self.process is not None:
                    self.process.wait()
                    self._finish(self.process.returncode)
                self._heartbeat("error")
                raise
            finally:
                signal.signal(signal.SIGTERM, previous_handler)
