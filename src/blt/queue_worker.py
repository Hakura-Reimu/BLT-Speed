from __future__ import annotations

import os
from pathlib import Path
import sys

from .atomic import atomic_json


def main() -> None:
    marker = Path(sys.argv[1])
    command = sys.argv[2:]
    if not command:
        raise ValueError("missing worker command")
    pid = os.getpid()
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    atomic_json(marker, {"pid": pid, "start_time": stat[19]})
    os.environ["BLT_QUEUE_OUTCOME_FILE"] = str(marker.with_suffix(".outcome"))
    os.execvpe(command[0], command, os.environ)


if __name__ == "__main__":
    main()
