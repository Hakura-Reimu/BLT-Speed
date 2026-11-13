from __future__ import annotations

import argparse
from pathlib import Path

from blt.campaign_queue import QueueDaemon


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the single-instance preemptible campaign queue")
    parser.add_argument("--queue", type=Path, default=Path.home() / "blt/queue")
    parser.add_argument("--pause-file", type=Path, default=Path("control/PAUSE"))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=60)
    args = parser.parse_args()
    QueueDaemon(args.queue, args.pause_file, max_retries=args.max_retries, backoff_seconds=args.backoff_seconds).run()


if __name__ == "__main__":
    main()
