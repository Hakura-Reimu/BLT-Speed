# usage: RUNS="<run> ..." [PRIORITY=20]; queues one test-split score of each run's 3 GB checkpoint (run inside WSL).
# The test split is opened once, after the selection and the analysis protocol are frozen (STATUS 2026-11-11 09:45):
# a run already scored or queued is skipped, never rescored. Low priority, so the jobs run after the registered
# training and timing work. Outputs: ~/blt/results/test-scores/<run>-bytes-3000000000.json.
mkdir -p ~/blt/results/test-scores
RUNS="$RUNS" PRIORITY="${PRIORITY:-20}" python3 - <<'PY'
import json, os, datetime
home = os.path.expanduser("~")
queue = f"{home}/blt/queue"
for name in os.environ["RUNS"].split():
    run = f"{home}/blt/runs/{name}"
    checkpoint = f"{run}/ckpt/bytes-3000000000.pt"
    output = f"{home}/blt/results/test-scores/{name}-bytes-3000000000.json"
    job_name = f"test-{name}-bytes-3000000000"
    if os.path.exists(output) or any(os.path.exists(f"{queue}/{state}/{job_name}.json")
                                      for state in ("pending", "running", "done", "failed")):
        print("skip", job_name)
        continue
    if not os.path.exists(checkpoint):
        print("missing", checkpoint)
        continue
    train_job = next(f"{queue}/done/{name}-{phase}.json" for phase in ("main", "repeat")
                     if os.path.exists(f"{queue}/done/{name}-{phase}.json"))
    train = json.load(open(train_job))["command"]
    command = ["python", "scripts/score.py", "--run", run, "--checkpoint", checkpoint,
               "--supply", f"{home}/blt/data/supply-v1", "--split", "test", "--output", output, "--device", "cuda"]
    if "--tokenizer" in train:
        command += ["--tokenizer", train[train.index("--tokenizer") + 1]]
    job = {"name": job_name, "priority": float(os.environ["PRIORITY"]),
           "created": datetime.datetime.now(datetime.timezone.utc).isoformat(), "command": command,
           "resume_command": command}
    path = f"{queue}/pending/{job_name}.json"
    with open(f"{path}.{os.getpid()}.tmp", "w") as handle:
        json.dump(job, handle, indent=1)
    os.replace(f"{path}.{os.getpid()}.tmp", path)
    print("queued", job_name)
PY
