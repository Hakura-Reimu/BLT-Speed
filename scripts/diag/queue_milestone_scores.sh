# usage: RUN=<run name>; creates full-validation score jobs for every milestone checkpoint of the run.
# Safe to rerun while the run is still training: checkpoints already scored or already queued are skipped, so the
# scores of finished milestones can be queued early and do not wait behind the next training job.
mkdir -p ~/blt/results/main-scores
RUN="$RUN" python3 - <<'PY'
import json, os, datetime
name = os.environ["RUN"]
home = os.path.expanduser("~")
run = f"{home}/blt/runs/{name}"
queue = f"{home}/blt/queue"
train_job = next(f"{queue}/{state}/{name}-{phase}.json" for phase in ("main", "repeat") for state in ("done", "running")
                 if os.path.exists(f"{queue}/{state}/{name}-{phase}.json"))
train = json.load(open(train_job))["command"]
names = [f for f in sorted(os.listdir(run + "/ckpt")) if f.startswith(("bytes-", "active-")) and f.endswith(".pt")]
for f in names:
    tag = f[:-3]
    if os.path.exists(f"{home}/blt/results/main-scores/{name}-{tag}.json") or any(
            os.path.exists(f"{queue}/{state}/score-{name}-{tag}.json") for state in ("pending", "running", "done", "failed")):
        print("skip", f"score-{name}-{tag}")
        continue
    command = ["python", "scripts/score.py", "--run", run, "--checkpoint", f"{run}/ckpt/{f}",
               "--supply", f"{home}/blt/data/supply-v1", "--split", "val",
               "--output", f"{home}/blt/results/main-scores/{name}-{tag}.json", "--device", "cuda"]
    if "--tokenizer" in train:
        command += ["--tokenizer", train[train.index("--tokenizer") + 1]]
    # resume_command: a requeued job (after a pause or crash) would otherwise get --resume-if-exists, which score.py
    # rejects, and fail on every retry. Scoring has no state to resume, so it simply reruns.
    job = {"name": f"score-{name}-{tag}", "priority": 960,
           "created": datetime.datetime.now(datetime.timezone.utc).isoformat(), "command": command,
           "resume_command": command}
    path = f"{home}/blt/queue/pending/score-{name}-{tag}.json"
    with open(f"{path}.{os.getpid()}.tmp", "w") as handle:
        json.dump(job, handle, indent=1)
    os.replace(f"{path}.{os.getpid()}.tmp", path)
    print("queued", job["name"])
PY
