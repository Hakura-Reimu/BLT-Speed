# Registered seed-1 repeats of the validation-selected pair, M-BPE16K and M-P6 (best 3 GB validation BPB of each
# family; the no-hash ablation is not eligible), queued behind the final timing (run inside WSL:
# bash -l < scripts/diag/queue_repeats.sh). The commands are the seed-0 mains' with only --run-name and --init-seed
# changed (same data order, hash seed, schedule, learning rate and micro-batch), as make_jobs.py stages repeats.
# Waits until final-timing is queued, so the repeats cannot start in a gap of the XS-BPE procedure; then queues the
# milestone scores of both runs as checkpoints appear, until both 3 GB scores exist. Stops at the first failed job.
cd ~/blt/code && Q=${Q:-~/blt/queue}
log() { echo "UK $(date +%H:%M) $*"; }
failed() { ls $Q/failed | grep -q . && { log "failed jobs:"; ls $Q/failed; exit 1; }; }
timing_queued() { for state in pending running done; do [ -f $Q/$state/final-timing.json ] && return 0; done; return 1; }
until timing_queued; do failed; sleep 60; done
Q=$Q python3 - <<'PY'
from datetime import datetime, timezone
import json, os
queue = os.path.expanduser(os.environ["Q"])
created = datetime.now(timezone.utc).isoformat()
for priority, source in ((40.0, "m-bpe16k-seed0"), (39.0, "m-p6-seed0")):
    command = list(json.load(open(f"{queue}/done/{source}-main.json"))["command"])
    name = source.replace("-seed0", "-seed1")
    for flag, value in (("--run-name", name), ("--init-seed", "1")):
        command[command.index(flag) + 1] = value
    path = f"{queue}/pending/{name}-repeat.json"
    if any(os.path.exists(f"{queue}/{state}/{name}-repeat.json") for state in ("pending", "running", "done", "failed")):
        print("skip", name)
        continue
    job = {"name": name, "priority": priority, "created": created, "command": command, "phase": "repeat",
           "source_job": f"{source}-main.json"}
    with open(f"{path}.{os.getpid()}.tmp", "w") as handle:
        json.dump(job, handle, indent=1)
    os.replace(f"{path}.{os.getpid()}.tmp", path)
    print("queued", name, priority)
PY
S=~/blt/results/main-scores
until [ -f $S/m-bpe16k-seed1-bytes-3000000000.json ] && [ -f $S/m-p6-seed1-bytes-3000000000.json ]; do
  for r in m-bpe16k-seed1 m-p6-seed1; do
    if { [ -f $Q/running/$r-repeat.json ] || [ -f $Q/done/$r-repeat.json ]; } && [ -d ~/blt/runs/$r/ckpt ]; then
      RUN=$r bash scripts/diag/queue_milestone_scores.sh | grep '^queued' | sed "s/^/UK $(date +%H:%M) /"
    fi
  done
  failed; sleep 60
done
log "seed-1 repeats scored"
