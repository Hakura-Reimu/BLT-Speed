# Queue scripts/final_timing.sh once every 3 GB model is scored and the queue is empty, then wait for it to finish
# (run inside WSL: bash -l < scripts/diag/queue_final_timing.sh). Queuing it only then keeps the daemon from starting
# the timing in a gap of the XS-BPE procedure, while drive_xs.sh is between stages.
log() { echo "UK $(date +%H:%M) $*"; }
Q=~/blt/queue; S=~/blt/results/main-scores
ready() {
  for run in xs-bpe50k xs-bpe16k m-p6-nohash; do [ -f $S/$run-seed0-bytes-3000000000.json ] || return 1; done
  [ -z "$(find $Q/pending $Q/running -type f)" ]
}
until ready; do
  ls $Q/failed | grep -q . && { log "failed jobs, not queuing the timing:"; ls $Q/failed; exit 1; }
  sleep 60
done
~/blt/venv/bin/python - <<'PY'
from datetime import datetime, timezone
import json, os
command = ["bash", "scripts/final_timing.sh"]
job = {"name": "final-timing", "priority": 50.0, "created": datetime.now(timezone.utc).isoformat(),
       "command": command, "resume_command": command}
path = os.path.expanduser("~/blt/queue/pending/final-timing.json")
with open(f"{path}.{os.getpid()}.tmp", "w") as handle:
    json.dump(job, handle, indent=1)
os.replace(f"{path}.{os.getpid()}.tmp", path)
PY
log "queued final-timing"
until [ -f $Q/done/final-timing.json ] || [ -f $Q/failed/final-timing.json ]; do sleep 60; done
log "final-timing $(ls $Q/done/final-timing.json $Q/failed/final-timing.json 2>/dev/null)"
tail -n 3 ~/blt/results/final-timing/timing.log
