# Representation probes (plan section 4 "internal representations"; secondary claim "internal representations" in
# section 5) for the registered primary pair M-P6 / M-BPE16K at the 3 GB checkpoints. Run as a queue job:
# bash scripts/probe_run.sh. The example set ~/blt/data/probe-v1 was built from the validation split only, by
# scripts/diag/build_probe_examples.py, before any probe ran; its manifest.json holds the file hashes, checked here.
#   1. seed 0 pair (states cached);
#   2. the independently trained seed-1 pair on the same examples and split, with --replicate seed 0: this output
#      carries the registered verdict (blt.probe.probe_criteria);
#   3. seed 0 again with --replicate seed 1 (cached states, CPU fitting only), for the symmetric record.
# Natural patch ends only: every anchor was checked to close a byte patch without forcing (no --allow-forced-patch).
# Outputs: ~/blt/results/probes/probe-seed{0,1}.json, probe-seed0-vs-seed1.json, logs; a failed step leaves no output.
set -u
cd ~/blt/code && source ~/blt/venv/bin/activate
export PYTHONPATH=src
OUT=~/blt/results/probes; D=~/blt/data/probe-v1
mkdir -p $OUT/bad
K=~/blt/data/tokenizer16k-v1/tokenizer.json; SUPPLY=~/blt/data/supply-v1
failures=0
log() { echo "UK $(date +%H:%M:%S) $*" | tee -a $OUT/probes.log; }
python - <<'PY' || { log "PROBE_DATA_CHANGED"; exit 1; }
import hashlib, json, os
root = os.path.expanduser("~/blt/data/probe-v1")
manifest = json.load(open(f"{root}/manifest.json"))
for name, digest in manifest["files"].items():
    with open(f"{root}/{name}", "rb") as handle:
        if hashlib.sha256(handle.read()).hexdigest() != digest:
            raise SystemExit(f"{name} differs from the manifest")
PY

probe() {  # probe SEED OUTPUT_NAME [extra probe.py arguments]
  local seed=$1 name=$2; shift 2
  local out=$OUT/$name.json
  [ -f $out ] && return
  log "start $name"
  timeout 5400 python scripts/probe.py --byte-run ~/blt/runs/m-p6-seed$seed --bpe-run ~/blt/runs/m-bpe16k-seed$seed \
    --byte-checkpoint ~/blt/runs/m-p6-seed$seed/ckpt/bytes-3000000000.pt \
    --bpe-checkpoint ~/blt/runs/m-bpe16k-seed$seed/ckpt/bytes-3000000000.pt \
    --tokenizer $K --supply $SUPPLY --examples $D/examples.jsonl --robustness-text $D/robustness.txt \
    --state-cache $OUT/states-seed$seed.npz --device cuda --output $out.tmp "$@" > $OUT/$name.log 2>&1
  local code=$?
  if [ $code -eq 0 ] && [ -f $out.tmp ]; then
    mv $out.tmp $out; log "ok $name: $(tail -n 1 $OUT/$name.log)"
  else
    [ -f $out.tmp ] && mv $out.tmp $OUT/bad/$name-$(date +%s).json
    failures=$((failures + 1)); log "FAILED $name exit $code"
  fi
}

nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu --format=csv,noheader | tee -a $OUT/probes.log
git log --oneline -1 | tee -a $OUT/probes.log
probe 0 probe-seed0
[ -f $OUT/probe-seed0.json ] && probe 1 probe-seed1 --replicate $OUT/probe-seed0.json
[ -f $OUT/probe-seed1.json ] && probe 0 probe-seed0-vs-seed1 --replicate $OUT/probe-seed1.json
if [ $failures -gt 0 ]; then log "PROBES_INCOMPLETE $failures failed"; exit 1; fi
log "PROBES_DONE"
