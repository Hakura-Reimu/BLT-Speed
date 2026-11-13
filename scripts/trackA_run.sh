# Track A1 (plan section 4): released Bolmo-1B against OLMo-2-0425-1B on our validation split, within the 3 GPU-hour
# cap. Run as a queue job: bash scripts/trackA_run.sh. Steps, each skipped when its output exists:
#   1. folded BPB of both models on the validation split (scripts/trackA_bpb.py; the test split stays closed);
#   2. paired prefill timing at 8 and 16 KiB, batches 1 and 4, Bolmo stock and forced P = 4.4, 6, 8
#      (scripts/trackA_bench.py, 10 warm-ups, 30 timed requests, alternating order, with the Bolmo event profile).
# The Thin-Bolmo gate computed from step 2 is defined in STATUS 2026-11-11 19:20, before this run.
# Outputs: ~/blt/results/trackA/{bpb-val,prefill}.json and logs; a step that fails or times out leaves no output.
set -u
cd ~/blt/code && source ~/blt/venv/bin/activate
OUT=~/blt/results/trackA
mkdir -p $OUT/bad
B=~/blt/models/Bolmo-1B; O=~/blt/models/OLMo-2-0425-1B; SUPPLY=~/blt/data/supply-v1
failures=0
log() { echo "UK $(date +%H:%M:%S) $*" | tee -a $OUT/trackA.log; }
step() {  # step NAME SECONDS COMMAND...
  local name=$1 limit=$2; shift 2
  [ -f $OUT/$name.json ] && return
  log "start $name"
  timeout $limit "$@" --output $OUT/$name.json.tmp > $OUT/$name.log 2>&1
  local code=$?
  if [ $code -eq 0 ] && [ -f $OUT/$name.json.tmp ]; then
    mv $OUT/$name.json.tmp $OUT/$name.json; log "ok $name"
  else
    [ -f $OUT/$name.json.tmp ] && mv $OUT/$name.json.tmp $OUT/bad/$name-$(date +%s).json
    failures=$((failures + 1)); log "FAILED $name exit $code"
  fi
}

nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu --format=csv,noheader | tee -a $OUT/trackA.log
git log --oneline -1 | tee -a $OUT/trackA.log
step bpb-val 3600 python scripts/trackA_bpb.py --bolmo $B --olmo $O --supply $SUPPLY --split val
step prefill 5400 python scripts/trackA_bench.py --bolmo $B --olmo $O --supply $SUPPLY --mode prefill \
  --prompt-kib 8 16 --batches 1 4 --forced-p stock 4.4 6 8 --warmup 10 --repetitions 30 --seed 0
if [ $failures -gt 0 ]; then log "TRACKA_INCOMPLETE $failures failed"; exit 1; fi
log "TRACKA_DONE"
