# Final paired prefill timing of the trained 3 GB checkpoints (protocol of Sol rounds 9-10 and 14, STATUS 2026-11-10).
# Run as a queue job (bash scripts/final_timing.sh), so nothing else holds the GPU and the daemon's contention check
# pauses it when another program needs the GPU.
#   pairs:    every byte model against every BPE model, plus M-P6 against M-P6-nohash (hash cost);
#   solo:     every model alone under the same memory policy, for absolute throughput on the frontier;
#   cells:    end-to-end and GPU-resident prefill at 8 and 16 KiB, batch 1, CUDA graphs; the frozen preprocessing and
#             numerics are passed explicitly (tiktoken, device patching for bytes, fp32 GEMM reductions, bucket64);
#   requests: 128 calibration requests fill the graph cache (both models resident), whose membership is then frozen;
#             100 timed requests in alternating order; misses run eager inside the timing and count;
#   sweeps:   three, each a separate process per item over the same prompts (seed 0), items in a seeded random order,
#             with the S-P6/S-BPE50K anchor at the start and end of each sweep for drift; sweep 2 swaps which model of
#             a pair starts; the cache size per cell is calibrated once and frozen in the pair's or model's policy file;
#   curve:    one extra sweep at 2, 4 and 32 KiB for four same-trunk or equal-quality pairs.
# An output is kept only if timing_summary.py check accepts it (every cell present with the expected trials, or a
# memory-budget error, which is deterministic); anything else is moved to bad/ and the job exits nonzero, so the
# queue retries it and the rerun skips what is already valid. Outputs: ~/blt/results/final-timing/<sweep>/<item>.json.
set -u
cd ~/blt/code && source ~/blt/venv/bin/activate
OUT=~/blt/results/final-timing
mkdir -p $OUT/policy $OUT/bad
G=~/blt/models/gpt2-tokenizer/tokenizer.json; K=~/blt/data/tokenizer16k-v1/tokenizer.json
BYTE="s-p6 m-p6 m-p8 s-p8 m-p6-nohash"
BPE="xs-bpe50k xs-bpe16k s-bpe50k s-bpe16k m-bpe16k m-bpe50k"
REPS=100
FLAGS=(--mode prefill --batches 1 --execution graph --warmup 128 --repetitions $REPS --seed 0 --capacity bucket64
       --memory-budget-bytes 10500000000 --bpe-encoder tiktoken --byte-e2e-patching device
       --bf16-reduced-precision-reduction false --supply ~/blt/data/supply-v1)
failures=0
log() { echo "UK $(date +%H:%M:%S) $*" | tee -a $OUT/timing.log; }
ckpt() { echo ~/blt/runs/$1-seed0/ckpt/bytes-3000000000.pt; }
tokenizer() { case $1 in *bpe50k) echo $G ;; *bpe16k) echo $K ;; esac; }
check() { python scripts/diag/timing_summary.py check "$@"; }

measure() {  # measure SWEEP NAME POLICY "KIB..." RUN [RUN]: one run is solo, two are a pair with the first as A
  local sweep=$1 name=$2 policy=$3 kibs=$4; shift 4
  local out=$OUT/$sweep/$name.json args
  local expect=(--kib $kibs --warmup 128 --repetitions $REPS --runs "$@")
  if [ -f $out ]; then
    check $out "${expect[@]}" > /dev/null && return
    log "invalid $sweep/$name, measuring again"; mv $out $OUT/bad/$sweep-$name-$(date +%s).json
  fi
  if [ $# -eq 1 ]; then
    args=(--run ~/blt/runs/$1-seed0 --checkpoint "$(ckpt $1)")
    [ -n "$(tokenizer $1)" ] && args+=(--tokenizer "$(tokenizer $1)")
  else
    args=(--compare ~/blt/runs/$1-seed0 ~/blt/runs/$2-seed0 --checkpoint-a "$(ckpt $1)" --checkpoint-b "$(ckpt $2)")
    [ -n "$(tokenizer $1)" ] && args+=(--tokenizer "$(tokenizer $1)")
    [ -n "$(tokenizer $2)" ] && args+=(--tokenizer-b "$(tokenizer $2)")
  fi
  mkdir -p $OUT/$sweep
  log "start $sweep $name ($*)"
  python scripts/bench_infer.py "${args[@]}" "${FLAGS[@]}" --prompt-kib $kibs --output $out.tmp \
    --cache-policy $OUT/policy/$policy.json > $OUT/$sweep/$name.log 2>&1
  local code=$?
  if [ -f $out.tmp ] && check $out.tmp "${expect[@]}" >> $OUT/$sweep/$name.log 2>&1; then
    mv $out.tmp $out; log "ok $sweep $name exit $code"
  else
    [ -f $out.tmp ] && mv $out.tmp $OUT/bad/$sweep-$name-$(date +%s).json
    failures=$((failures + 1)); log "FAILED $sweep $name exit $code"
  fi
}

for run in $BYTE $BPE; do
  [ -f "$(ckpt $run)" ] || { log "preflight: no 3 GB checkpoint for $run"; exit 1; }
done
python - "$OUT" "${FLAGS[@]}" <<'PY'
from datetime import datetime, timezone
import hashlib, json, subprocess, sys
from pathlib import Path
files = ("scripts/final_timing.sh", "scripts/bench_infer.py", "src/blt/bench.py", "scripts/diag/timing_summary.py")
stamp = datetime.now(timezone.utc)
manifest = {"started": stamp.isoformat(), "flags": sys.argv[2:],
            "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "sha256": {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}
Path(sys.argv[1], f"manifest-{stamp:%Y%m%dT%H%M%S}.json").write_text(json.dumps(manifest, indent=1) + "\n")
PY
nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu --format=csv,noheader | tee -a $OUT/timing.log
git log --oneline -1 | tee -a $OUT/timing.log

items=()
for a in $BYTE; do for b in $BPE; do items+=("$a:$b"); done; done
items+=("m-p6:m-p6-nohash")
for m in $BYTE $BPE; do items+=("$m"); done
for sweep in 1 2 3; do
  measure anchor s$sweep-start s-p6--s-bpe50k "8 16" s-p6 s-bpe50k
  for item in $(python -c "import random, sys; x = sys.argv[2:]; random.Random(int(sys.argv[1])).shuffle(x); print(*x)" $sweep "${items[@]}"); do
    if [[ $item == *:* ]]; then
      a=${item%%:*}; b=${item##*:}
      if [ $sweep -eq 2 ]; then measure s2 $a--$b $a--$b "8 16" $b $a; else measure s$sweep $a--$b $a--$b "8 16" $a $b; fi
    else
      measure s$sweep solo-$item solo-$item "8 16" $item
    fi
  done
  measure anchor s$sweep-end s-p6--s-bpe50k "8 16" s-p6 s-bpe50k
done
for spec in "s-p6 xs-bpe50k" "s-p6 s-bpe50k" "m-p6 m-bpe16k" "m-p8 m-bpe16k"; do
  set -- $spec; measure curve $1--$2 $1--$2 "2 4 32" $1 $2
done
if [ $failures -gt 0 ]; then log "FINAL_TIMING_INCOMPLETE $failures failed"; exit 1; fi
log "FINAL_TIMING_DONE"
