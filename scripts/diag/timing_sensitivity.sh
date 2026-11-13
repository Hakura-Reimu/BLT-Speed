# One extra timing session with a fresh prompt seed (Sol round 15, STATUS 2026-11-11 09:45), reported separately from
# the three-sweep final timing, which stays primary. Same flags as scripts/final_timing.sh except --seed 1; each pair
# reuses the cache size frozen in the final timing's policy file, so nothing is recalibrated. bench_infer.py shrinks
# even a frozen size after a timed run over the memory budget and writes it back (Sol round 16), so this session
# works on a copy of the policy files: the final timing's own policies are never modified, and any reduction shows in
# the copy and in each result's graph_cache_policy.source. Three pairs at 8 and 16 KiB, with the S-P6/S-BPE50K
# anchor at the start and end. Run as a queue job: bash scripts/diag/timing_sensitivity.sh.
# Outputs: ~/blt/results/final-timing-seed1/{anchor,s1}/<item>.json.
set -u
cd ~/blt/code && source ~/blt/venv/bin/activate
OUT=~/blt/results/final-timing-seed1
FROZEN=~/blt/results/final-timing/policy
POLICY=$OUT/policy
mkdir -p $OUT/bad $POLICY
for pair in s-p6:s-bpe50k m-p6:m-bpe16k m-p6:xs-bpe50k; do
  f=${pair%%:*}--${pair##*:}.json
  [ -f $POLICY/$f ] || cp $FROZEN/$f $POLICY/$f
done
G=~/blt/models/gpt2-tokenizer/tokenizer.json; K=~/blt/data/tokenizer16k-v1/tokenizer.json
REPS=100
FLAGS=(--mode prefill --batches 1 --execution graph --warmup 128 --repetitions $REPS --seed 1 --capacity bucket64
       --memory-budget-bytes 10500000000 --bpe-encoder tiktoken --byte-e2e-patching device
       --bf16-reduced-precision-reduction false --supply ~/blt/data/supply-v1)
PAIRS="s-p6:s-bpe50k m-p6:m-bpe16k m-p6:xs-bpe50k"
failures=0
log() { echo "UK $(date +%H:%M:%S) $*" | tee -a $OUT/timing.log; }
ckpt() { echo ~/blt/runs/$1-seed0/ckpt/bytes-3000000000.pt; }
tokenizer() { case $1 in *bpe50k) echo $G ;; *bpe16k) echo $K ;; esac; }
check() { python scripts/diag/timing_summary.py check "$@"; }

measure() {  # measure SWEEP NAME A B: a pair with A first, under the frozen policy of the pair A--B
  local sweep=$1 name=$2 a=$3 b=$4
  local out=$OUT/$sweep/$name.json args
  local expect=(--kib 8 16 --warmup 128 --repetitions $REPS --runs $a $b)
  if [ -f $out ]; then
    check $out "${expect[@]}" > /dev/null && return
    log "invalid $sweep/$name, measuring again"; mv $out $OUT/bad/$sweep-$name-$(date +%s).json
  fi
  args=(--compare ~/blt/runs/$a-seed0 ~/blt/runs/$b-seed0 --checkpoint-a "$(ckpt $a)" --checkpoint-b "$(ckpt $b)")
  [ -n "$(tokenizer $a)" ] && args+=(--tokenizer "$(tokenizer $a)")
  [ -n "$(tokenizer $b)" ] && args+=(--tokenizer-b "$(tokenizer $b)")
  mkdir -p $OUT/$sweep
  log "start $sweep $name ($a $b)"
  python scripts/bench_infer.py "${args[@]}" "${FLAGS[@]}" --prompt-kib 8 16 --output $out.tmp \
    --cache-policy $POLICY/$a--$b.json > $OUT/$sweep/$name.log 2>&1
  local code=$?
  if [ -f $out.tmp ] && check $out.tmp "${expect[@]}" >> $OUT/$sweep/$name.log 2>&1; then
    mv $out.tmp $out; log "ok $sweep $name exit $code"
  else
    [ -f $out.tmp ] && mv $out.tmp $OUT/bad/$sweep-$name-$(date +%s).json
    failures=$((failures + 1)); log "FAILED $sweep $name exit $code"
  fi
}

for pair in $PAIRS; do
  a=${pair%%:*}; b=${pair##*:}
  for run in $a $b; do [ -f "$(ckpt $run)" ] || { log "preflight: no 3 GB checkpoint for $run"; exit 1; }; done
  [ -f $POLICY/$a--$b.json ] || { log "preflight: no frozen policy for $a--$b"; exit 1; }
done
nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu --format=csv,noheader | tee -a $OUT/timing.log
git log --oneline -1 | tee -a $OUT/timing.log
measure anchor s1-start s-p6 s-bpe50k
for pair in $PAIRS; do a=${pair%%:*}; b=${pair##*:}; measure s1 $a--$b $a $b; done
measure anchor s1-end s-p6 s-bpe50k
for f in $POLICY/*.json; do cmp -s $f $FROZEN/$(basename $f) || log "policy changed in this session: $(basename $f)"; done
if [ $failures -gt 0 ]; then log "TIMING_SENSITIVITY_INCOMPLETE $failures failed"; exit 1; fi
log "TIMING_SENSITIVITY_DONE"
