# Registered decode measurement of the trained 3 GB checkpoints (plan section 4 "decode"; secondary claims "batched
# decode" and "batch-1 latency"), with the cells frozen in ~/blt/results/frozen-analysis.json ("decode") before any
# decode result was read (STATUS 2026-11-11 18:50). Speculative decoding is out of scope (user, 10:45). Run as a queue
# job: bash scripts/decode_timing.sh. Uses the decode benchmark of Terra task 12 (dbd956f):
#   gate:     the CUDA decode tests (graph and graph-split parity on distinct rows) must pass first;
#   primary:  M-P6 against M-BPE16K, 8 KiB prompts, batches 1 and 8, 256 requested bytes per row, three sessions
#             over the same ordered prompt batches; in each session every non-speculative mode of each model is
#             checked for greedy parity (>= 512 bytes per row) and calibrated on separate prompts, and each model's
#             fastest mode is then timed head-to-head (10 warm-ups, 30 requests, alternating order);
#   diagnostic (one session each): the same pair at batch 32 and at 2 KiB; S-P6/S-BPE50K and the exploratory
#             M-P6/XS-BPE50K at 8 KiB, batches 1, 8 and 32.
# An output is kept only if decode_summary.py check accepts it; otherwise it moves to bad/ and the job exits nonzero,
# so the queue retries and the rerun skips what is already valid. Every step has a time limit (about 2 GPU hours in
# all). Outputs: ~/blt/results/decode/{primary/s1,s2,s3,diag}/<a>--<b>-<kib>k.json; summary in primary/summary.md.
set -u
cd ~/blt/code && source ~/blt/venv/bin/activate
export PYTHONPATH=src
OUT=~/blt/results/decode
mkdir -p $OUT/bad
G=~/blt/models/gpt2-tokenizer/tokenizer.json; K=~/blt/data/tokenizer16k-v1/tokenizer.json
failures=0
log() { echo "UK $(date +%H:%M:%S) $*" | tee -a $OUT/decode.log; }
ckpt() { echo ~/blt/runs/$1-seed0/ckpt/bytes-3000000000.pt; }
tokenizer() { case $1 in *bpe50k) echo $G ;; *bpe16k) echo $K ;; esac; }

measure() {  # measure DIR A B KIB "BATCHES" SECONDS
  local dir=$1 a=$2 b=$3 kib=$4 batches=$5 limit=$6
  local name=$a--$b-${kib}k out
  out=$OUT/$dir/$name.json
  local expect=(--runs $a $b --batches $batches --kib $kib --repetitions 30)
  if [ -f $out ]; then
    python scripts/diag/decode_summary.py check $out "${expect[@]}" > /dev/null && return
    log "invalid $dir/$name, measuring again"; mv $out $OUT/bad/$(echo $dir | tr / -)-$name-$(date +%s).json
  fi
  local args=(--compare ~/blt/runs/$a-seed0 ~/blt/runs/$b-seed0 --checkpoint-a "$(ckpt $a)" --checkpoint-b "$(ckpt $b)")
  [ -n "$(tokenizer $a)" ] && args+=(--tokenizer "$(tokenizer $a)")
  [ -n "$(tokenizer $b)" ] && args+=(--tokenizer-b "$(tokenizer $b)")
  mkdir -p $OUT/$dir
  log "start $dir $name batches $batches"
  timeout $limit python scripts/bench_decode.py "${args[@]}" --supply ~/blt/data/supply-v1 --mode all \
    --prompt-kib $kib --batches $batches --output-bytes 256 --warmup 10 --calibration-requests 10 --repetitions 30 \
    --verify-bytes 512 --memory-budget-bytes 10500000000 --seed 0 --output $out.tmp > $OUT/$dir/$name.log 2>&1
  local code=$?
  if [ -f $out.tmp ] && python scripts/diag/decode_summary.py check $out.tmp "${expect[@]}" >> $OUT/$dir/$name.log 2>&1; then
    mv $out.tmp $out; log "ok $dir $name exit $code"
  else
    [ -f $out.tmp ] && mv $out.tmp $OUT/bad/$(echo $dir | tr / -)-$name-$(date +%s).json
    failures=$((failures + 1)); log "FAILED $dir $name exit $code"
  fi
}

for run in m-p6 m-bpe16k s-p6 s-bpe50k xs-bpe50k; do
  [ -f "$(ckpt $run)" ] || { log "preflight: no 3 GB checkpoint for $run"; exit 1; }
done
nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu --format=csv,noheader | tee -a $OUT/decode.log
git log --oneline -1 | tee -a $OUT/decode.log
if [ ! -f $OUT/cuda-tests.ok ]; then
  log "CUDA decode tests"
  if timeout 1800 python -m pytest -q -p no:cacheprovider tests/test_decode.py tests/test_decode_summary.py \
      tests/test_decode_rope_tables.py > $OUT/cuda-tests.log 2>&1; then
    tail -n 1 $OUT/cuda-tests.log > $OUT/cuda-tests.ok; log "CUDA decode tests passed: $(cat $OUT/cuda-tests.ok)"
  else
    log "DECODE_GATE_FAILED: CUDA decode tests, see cuda-tests.log"; exit 1
  fi
fi
for session in s1 s2 s3; do measure primary/$session m-p6 m-bpe16k 8 "1 8" 1500; done
if [ $failures -eq 0 ]; then
  python scripts/diag/decode_summary.py summary $OUT/primary > $OUT/primary/summary.md 2>> $OUT/decode.log \
    && log "primary summary written" || { log "FAILED primary summary"; failures=$((failures + 1)); }
fi
measure diag m-p6 m-bpe16k 8 32 1200
measure diag m-p6 m-bpe16k 2 "1 8 32" 1200
measure diag s-p6 s-bpe50k 8 "1 8 32" 1200
measure diag m-p6 xs-bpe50k 8 "1 8 32" 1200
if [ $failures -gt 0 ]; then log "DECODE_INCOMPLETE $failures failed"; exit 1; fi
log "DECODE_DONE"
