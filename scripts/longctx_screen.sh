# Long-context screen (plan section 4 "long context" and risk 7): pass-key retrieval on the unmodified 3 GB checkpoints
# at 2, 4 and 8 KiB (depths 0.1, 0.5, 0.9; 200 trials per cell; scripts/retrieval_screen.py, prompt format v2 of
# 2026-11-11 23:30 in blt.retrieval). Run as a queue job: bash scripts/longctx_screen.sh. The rule, written in STATUS before this
# job ran: a model passes the screen if its greedy exact accuracy is >= 0.90 at 8 KiB at every depth; if neither
# M-P6 nor M-BPE16K passes, no long-context continuation is trained and only capacity and memory are reported.
# S-P6, S-BPE50K and M-P6-nohash are diagnostics. Outputs: ~/blt/results/longctx-screen/<run>.json and logs.
set -u
cd ~/blt/code && source ~/blt/venv/bin/activate
export PYTHONPATH=src
OUT=~/blt/results/longctx-screen
mkdir -p $OUT/bad
G=~/blt/models/gpt2-tokenizer/tokenizer.json; K=~/blt/data/tokenizer16k-v1/tokenizer.json
failures=0
log() { echo "UK $(date +%H:%M:%S) $*" | tee -a $OUT/screen.log; }
tokenizer() { case $1 in *bpe50k) echo $G ;; *bpe16k) echo $K ;; esac; }
nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu --format=csv,noheader | tee -a $OUT/screen.log
git log --oneline -1 | tee -a $OUT/screen.log
for run in m-p6 m-bpe16k s-p6 s-bpe50k m-p6-nohash; do
  out=$OUT/$run.json
  checkpoint=~/blt/runs/$run-seed0/ckpt/bytes-3000000000.pt
  check=(python scripts/diag/check_retrieval.py --checkpoint $checkpoint --kib 2 4 8 --trials 200 --seed 0)
  if [ -f $out ]; then
    "${check[@]}" $out 2>> $OUT/screen.log && continue
    log "invalid $run, measuring again"; mv $out $OUT/bad/$run-invalid-$(date +%s).json
  fi
  args=(--run ~/blt/runs/$run-seed0 --checkpoint $checkpoint
        --supply ~/blt/data/supply-v1 --kib 2 4 8 --trials 200 --seed 0 --device cuda)
  [ -n "$(tokenizer $run)" ] && args+=(--tokenizer "$(tokenizer $run)")
  log "start $run"
  timeout 2400 python scripts/retrieval_screen.py "${args[@]}" --output $out.tmp > $OUT/$run.log 2>&1
  code=$?
  if [ $code -eq 0 ] && [ -f $out.tmp ] && "${check[@]}" $out.tmp 2>> $OUT/screen.log; then
    mv $out.tmp $out; log "ok $run"
  else
    [ -f $out.tmp ] && mv $out.tmp $OUT/bad/$run-$(date +%s).json
    failures=$((failures + 1)); log "FAILED $run exit $code"
  fi
done
if [ $failures -gt 0 ]; then log "SCREEN_INCOMPLETE $failures failed"; exit 1; fi
log "SCREEN_DONE"
