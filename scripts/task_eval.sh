# Registered task evaluation (plan section 5, "task retention"): zero-shot PIQA and HellaSwag on their full validation
# splits (PIQA 1838, HellaSwag 10042 examples), every candidate scored by its continuation log-likelihood with the
# context and continuation encoded separately (src/blt/tasks.py), reporting raw and byte-normalised accuracy. Every
# 3 GB checkpoint is evaluated: the 11 seed-0 models and the two seed-1 repeats once they exist. Run as a queue job:
# bash scripts/task_eval.sh. An output is kept only if eval_tasks.py exits 0; a rerun skips kept outputs.
# Outputs: ~/blt/results/tasks/<run>-<task>.json.
set -u
cd ~/blt/code && source ~/blt/venv/bin/activate
OUT=~/blt/results/tasks
mkdir -p $OUT/bad
G=~/blt/models/gpt2-tokenizer/tokenizer.json; K=~/blt/data/tokenizer16k-v1/tokenizer.json
RUNS="m-bpe16k-seed0 m-p6-seed0 s-bpe50k-seed0 s-p6-seed0 xs-bpe50k-seed0 xs-bpe16k-seed0 s-bpe16k-seed0 m-bpe50k-seed0
      s-p8-seed0 m-p8-seed0 m-p6-nohash-seed0 m-bpe16k-seed1 m-p6-seed1"
failures=0
log() { echo "UK $(date +%H:%M:%S) $*" | tee -a $OUT/tasks.log; }
tokenizer() { case $1 in *bpe50k*) echo $G ;; *bpe16k*) echo $K ;; esac; }

for run in $RUNS; do
  checkpoint=~/blt/runs/$run/ckpt/bytes-3000000000.pt
  [ -f $checkpoint ] || { log "FAILED $run: no 3 GB checkpoint"; failures=$((failures + 1)); continue; }
  for task in piqa hellaswag; do
    out=$OUT/$run-$task.json
    [ -f $out ] && continue
    args=(--run ~/blt/runs/$run --checkpoint $checkpoint --task $task --dataset ~/blt/data/eval/$task
          --limit 100000 --seed 0 --output $out.tmp)
    [ -n "$(tokenizer $run)" ] && args+=(--tokenizer "$(tokenizer $run)")
    log "start $run $task"
    python scripts/eval_tasks.py "${args[@]}" > $OUT/$run-$task.log 2>&1
    code=$?
    if [ $code -eq 0 ] && [ -f $out.tmp ]; then
      mv $out.tmp $out
      log "ok $run $task $(python -c "import json; r = json.load(open('$out')); print(r['count'], round(r['accuracy'], 4), round(r['accuracy_byte_normalised'], 4))")"
    else
      [ -f $out.tmp ] && mv $out.tmp $OUT/bad/$run-$task-$(date +%s).json
      failures=$((failures + 1)); log "FAILED $run $task exit $code"
    fi
  done
done
if [ $failures -gt 0 ]; then log "TASKS_INCOMPLETE $failures failed"; exit 1; fi
log "TASKS_DONE"
