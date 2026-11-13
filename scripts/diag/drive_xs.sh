# Drive the exploratory XS-BPE control through select_lr.py's procedure without manual steps (run inside WSL:
# bash -l < scripts/diag/drive_xs.sh, after `xs_bpe.py screen` and `enqueue` of the screen jobs).
# screen scores -> stage and queue confirmations -> (edge extension, if the rules stage one) -> learning rates ->
# stage and queue the two 3 GB mains -> queue their milestone scores as they appear. Stops at the first failed job.
cd ~/blt/code && source ~/blt/venv/bin/activate
log() { echo "UK $(date +%H:%M) $*"; }
failed() { ls ~/blt/queue/failed | grep -q . && { log "failed jobs:"; ls ~/blt/queue/failed; exit 1; }; }
scored() {  # every training job staged in $1 has a score in $2
  for f in "$1"/*.json; do
    python -c "import json,sys; sys.exit(0 if 'scripts/train.py' in json.load(open(sys.argv[1])).get('command', []) else 1)" "$f" || continue
    [ -f "$2/$(basename "$f")" ] || return 1
  done
}
until scored ~/blt/jobs/staging/lr-pilot-xs ~/blt/results/lr-screen-xs; do failed; sleep 60; done
log "screen scored"; python scripts/diag/xs_bpe.py select screen && python scripts/diag/xs_bpe.py enqueue ~/blt/jobs/staging/lr-confirm-xs || exit 1
until [ -f ~/blt/results/learning_rates_xs.json ]; do
  until scored ~/blt/jobs/staging/lr-confirm-xs ~/blt/results/lr-confirm-xs; do failed; sleep 60; done
  log "confirmations scored"; python scripts/diag/xs_bpe.py select confirm || exit 1
  [ -f ~/blt/results/learning_rates_xs.json ] || python scripts/diag/xs_bpe.py enqueue ~/blt/jobs/staging/lr-confirm-xs
done
python scripts/diag/xs_bpe.py main && python scripts/diag/xs_bpe.py enqueue ~/blt/jobs/staging/xs-bpe || exit 1
until [ -f ~/blt/results/main-scores/xs-bpe50k-seed0-bytes-3000000000.json ] && [ -f ~/blt/results/main-scores/xs-bpe16k-seed0-bytes-3000000000.json ]; do
  for r in xs-bpe50k-seed0 xs-bpe16k-seed0; do
    if { [ -f ~/blt/queue/running/$r-main.json ] || [ -f ~/blt/queue/done/$r-main.json ]; } && [ -d ~/blt/runs/$r/ckpt ]; then
      RUN=$r bash scripts/diag/queue_milestone_scores.sh | grep '^queued' | sed "s/^/UK $(date +%H:%M) /"
    fi
  done
  failed; sleep 60
done
log "XS mains scored"
