# Track A2 bounded router programming (design: STATUS 2026-11-12 04:50; scripts/router_bounded.py). Queue job:
# bash scripts/router_run.sh. One hour cap. Output ~/blt/results/router/a2.json and log; a failed run leaves its
# partial JSON in bad/.
set -u
cd ~/blt/code && source ~/blt/venv/bin/activate
export PYTHONPATH=src
OUT=~/blt/results/router
mkdir -p $OUT/bad
out=$OUT/a2.json
[ -f $out ] && exit 0
git log --oneline -1 > $OUT/a2.log
timeout 3600 python scripts/router_bounded.py --bolmo ~/blt/models/Bolmo-1B --supply ~/blt/data/supply-v1 \
  --trackA-bpb ~/blt/results/trackA/bpb-val.json --output $out.tmp >> $OUT/a2.log 2>&1
code=$?
if [ $code -eq 0 ] && [ -f $out.tmp ]; then mv $out.tmp $out; echo ok; else
  [ -f $out.tmp ] && mv $out.tmp $OUT/bad/a2-$(date +%s).json; echo "FAILED exit $code"; exit 1; fi
