# POST-HOC, EXPLORATORY (decided 2026-11-12 00:40 after the registered probe verdicts were read; Sol round 17):
# the same probes with the byte model M-P6-nohash (seed 0, 3 GB) against M-BPE16K seed 0, on exactly the registered
# example set, split, controls and classifier (scripts/probe.py defaults, as in scripts/probe_run.sh). Purpose: check
# the explanation that the registered controlled gains are negative because the byte input layer carries hash
# n-grams. It does not replace the registered verdict and is not a clean causal hash ablation (one seed, different
# training randomness). Run as a queue job: bash scripts/probe_posthoc_nohash.sh.
# Output: ~/blt/results/probes-posthoc/nohash-seed0.json and log.
set -u
cd ~/blt/code && source ~/blt/venv/bin/activate
export PYTHONPATH=src
OUT=~/blt/results/probes-posthoc; D=~/blt/data/probe-v1
mkdir -p $OUT/bad
python - <<'PY' || { echo "PROBE_DATA_CHANGED"; exit 1; }
import hashlib, json, os
root = os.path.expanduser("~/blt/data/probe-v1")
manifest = json.load(open(f"{root}/manifest.json"))
for name, digest in manifest["files"].items():
    with open(f"{root}/{name}", "rb") as handle:
        if hashlib.sha256(handle.read()).hexdigest() != digest:
            raise SystemExit(f"{name} differs from the manifest")
PY
out=$OUT/nohash-seed0.json
[ -f $out ] && exit 0
timeout 5400 python scripts/probe.py --byte-run ~/blt/runs/m-p6-nohash-seed0 --bpe-run ~/blt/runs/m-bpe16k-seed0 \
  --byte-checkpoint ~/blt/runs/m-p6-nohash-seed0/ckpt/bytes-3000000000.pt \
  --bpe-checkpoint ~/blt/runs/m-bpe16k-seed0/ckpt/bytes-3000000000.pt \
  --tokenizer ~/blt/data/tokenizer16k-v1/tokenizer.json --supply ~/blt/data/supply-v1 \
  --examples $D/examples.jsonl --robustness-text $D/robustness.txt \
  --state-cache $OUT/states-nohash-seed0.npz --device cuda --output $out.tmp > $OUT/nohash-seed0.log 2>&1
code=$?
if [ $code -eq 0 ] && [ -f $out.tmp ]; then mv $out.tmp $out; echo "ok"; else
  [ -f $out.tmp ] && mv $out.tmp $OUT/bad/nohash-seed0-$(date +%s).json; echo "FAILED exit $code"; exit 1; fi
