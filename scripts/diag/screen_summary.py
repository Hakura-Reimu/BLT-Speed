"""Apply the pre-registered long-context screen rule (STATUS 2026-11-11 23:10; scripts/longctx_screen.sh header):
a model passes if greedy exact accuracy >= 0.90 at 8 KiB at every depth; if neither M-P6 nor M-BPE16K passes,
no long-context continuation is trained. S-P6, S-BPE50K and M-P6-nohash are diagnostics."""
import json, os, sys

root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/blt/results/longctx-screen")
runs = ["m-p6", "m-bpe16k", "s-p6", "s-bpe50k", "m-p6-nohash"]
summary = {"rule": "greedy exact >= 0.90 at 8 KiB at every depth (0.1, 0.5, 0.9); primary models m-p6, m-bpe16k",
           "models": {}}
for run in runs:
    path = f"{root}/{run}.json"
    if not os.path.exists(path):
        print(run, "MISSING"); continue
    data = json.load(open(path))
    cells = sorted(data["cells"], key=lambda c: (c["context_kib"], c["depth"]))
    rows = []
    for c in cells:
        rows.append({"kib": c["context_kib"], "depth": c["depth"], "n": c["count"],
                     "greedy": c["greedy_accuracy"], "rank1": c.get("rank1_accuracy"),
                     "mean_rank": c.get("mean_true_key_rank")})
    eight = {r["depth"]: r for r in rows if r["kib"] == 8}
    passes = (len([r for r in rows if r["kib"] == 8]) == 3 and set(eight) == {0.1, 0.5, 0.9}
              and all(r["greedy"] >= 0.90 for r in eight.values()))
    summary["models"][run] = {"cells": rows, "passes": passes}
    print(f"{run}: {'PASS' if passes else 'fail'}")
    for r in rows:
        print(f"   {r['kib']} KiB depth {r['depth']}: greedy {r['greedy']:.3f}  rank-1 {r['rank1']:.3f}  "
              f"mean rank {r['mean_rank']:.2f}  (n={r['n']})")
primary = [summary["models"].get(run, {}).get("passes", False) for run in ("m-p6", "m-bpe16k")]
summary["continuation"] = any(primary)
print("LONG_CONTEXT_CONTINUATION", "YES" if any(primary) else "NO (report capacity and memory only)")
json.dump(summary, open(f"{root}/summary.json", "w"), indent=1)
