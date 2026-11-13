import collections, math, torch
from pathlib import Path
from blt.data import iter_eval_chunks
from blt.trackA import load_reference
from blt.router import _prefill_boundaries
bundle = load_reference("bolmo", Path.home() / "blt/models/Bolmo-1B")
pred = bundle.model.model.local_encoder.boundary_predictor_module
seen = []
h = pred.register_forward_hook(lambda m, a, out: seen.append(out[0].detach().float().cpu()))
docs = []
for chunk in iter_eval_chunks(Path.home() / "blt/data/supply-v1", "train"):
    if len(chunk.data) >= 2048:
        docs.append(chunk.data[:2048])
    if len(docs) == 4:
        break
for raw in docs:
    _prefill_boundaries(bundle.model, raw)
h.remove()
lp = torch.cat([s.reshape(-1) for s in seen])
valid = (lp < 0) & (lp > -1000)
v = lp[valid]
print("dtype of predictor output:", seen[0].dtype, "valid", int(valid.sum()), "of", lp.numel(), "exact zeros", int((lp == 0).sum()))
safe = v.clamp(min=-80, max=-torch.finfo(torch.float32).eps)
odds = safe - torch.log(-torch.expm1(safe))
counts = collections.Counter(round(float(x), 4) for x in odds)
print("most common boundary log-odds:", counts.most_common(6))
print("fraction with log-odds > 0 (boundary):", float((odds > 0).float().mean()))
for t in (5.0, 6.0, 6.2, 6.3, 7.0):
    print(f"fraction with log-odds > {t}:", round(float((odds > t).float().mean()), 4))
