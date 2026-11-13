"""CPU diagnostic for a trained byte checkpoint: NLL by position, branch RMS, gradient flow.

usage: python diag_byte.py RUN_DIR SUPPLY N_WINDOWS OUT_JSON
"""
import json
import math
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from blt.data import iter_eval_chunks
from blt.eval_io import load_evaluation_model
from blt.hashes import hashes_at_ends
from blt.patcher import patch_batch

torch.set_num_threads(4)
run, supply, n_windows, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
bundle = load_evaluation_model(run=run, device="cpu", supply=supply)
model, c = bundle.model.float().eval(), bundle.model.config
DELIM = set(b" \t\n\r.,;:!?\"'()[]{}-/")

groups = defaultdict(lambda: [0.0, 0])  # name -> [nats, count]
rms = defaultdict(list)
total_nats = total_bytes = 0
windows = []
for i, chunk in enumerate(iter_eval_chunks(supply, "val", 8192)):
    if i >= n_windows:
        break
    windows.append(chunk)
    raw = torch.tensor(list(chunk.data), dtype=torch.long)[None, :]
    patches = patch_batch(raw.numpy().astype(np.uint8), c.min_len, c.max_len, c.block_size)
    boundaries = patches.boundaries[0]
    with torch.no_grad():
        logits = model(raw)
        nll = F.cross_entropy(logits.float()[0], raw[0], reduction="none").numpy()
        # Branch magnitudes, recomputed exactly as _encode_global / _full_hidden do.
        ends = torch.as_tensor(patches.ends, dtype=torch.long)
        valid = ends[0] >= 0
        enc, _ = model._local(model.byte_embedding(raw), model.encoder)
        selected = enc.gather(1, ends.clamp_min(0).unsqueeze(-1).expand(-1, -1, c.local_width))
        pooled = model.pool(selected)[0][valid]
        rms["encoder_out"].append(enc.pow(2).mean().sqrt().item())
        rms["pooled"].append(pooled.pow(2).mean().sqrt().item())
        if model.hash_embedding is not None:
            ids = hashes_at_ends(raw, ends, c.hash_rows, c.hash_seed,
                                 tables=(model.hash_prefixes, model.hash_powers, model.hash_ngrams,
                                         model.hash_window_offsets, model.hash_coefficients))
            hashed = model._hash_vectors(ids)[0][valid]
            rms["hash"].append(hashed.pow(2).mean().sqrt().item())
            cos = F.cosine_similarity(pooled, hashed, dim=-1)
            rms["cos_pooled_hash"].append(cos.mean().item())
        _, _, global_states, _ = model._encode_global(raw, ends)
        ids_t = torch.as_tensor(patches.patch_ids, dtype=torch.long)
        cond = model.condition(global_states.gather(1, ids_t.unsqueeze(-1).expand(-1, -1, c.trunk.width)))
        previous = torch.cat((model.local_bos.expand(1, 1, -1), enc[:, :-1]), dim=1)
        rms["decoder_local_in"].append(previous.pow(2).mean().sqrt().item())
        rms["decoder_global_in"].append(cond.pow(2).mean().sqrt().item())
    last = -1
    data = chunk.data
    for t in range(len(data)):
        offset = t - last - 1
        keys = [f"patch_offset={min(offset, 8)}", f"block_offset={'0' if t % 64 == 0 else '1-7' if t % 64 < 8 else '8+'}",
                f"target_delim={data[t] in DELIM}", f"prev_delim={t > 0 and data[t - 1] in DELIM}",
                f"first_byte_of_patch_and_target_delim={(offset == 0, data[t] in DELIM)}"]
        if chunk.start == 0 and t < 256:
            keys.append("document_first_256")
        for key in keys:
            groups[key][0] += float(nll[t])
            groups[key][1] += 1
        if boundaries[t]:
            last = t
    total_nats += float(nll.sum())
    total_bytes += len(data)

# One backward pass on the first window: gradient RMS by module group.
model.train()
raw = torch.tensor(list(windows[0].data), dtype=torch.long)[None, :]
loss = F.cross_entropy(model(raw).float()[0], raw[0])
loss.backward()
grads = defaultdict(list)
for name, p in model.named_parameters():
    if p.grad is not None:
        g = p.grad
        grads[name.split(".")[0]].append((g.pow(2).sum().item(), g.numel(), bool(torch.isfinite(g).all())))
grad_rms = {k: {"rms": math.sqrt(sum(s for s, _, _ in v) / sum(n for _, n, _ in v)), "finite": all(f for _, _, f in v)}
            for k, v in grads.items()}
weights = {name: p.detach().pow(2).mean().sqrt().item() for name, p in model.named_parameters()
           if name in ("pool.weight", "condition.weight", "hash_embedding.weight", "byte_embedding.weight", "head.weight")}

result = {
    "run": run, "windows": len(windows), "bytes": total_bytes,
    "bpb": total_nats / (math.log(2) * total_bytes),
    "groups": {k: {"bpb": v[0] / (math.log(2) * v[1]), "share_bytes": v[1] / total_bytes, "share_nats": v[0] / total_nats}
               for k, v in sorted(groups.items())},
    "rms": {k: float(np.mean(v)) for k, v in rms.items()},
    "weight_rms": weights, "grad_rms": grad_rms,
}
json.dump(result, open(out, "w"), indent=1)
print(json.dumps({k: result[k] for k in ("bpb", "rms", "weight_rms")}, indent=1))
for k, v in result["groups"].items():
    print(f"{k:55s} bpb {v['bpb']:.3f}  bytes {v['share_bytes']:.3f}  nats {v['share_nats']:.3f}")
print(json.dumps(grad_rms, indent=1))
