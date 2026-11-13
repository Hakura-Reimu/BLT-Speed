"""Frozen-backbone readout probe (Sol rounds 11-12): does the global state hold information the local decoder misses?

The checkpoint is frozen. Zero-initialised additive readouts are fitted on the same unseen training windows (the run's
own sampler, continuing at the checkpoint's committed sampler cursor; the sampler is a per-epoch permutation, so these
windows are distinct from every window the checkpoint consumed):
  decoder: logits = head(dec) + dec @ W_d.T                (control: more training of a decoder-state readout)
  global:  logits = head(dec) + cond @ W_g.T               (cond = the gathered global state the decoder sees)
  joint:   logits = head(dec) + dec @ W_d.T + cond @ W_g.T (nested with decoder: joint minus decoder is the gain
                                                            from global information given a refitted decoder readout)
Each kind is fitted at every learning rate in --lrs. Full-validation BPB is reported for the frozen baseline and each
readout, overall and split by the first byte of a patch versus later bytes, with per-chunk nats so paired gains get
document-clustered bootstrap intervals. The baseline must match the official score.

usage: python frozen_readout.py RUN_DIR CHECKPOINT SUPPLY OUT_JSON [--first-index N] [--train-windows 16384]
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from blt.data import WindowIndex, iter_eval_chunks, sample_window_index
from blt.eval_io import load_evaluation_model
from blt.score import autocast_for, configure_matmul_precision

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("checkpoint")
parser.add_argument("supply")
parser.add_argument("out")
parser.add_argument("--first-index", type=int, default=None, help="optional; must equal the checkpoint's sampler cursor")
parser.add_argument("--train-windows", type=int, default=16384)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--lrs", default="3e-4,1e-3")
parser.add_argument("--kinds", default="decoder,global,joint")
parser.add_argument("--bootstrap", type=int, default=2000)
parser.add_argument("--device", default="cuda")
parser.add_argument("--max-val-chunks", type=int, default=None, help="smoke tests only")
args = parser.parse_args()

device = torch.device(args.device)
configure_matmul_precision()
payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
training, cursor = payload["config"], payload["sampler_cursor"]
del payload
if args.first_index is not None and args.first_index != cursor:
    raise SystemExit(f"--first-index {args.first_index} differs from the checkpoint's sampler cursor {cursor}")
bundle = load_evaluation_model(run=args.run, checkpoint=args.checkpoint, device=device, supply=args.supply)
model, c = bundle.model.eval(), bundle.model.config
for parameter in model.parameters():
    parameter.requires_grad_(False)


def pack(rows: list[bytes]) -> tuple[torch.Tensor, torch.Tensor]:
    # Right padding is causally irrelevant to earlier positions; padded targets are ignored.
    length = max(len(row) for row in rows)
    raw = np.zeros((len(rows), length), dtype=np.int64)
    targets = np.full((len(rows), length), -100, dtype=np.int64)
    for i, row in enumerate(rows):
        values = np.frombuffer(row, dtype=np.uint8)
        raw[i, :len(values)] = values
        targets[i, :len(values)] = values
    return torch.from_numpy(raw).to(device), torch.from_numpy(targets).to(device)


@torch.no_grad()
def features(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with autocast_for(device):
        dec, info = model._full_hidden(raw)
        base = model.head(dec)
    # patch_ids[t] counts boundaries strictly before t, so byte t opens a patch exactly where the count steps up.
    ids = torch.as_tensor(info["patches"].patch_ids, device=device)
    first = torch.ones_like(ids, dtype=torch.bool)
    first[:, 1:] = ids[:, 1:] != ids[:, :-1]
    return dec.float(), info["cond"].float(), base.float(), first


def nats(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
    flat = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100,
                           reduction=reduction)
    return flat if reduction == "sum" else flat.reshape(targets.shape)


def logits(name: str, dec: torch.Tensor, cond: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    if name == "baseline":
        return base
    kind, _, weights, _ = readouts[name]
    out = base
    if "decoder" in weights:
        out = out + dec @ weights["decoder"].T
    if "global" in weights:
        out = out + cond @ weights["global"].T
    return out


lrs = [float(value) for value in args.lrs.split(",")]
sources = {"decoder": ("decoder",), "global": ("global",), "joint": ("decoder", "global")}
widths = {"decoder": c.local_width, "global": c.trunk.width}
readouts = {}
for kind in args.kinds.split(","):
    for lr in lrs:
        weights = {source: torch.zeros(c.output_classes, widths[source], device=device, requires_grad=True)
                   for source in sources[kind]}
        readouts[f"{kind}@{lr:g}"] = (kind, lr, weights, torch.optim.AdamW(weights.values(), lr=lr, weight_decay=0.0))


@torch.no_grad()
def evaluate() -> dict:
    names = ["baseline", *readouts]
    chunks = list(iter_eval_chunks(args.supply, "val", 8192))[:args.max_val_chunks]
    per_chunk = {name: np.zeros(len(chunks)) for name in names}
    split = {name: {"first": 0.0, "later": 0.0} for name in names}
    split_bytes = {"first": 0, "later": 0}
    for start in range(0, len(chunks), args.batch):
        raw, targets = pack([chunk.data for chunk in chunks[start:start + args.batch]])
        dec, cond, base, first = features(raw)
        real = targets >= 0
        split_bytes["first"] += int((first & real).sum())
        split_bytes["later"] += int((~first & real).sum())
        for name in names:
            values = nats(logits(name, dec, cond, base), targets, "none")
            per_chunk[name][start:start + len(raw)] = values.sum(1).cpu().numpy()
            split[name]["first"] += values[first & real].sum().item()
            split[name]["later"] += values[~first & real].sum().item()
    sizes = np.array([len(chunk.data) for chunk in chunks])
    documents = [chunk.document for chunk in chunks]
    groups = {}
    for i, document in enumerate(documents):
        groups.setdefault(document, []).append(i)
    members = list(groups.values())
    group_bytes = np.array([sizes[m].sum() for m in members])
    group_nats = {name: np.array([values[m].sum() for m in members]) for name, values in per_chunk.items()}
    draws = np.random.default_rng(0).integers(0, len(members), (args.bootstrap, len(members)))

    def gain(reference: str, name: str) -> dict:
        point = (per_chunk[reference].sum() - per_chunk[name].sum()) / (math.log(2) * sizes.sum())
        boot = (group_nats[reference][draws].sum(1) - group_nats[name][draws].sum(1)) / (
            math.log(2) * group_bytes[draws].sum(1))
        low, high = np.percentile(boot, [2.5, 97.5])
        return {"gain": point, "ci95": [low, high]}

    bpb = {name: values.sum() / (math.log(2) * sizes.sum()) for name, values in per_chunk.items()}
    gains = {name: gain("baseline", name) for name in readouts}
    for lr in lrs:
        if f"joint@{lr:g}" in readouts and f"decoder@{lr:g}" in readouts:
            gains[f"joint-minus-decoder@{lr:g}"] = gain(f"decoder@{lr:g}", f"joint@{lr:g}")
    split_bpb = {name: {part: value / (math.log(2) * split_bytes[part]) for part, value in parts.items()}
                 for name, parts in split.items()}
    return {"val_bytes": int(sizes.sum()), "val_bpb": bpb, "gains": gains, "split_bytes": split_bytes,
            "split_bpb": split_bpb, "per_chunk_nats": {name: values.tolist() for name, values in per_chunk.items()},
            "chunk_bytes": sizes.tolist(), "chunk_documents": documents}


steps = math.ceil(args.train_windows / args.batch)
curves = {name: [] for name in readouts}
began = time.time()
with WindowIndex(Path(args.supply) / "train.bin", training["window_bytes"]) as windows:
    for step in range(steps):
        rows = [windows[sample_window_index(training["data_seed"], cursor + step * args.batch + i, 0, 1,
                                            len(windows))].data for i in range(args.batch)]
        raw, targets = pack(rows)
        dec, cond, base, _ = features(raw)
        scale = min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * step / steps))
        count = int((targets >= 0).sum())
        for name, (kind, lr, weights, optimizer) in readouts.items():
            loss = nats(logits(name, dec, cond, base), targets) / count
            for group in optimizer.param_groups:
                group["lr"] = lr * scale
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 64 == 0 or step == steps - 1:
                curves[name].append((step, loss.item() / math.log(2)))
trained = time.time() - began

scores = evaluate()
result = {"run": args.run, "checkpoint": args.checkpoint, "checkpoint_identity": bundle.identity,
          "sampler_cursor": cursor, "data_seed": training["data_seed"], "train_windows": steps * args.batch,
          "batch": args.batch, "lrs": lrs, "kinds": args.kinds.split(","), "train_seconds": trained, **scores,
          "readout_rms": {f"{name}/{source}": weight.detach().pow(2).mean().sqrt().item()
                          for name, (_, _, weights, _) in readouts.items() for source, weight in weights.items()},
          "train_bpb_curves": curves}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
print(json.dumps({"val_bpb": scores["val_bpb"], "gains": scores["gains"], "split_bpb": scores["split_bpb"],
                  "train_seconds": round(trained)}, indent=1))
