"""Decoder-capacity continuation (review round 12; exploratory): can a deeper decoder use unchanged upstream states?

Both arms continue the same checkpoint on the same unseen training windows (the run's own sampler, continuing at the
checkpoint's committed sampler cursor) with the byte embedding, local BOS, encoder, pool, hash table and global trunk
frozen. The condition projection, decoder and head are trained with a fresh optimiser:
  --decoder-layers equal to the checkpoint's (dec2): continuation control
  more layers (dec4): residual blocks appended before the decoder's final norm, their attention and FFN output
                      projections zeroed so the initial function equals the checkpoint's (checked before any update)
Each arm is scored with the official full-validation scorer, whose per-chunk nats allow paired document-bootstrap
intervals between arms. The trained parameters are saved next to the result; frozen parameters are checked unchanged.

usage: python decoder_capacity.py RUN_DIR CHECKPOINT SUPPLY OUT_JSON --decoder-layers N [--train-windows 32768]
"""
import argparse
import dataclasses
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from blt.data import WindowIndex, iter_eval_chunks, sample_window_index
from blt.eval_io import load_evaluation_model
from blt.score import autocast_for, configure_matmul_precision, score_chunks
from blt.train import _loss, git_revision, make_model

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("checkpoint")
parser.add_argument("supply")
parser.add_argument("out")
parser.add_argument("--decoder-layers", type=int, required=True)
parser.add_argument("--first-index", type=int, default=None, help="optional; must equal the checkpoint's sampler cursor")
parser.add_argument("--train-windows", type=int, default=32768, help="256MiB of 8KiB windows")
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--micro-batch", type=int, default=8)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=0, help="initialises the added blocks' input projections")
parser.add_argument("--device", default="cuda")
parser.add_argument("--max-val-chunks", type=int, default=None, help="smoke tests only")
args = parser.parse_args()

revision = git_revision()
device = torch.device(args.device)
configure_matmul_precision()
payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
training, cursor = payload["config"], payload["sampler_cursor"]
del payload
if args.first_index is not None and args.first_index != cursor:
    raise SystemExit(f"--first-index {args.first_index} differs from the checkpoint's sampler cursor {cursor}")
bundle = load_evaluation_model(run=args.run, checkpoint=args.checkpoint, device=device, supply=args.supply)
original, c = bundle.model.eval(), bundle.model.config
if original.global_readout is not None or args.decoder_layers < c.dec_layers:
    raise SystemExit("expects a checkpoint without a global readout and at least its own decoder depth")
if args.decoder_layers == c.dec_layers:
    model = original
else:
    torch.manual_seed(args.seed)
    model = make_model(dataclasses.replace(c, dec_layers=args.decoder_layers))
    missing, unexpected = model.load_state_dict(original.state_dict(), strict=False)
    added = tuple(f"decoder.blocks.{i}." for i in range(c.dec_layers, args.decoder_layers))
    if unexpected or not missing or not all(key.startswith(added) for key in missing):
        raise SystemExit(f"unexpected state-dict mismatch: missing {missing}, unexpected {unexpected}")
    with torch.no_grad():
        for block in model.decoder.blocks[c.dec_layers:]:
            block.attn.out.weight.zero_()
            block.down.weight.zero_()
    model.to(device)

trainable = ("condition.", "decoder.", "head.")
for name, parameter in model.named_parameters():
    parameter.requires_grad_(name.startswith(trainable))
trained = [parameter for parameter in model.parameters() if parameter.requires_grad]
frozen = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
          if not parameter.requires_grad}
# Same AdamW settings as make_optimizer; none of the trained parameters is an embedding.
optimizer = torch.optim.AdamW([{"params": [p for p in trained if p.ndim >= 2], "weight_decay": args.weight_decay},
                               {"params": [p for p in trained if p.ndim < 2], "weight_decay": 0.0}],
                              lr=args.lr, betas=(0.9, 0.95), fused=device.type == "cuda")


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


steps = math.ceil(args.train_windows / args.batch)
curve, initial_difference = [], None
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)
began = time.time()
with WindowIndex(Path(args.supply) / "train.bin", training["window_bytes"]) as windows:
    for step in range(steps):
        rows = [windows[sample_window_index(training["data_seed"], cursor + step * args.batch + i, 0, 1,
                                            len(windows))].data for i in range(args.batch)]
        if step == 0 and model is not original:
            with torch.no_grad(), autocast_for(device):
                raw, _ = pack(rows[:args.micro_batch])
                initial_difference = (model(raw).float() - original(raw).float()).abs().max().item()
            if initial_difference != 0.0:
                raise SystemExit(f"the expanded decoder changes the initial function (max |logit diff| {initial_difference})")
        total = sum(len(row) for row in rows)
        scale = (step + 1) / args.warmup if step < args.warmup else \
            0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, steps - args.warmup)))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale
        optimizer.zero_grad(set_to_none=True)
        nll = None
        for start in range(0, args.batch, args.micro_batch):
            raw, targets = pack(rows[start:start + args.micro_batch])
            with autocast_for(device):
                part = _loss(model, raw, targets)
            (part / total).backward()
            nll = part.detach() if nll is None else nll + part.detach()
        norm = torch.nn.utils.clip_grad_norm_(trained, args.grad_clip)
        optimizer.step()
        loss, norm = torch.stack((nll.float() / (total * math.log(2)), norm.float())).cpu().tolist()
        curve.append((step, loss, norm))
trained_seconds = time.time() - began
peak = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None

unchanged = all(torch.equal(parameter.detach().cpu(), frozen[name]) for name, parameter in model.named_parameters()
                if name in frozen)
if not unchanged:
    raise SystemExit("a frozen parameter changed during training")
chunks = itertools.islice(iter_eval_chunks(args.supply, "val", 8192), args.max_val_chunks)
score = score_chunks(model, chunks, device=device, split="val", checkpoint=bundle.identity)
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"decoder_layers": args.decoder_layers, "config": dataclasses.asdict(model.config),
            "trained": {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}},
           out.with_suffix(".pt"))
result = {"run": args.run, "checkpoint": args.checkpoint, "checkpoint_identity": bundle.identity,
          "git_revision": revision, "decoder_layers": args.decoder_layers, "checkpoint_decoder_layers": c.dec_layers,
          "sampler_cursor": cursor, "data_seed": training["data_seed"], "seed": args.seed,
          "train_windows": steps * args.batch, "batch": args.batch, "micro_batch": args.micro_batch, "lr": args.lr,
          "warmup": args.warmup, "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
          "trained_parameters": sum(p.numel() for p in trained), "frozen_parameters": sum(p.numel() for p in frozen.values()),
          "initial_max_abs_logit_difference": initial_difference, "frozen_unchanged": unchanged,
          "train_seconds": trained_seconds, "train_bytes_per_second": steps * args.batch * training["window_bytes"] / trained_seconds,
          "peak_reserved_bytes": peak, "train_curve": curve, "val_bpb": score["bpb"], "val_bytes": score["bytes"],
          "chunk_count": score["chunk_count"], "chunks": score["chunks"]}
out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
print(json.dumps({"decoder_layers": args.decoder_layers, "val_bpb": score["bpb"], "train_seconds": round(trained_seconds),
                  "last_train_bpb": float(np.mean([row[1] for row in curve[-128:]])),
                  "initial_difference": initial_difference, "peak_reserved_bytes": peak}, indent=1))
