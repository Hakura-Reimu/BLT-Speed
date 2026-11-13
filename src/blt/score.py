from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .config import BPEConfig, ByteConfig
from .data import ByteWindow, iter_eval_chunks, tokenize_window
from .patcher import patch_batch


def autocast_for(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


NUMERICS_POLICY = {"version": 1, "autocast": "bfloat16", "bf16_reduced_precision_reduction": False}


def configure_matmul_precision() -> None:
    """Use fp32 reductions in cuBLAS bf16 GEMMs for training and BPB scoring (not for timed inference).

    With PyTorch's default reduced-precision split-K reductions, the S-BPE50K gradient under bf16 autocast was
    1.6% (whole head) and 2.5% (1,024-position chunks) from an fp32 reference; with fp32 reductions both were 0.8%.
    """
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = NUMERICS_POLICY["bf16_reduced_precision_reduction"]


@torch.inference_mode()
def score_chunks(model: torch.nn.Module, chunks: Iterable[ByteWindow], *, family: str | None = None,
                 tokenizer: Any = None, bos_id: int | None = None, device: str | torch.device = "cpu",
                 split: str = "val", checkpoint: dict[str, Any] | None = None) -> dict[str, Any]:
    configure_matmul_precision()
    config = getattr(model, "config", None)
    family = family or getattr(config, "family", None)
    if family not in {"byte", "bpe"}:
        raise ValueError("family must be byte or bpe")
    if family == "bpe" and (tokenizer is None or (bos_id is None and not isinstance(config, BPEConfig))):
        raise ValueError("BPE scoring requires a tokenizer and BOS ID")
    if bos_id is None and isinstance(config, BPEConfig):
        bos_id = config.bos_id
    target_device = torch.device(device)
    model.eval()
    total_nats, total_bytes, units = 0.0, 0, 0
    records = []
    for chunk in chunks:
        raw = chunk.data
        if not raw:
            continue
        if family == "byte":
            labels = torch.tensor(list(raw), dtype=torch.long, device=target_device)[None, :]
            inputs = labels
            if isinstance(config, ByteConfig):
                patches = patch_batch(np.frombuffer(raw, dtype=np.uint8)[None, :], config.min_len, config.max_len, config.block_size)
                count = int(patches.counts[0])
                units += count + int(not patches.boundaries[0, -1])
        else:
            ids = tokenize_window(tokenizer, raw)
            labels = torch.tensor(ids, dtype=torch.long, device=target_device)[None, :]
            inputs = torch.tensor([bos_id, *ids[:-1]], dtype=torch.long, device=target_device)[None, :]
            units += len(ids)
        with autocast_for(target_device):
            logits = model(inputs)
            nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
        value = float(nll.item())
        total_nats += value
        total_bytes += len(raw)
        records.append({"document": chunk.document, "start": chunk.start, "end": chunk.end,
                        "bytes": len(raw), "nats": value})
    if not total_bytes:
        raise ValueError("no text bytes to score")
    result = {"split": split, "chunk_count": len(records), "bytes": total_bytes, "total_nats": total_nats,
              "bpb": total_nats / (math.log(2) * total_bytes), "per_chunk_nll": [r["nats"] for r in records],
              "chunks": records, "checkpoint": checkpoint}
    if units:
        result["bytes_per_token" if family == "bpe" else "mean_patch_length"] = total_bytes / units
    return result


def score_supply(model: torch.nn.Module, supply: str, *, split: str = "val", window_bytes: int = 8192, **kwargs: Any) -> dict[str, Any]:
    if split not in {"val", "test"}:
        raise ValueError("scoring split must be val or test")
    return score_chunks(model, iter_eval_chunks(supply, split, window_bytes), split=split, **kwargs)
