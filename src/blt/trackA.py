from __future__ import annotations

"""Track A's local Hugging Face baselines and byte-level scoring utilities."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Iterable

import torch

from .data import ByteWindow, sha256_file


@dataclass
class ReferenceModel:
    family: str
    model: Any
    tokenizer: Any
    identity: dict[str, Any]
    device: torch.device
    bos_id: int


def checkpoint_identity(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"checkpoint is not a directory: {root}")
    names = ("*.safetensors", "*.safetensors.index.json", "*.bin", "config.json", "generation_config.json",
             "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
             "vocab*", "merges*", "*.model", "*.py")
    files = sorted({p for pattern in names for p in root.glob(pattern) if p.is_file()})
    return {"path": str(root), "revision": root.name if root.parent.name == "snapshots" else None,
            "files_sha256": {p.name: sha256_file(p) for p in files}}


def _cuda_home() -> None:
    # Bolmo imports xlstm while loading its remote code, even though it uses mLSTM.
    os.environ.setdefault("CUDA_HOME", str(Path(torch.__file__).resolve().parent.parent / "nvidia" / "cuda_runtime"))


def load_reference(family: str, checkpoint: str | Path, *, bolmo_subword_tokenizer: str | None = None) -> ReferenceModel:
    if family not in {"bolmo", "olmo"}:
        raise ValueError("family must be bolmo or olmo")
    if not torch.cuda.is_available():
        raise RuntimeError("Track A reference models require CUDA")
    _cuda_home()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    path = str(Path(checkpoint).expanduser().resolve(strict=True))
    identity = checkpoint_identity(path)
    if family == "bolmo":
        config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        tokenizer_config = dict(config.tokenizer_config)
        original = bolmo_subword_tokenizer or tokenizer_config.get("original_identifier")
        if not original:
            raise ValueError("Bolmo's subword tokenizer was not specified")
        local_path = Path(original).expanduser()
        if not local_path.exists():
            from huggingface_hub import snapshot_download
            try:
                local_path = Path(snapshot_download(repo_id=original, local_files_only=True))
            except Exception as exc:
                raise RuntimeError("Bolmo's subword tokenizer is not cached; pass --bolmo-subword-tokenizer with a local directory") from exc
        tokenizer_config["original_identifier"] = str(local_path.resolve(strict=True))
        config.tokenizer_config = tokenizer_config
        identity["subword_tokenizer"] = checkpoint_identity(local_path)
        model = AutoModelForCausalLM.from_pretrained(path, config=config, trust_remote_code=True,
                                                    local_files_only=True, torch_dtype=torch.bfloat16).to("cuda").eval()
        tokenizer = model.model.tokenizer
        if (tokenizer.bos_token_id, tokenizer.offset, model.config.vocab_size) != (1, 4, 520):
            raise ValueError("unexpected Bolmo special IDs or output vocabulary")
        bos_id = 1
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.bfloat16,
                                                    attn_implementation="sdpa").to("cuda").eval()
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = model.config.bos_token_id
        if bos_id is None:
            bos_id = tokenizer.eos_token_id
        if bos_id is None:
            raise ValueError("OLMo tokenizer and model define neither BOS nor an EOS fallback")
    if any(p.device.type != "cuda" or (p.is_floating_point() and p.dtype != torch.bfloat16) for p in model.parameters()):
        raise RuntimeError("reference model is not fully resident in BF16 on CUDA")
    config = model.config
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    kv_bytes_per_position = 2 * int(config.num_hidden_layers) * int(config.num_key_value_heads) * int(head_dim) * 2
    if kv_bytes_per_position != 131072:
        raise ValueError(f"unexpected reference-model BF16 global KV size: {kv_bytes_per_position} bytes per position")
    if family == "bolmo":
        from .router import _parameters
        _parameters(model)
    identity.update({"family": family, "dtype": "bfloat16", "device": str(torch.device("cuda")),
                     "model_revision": getattr(model.config, "_commit_hash", None), "bos_id": int(bos_id),
                     "bf16_global_kv_bytes_per_position": kv_bytes_per_position})
    return ReferenceModel(family, model, tokenizer, identity, torch.device("cuda"), int(bos_id))


def folded_log_probs(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] != 520:
        raise ValueError("Bolmo logits must have 520 classes")
    lp = logits.float().log_softmax(dim=-1)
    return torch.logaddexp(lp[..., 4:260], lp[..., 264:520])


def bolmo_ids(raw: bytes, device: torch.device | str, batch: int = 1) -> torch.Tensor:
    if not raw or batch <= 0:
        raise ValueError("a nonempty byte sequence and positive batch are required")
    return torch.tensor([1, *(value + 4 for value in raw)], dtype=torch.long, device=device)[None, :].repeat(batch, 1)


def olmo_ids(tokenizer: Any, raw: bytes) -> list[int]:
    text = raw.decode("utf-8")
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False).encode("utf-8") != raw:
        raise ValueError("OLMo tokenizer did not losslessly round-trip the chunk")
    return ids


def free_bolmo_cache(model: Any) -> None:
    model.model.local_encoder.free_inference_cache()
    model.model.local_decoder.free_inference_cache()


@torch.inference_mode()
def score_bolmo_chunk(bundle: ReferenceModel, raw: bytes, *, byte_nll: bool = False) -> dict[str, Any]:
    if bundle.family != "bolmo":
        raise ValueError("expected a Bolmo model")
    free_bolmo_cache(bundle.model)
    ids = bolmo_ids(raw, bundle.device)
    masks: list[torch.Tensor] = []
    def capture(_module, _args, output):
        masks.append(output[3].detach().clone())
    handle = bundle.model.model.local_encoder.register_forward_hook(capture)
    try:
        logits = bundle.model(input_ids=ids, use_cache=False).logits[:, :-1]
    finally:
        handle.remove()
        free_bolmo_cache(bundle.model)
    if len(masks) != 1 or masks[0].shape != ids.shape:
        raise RuntimeError("did not capture the actual Bolmo prefill boundary mask")
    targets = torch.tensor(list(raw), device=bundle.device, dtype=torch.long)[None, :, None]
    values = -folded_log_probs(logits).gather(-1, targets).squeeze(-1)[0]
    nats = float(values.sum().item())
    patches = int(masks[0].sum().item())
    if not patches or not math.isfinite(nats):
        raise RuntimeError("invalid Bolmo scoring result")
    result = {"bytes": len(raw), "nats": nats, "bpb": nats / (len(raw) * math.log(2)),
              "patches": patches, "bytes_per_patch": len(raw) / patches}
    if byte_nll:
        result["byte_nll"] = values.cpu().tolist()
    return result


@torch.inference_mode()
def score_olmo_chunk(bundle: ReferenceModel, raw: bytes) -> dict[str, Any]:
    if bundle.family != "olmo":
        raise ValueError("expected an OLMo model")
    tokens = olmo_ids(bundle.tokenizer, raw)
    limit = int(bundle.model.config.max_position_embeddings)
    if len(tokens) > limit:
        raise ValueError(f"chunk requires {len(tokens)} OLMo positions, exceeding {limit}")
    inputs = torch.tensor([bundle.bos_id, *tokens[:-1]], device=bundle.device, dtype=torch.long)[None, :]
    target = torch.tensor(tokens, device=bundle.device, dtype=torch.long)[None, :, None]
    logits = bundle.model(input_ids=inputs, use_cache=False).logits
    nats = float(-logits.float().log_softmax(-1).gather(-1, target).sum().item())
    if not math.isfinite(nats):
        raise RuntimeError("invalid OLMo scoring result")
    return {"bytes": len(raw), "nats": nats, "bpb": nats / (len(raw) * math.log(2)),
            "tokens": len(tokens), "bytes_per_token": len(raw) / len(tokens)}


def score_reference(bundle: ReferenceModel, chunks: Iterable[ByteWindow], *, split: str = "val",
                    domains: dict[str, str] | None = None) -> dict[str, Any]:
    records = []
    for chunk in chunks:
        if not chunk.data:
            continue
        row = score_bolmo_chunk(bundle, chunk.data) if bundle.family == "bolmo" else score_olmo_chunk(bundle, chunk.data)
        row.update({"document": chunk.document, "start": chunk.start, "end": chunk.end})
        if domains is not None:
            row["domain"] = domains.get(chunk.document, "unlabelled")
        records.append(row)
    if not records:
        raise ValueError("no text bytes to score")
    def aggregate(rows):
        size = sum(row["bytes"] for row in rows)
        nats = sum(row["nats"] for row in rows)
        unit = "patches" if bundle.family == "bolmo" else "tokens"
        return {"bytes": size, "nats": nats, "bpb": nats / (math.log(2) * size),
                unit: sum(row[unit] for row in rows),
                "bytes_per_patch" if unit == "patches" else "bytes_per_token": size / sum(row[unit] for row in rows)}
    result = {"split": split, "checkpoint": bundle.identity, "chunk_count": len(records),
              "metric": "folded BPB under Bolmo's prefill layout" if bundle.family == "bolmo" else "token NLL per original byte",
              "chunks": records, **aggregate(records)}
    if domains is not None:
        result["domains"] = {name: aggregate([r for r in records if r["domain"] == name])
                             for name in sorted({r["domain"] for r in records})}
    return result
