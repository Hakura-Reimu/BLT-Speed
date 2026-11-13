from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch

from .config import BPEConfig, ModelConfig, config_from_dict, normalized_model_dict, preset
from .data import sha256_file
from .train import make_model


@dataclass
class EvaluationModel:
    model: torch.nn.Module
    config: ModelConfig
    tokenizer: Any
    device: torch.device
    identity: dict[str, Any]


def load_evaluation_model(*, run: str | Path | None = None, checkpoint: str | Path | None = None,
                          preset_name: str | None = None, min_len: int | None = None,
                          tokenizer_path: str | Path | None = None, device: str | torch.device = "cuda",
                          supply: str | Path | None = None, seed: int = 0) -> EvaluationModel:
    if (run is None) == (preset_name is None):
        raise ValueError("choose exactly one run directory or random preset")
    tokenizer = None
    tokenizer_hash = sha256_file(tokenizer_path) if tokenizer_path is not None else None
    if tokenizer_path is not None:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if run is not None:
        root = Path(run)
        frozen = json.loads((root / "config.json").read_text(encoding="utf-8"))
        config = config_from_dict(frozen["training"]["model"])
        if tokenizer_hash != frozen.get("tokenizer_sha256"):
            raise ValueError("tokenizer hash does not match the run")
        if supply is not None and sha256_file(Path(supply) / "manifest.json") != frozen["data_manifest_sha256"]:
            raise ValueError("supply manifest does not match the run")
        if checkpoint is None:
            pointer = json.loads((root / "ckpt/latest.json").read_text(encoding="utf-8"))
            path = root / "ckpt" / pointer["checkpoint"]
        else:
            path = Path(checkpoint)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (normalized_model_dict(payload["config"]["model"]) != asdict(config)
                or payload.get("tokenizer_hash") != tokenizer_hash):
            raise ValueError("checkpoint does not match its run config or tokenizer")
        if payload.get("data_manifest_hash") != frozen["data_manifest_sha256"]:
            raise ValueError("checkpoint does not match the run's supply")
        model = make_model(config)
        model.load_state_dict(payload["model"], strict=True)
        identity = {"kind": "checkpoint", "path": str(path.resolve()), "sha256": sha256_file(path),
                    "step": payload.get("step"), "consumed_bytes": payload.get("consumed_bytes"),
                    "git_revision": payload.get("git_revision"), "tokenizer_sha256": tokenizer_hash,
                    "data_manifest_sha256": frozen["data_manifest_sha256"]}
    else:
        if checkpoint is not None:
            raise ValueError("checkpoint requires a run directory")
        initial = preset(preset_name, min_len=min_len)
        if isinstance(initial, BPEConfig):
            if tokenizer is None:
                raise ValueError("BPE presets need a local tokenizer")
            bos_name = "<|endoftext|>" if "50K" in preset_name else "<|bos|>"
            bos = tokenizer.token_to_id(bos_name)
            if bos is None:
                raise ValueError("tokenizer is missing the expected BOS")
            config = preset(preset_name, vocab_size=tokenizer.get_vocab_size(), bos_id=bos)
        else:
            config = initial
        torch.manual_seed(seed)
        model = make_model(config)
        identity = {"kind": "random", "preset": preset_name, "seed": seed,
                    "config": asdict(config), "tokenizer_sha256": tokenizer_hash}
    if isinstance(config, BPEConfig) and tokenizer is None:
        raise ValueError("BPE evaluation requires its local tokenizer")
    target = torch.device(device)
    model.to(target).eval()
    return EvaluationModel(model, config, tokenizer, target, identity)
