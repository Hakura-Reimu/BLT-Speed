from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TransformerConfig:
    layers: int
    width: int
    q_heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    max_positions: int = 32769

    def __post_init__(self) -> None:
        if min(self.layers, self.width, self.q_heads, self.kv_heads, self.head_dim, self.ffn, self.max_positions) <= 0:
            raise ValueError("transformer dimensions must be positive")
        if self.width != self.q_heads * self.head_dim or self.q_heads % self.kv_heads:
            raise ValueError("width/head dimensions or GQA head ratio are invalid")
        if self.head_dim % 2 or self.dropout != 0 or self.rope_theta <= 0 or self.norm_eps <= 0:
            raise ValueError("RoPE requires an even head dimension and dropout must be zero")


@dataclass(frozen=True)
class BPEConfig:
    name: str
    trunk: TransformerConfig
    vocab_size: int
    bos_id: int
    tied_embeddings: bool = True
    family: str = "bpe"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or not 0 <= self.bos_id < self.vocab_size or not self.tied_embeddings:
            raise ValueError("invalid BPE vocabulary, BOS, or embedding tying")


@dataclass(frozen=True)
class ByteConfig:
    name: str
    trunk: TransformerConfig
    local_width: int
    local_heads: int
    local_ffn: int
    min_len: int | None
    target_patch_mean: int
    max_len: int = 16
    block_size: int = 64
    enc_layers: int = 1
    dec_layers: int = 2
    hash_enabled: bool = True
    hash_rows: int = 8192
    hash_seed: int = 0
    byte_vocab_size: int = 259
    pad_id: int = 256
    bos_id: int = 257
    eos_id: int = 258
    output_eos_id: int = 256
    output_classes: int = 257
    local_block_resets: bool = True
    family: str = "byte"
    hash_algorithm: str = "mersenne31-v1"
    # "balanced" divides the pool projection by sqrt(local_width) at init so the pooled branch starts at the hash
    # branch's RMS (exploratory diagnostic, Sol round 10). Configs written before this field load as "normal".
    pool_init: str = "normal"
    # Adds a zero-initialised linear readout from the gathered global state straight to the byte logits, bypassing the
    # local-width decoder (exploratory C1, Sol round 11). Training and scoring only; cached inference is not implemented.
    global_readout: bool = False

    def __post_init__(self) -> None:
        if min(self.local_width, self.local_heads, self.local_ffn) <= 0 or self.local_width % self.local_heads or (self.local_width // self.local_heads) % 2:
            raise ValueError("invalid local attention dimensions")
        if not 1 <= self.max_len <= self.block_size or (self.min_len is not None and not 1 <= self.min_len <= self.max_len):
            raise ValueError("invalid patch lengths")
        if (self.byte_vocab_size, self.pad_id, self.bos_id, self.eos_id, self.output_eos_id, self.output_classes) != (259, 256, 257, 258, 256, 257):
            raise ValueError("invalid byte vocabulary contract")
        if min(self.enc_layers, self.dec_layers, self.hash_rows) <= 0 or not self.local_block_resets or self.hash_algorithm != "mersenne31-v1":
            raise ValueError("invalid local or hash configuration")
        if self.pool_init not in ("normal", "balanced"):
            raise ValueError("pool_init must be normal or balanced")


ModelConfig = BPEConfig | ByteConfig


def preset(name: str, *, min_len: int | None = None, vocab_size: int | None = None, bos_id: int | None = None, hash_seed: int = 0) -> ModelConfig:
    """Build an approved preset; byte presets must be calibrated before use.

    The S-P6-poolbal and S-P6-gread presets are exploratory diagnostics, not registered models. The XS-BPE presets
    are an exploratory smaller BPE control for the quality-speed frontier (Sol round 13), also not registered.
    """
    names = {"S-BPE50K", "S-BPE16K", "S-P6", "S-P8", "M-BPE50K", "M-BPE16K", "M-P6", "M-P8", "M-P6-nohash",
             "S-P6-poolbal", "S-P6-poolbal-w256", "S-P6-gread", "XS-BPE50K", "XS-BPE16K"}
    if name not in names:
        raise ValueError(f"unknown preset: {name}")
    small = name.startswith("S-")
    width, heads, ffn = (512, 8, 1536) if small else (768, 12, 2048)
    layers = (8 if small else 11) if "P8" in name else (6 if small else 9)
    trunk = (TransformerConfig(4, 256, 4, 4, 64, 768) if name.startswith("XS-")
             else TransformerConfig(layers, width, heads, 4, 64, ffn))
    if "BPE" in name:
        size = vocab_size if vocab_size is not None else (50257 if "50K" in name else 16384)
        if "50K" in name and size != 50257:
            raise ValueError("GPT-2 vocabulary must contain exactly 50257 entries")
        return BPEConfig(name, trunk, size, (50256 if "50K" in name else 0) if bos_id is None else bos_id)
    if vocab_size is not None or bos_id is not None:
        raise ValueError("vocab_size and bos_id apply only to BPE")
    local = (256, 4, 768) if name.endswith("-w256") else (128, 2, 384) if small else (192, 3, 512)
    return ByteConfig(name, trunk, *local, min_len, 8 if "P8" in name else 6, hash_enabled="nohash" not in name,
                      hash_seed=hash_seed, pool_init="balanced" if "poolbal" in name else "normal",
                      global_readout="gread" in name)


def config_from_dict(value: dict[str, Any]) -> ModelConfig:
    fields = dict(value)
    fields["trunk"] = TransformerConfig(**fields["trunk"])
    if fields.get("family") == "bpe":
        return BPEConfig(**fields)
    if fields.get("family") == "byte":
        return ByteConfig(**fields)
    raise ValueError("unknown model family")


def normalized_model_dict(value: dict[str, Any]) -> dict[str, Any]:
    """Fill fields added after a checkpoint was written with their defaults, so saved configs compare equal."""
    return asdict(config_from_dict(value))


def write_config(config: ModelConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parameter_counts(config: ModelConfig) -> dict[str, int]:
    """Count unique parameters, including norms and the untied byte output head."""
    def dense(c: TransformerConfig) -> int:
        return c.layers * (2 * c.width * c.width + 2 * c.width * c.kv_heads * c.head_dim + 3 * c.width * c.ffn)

    g = config.trunk
    global_dense = dense(g)
    global_norms = (2 * g.layers + 1) * g.width
    if isinstance(config, BPEConfig):
        embeddings = config.vocab_size * g.width
        return {"total": global_dense + global_norms + embeddings, "global_dense": global_dense, "embeddings_hash": embeddings}
    local = TransformerConfig(config.enc_layers + config.dec_layers, config.local_width, config.local_heads,
                              config.local_heads, config.local_width // config.local_heads, config.local_ffn)
    local_norms = (2 * local.layers + 2) * local.width
    embeddings = config.byte_vocab_size * local.width + (6 * config.hash_rows * g.width if config.hash_enabled else 0)
    # Two learned BOS vectors, pool/up projections and the 257-way byte head.
    other = local.width + g.width + 2 * local.width * g.width + config.output_classes * local.width
    if config.global_readout:
        other += config.output_classes * g.width
    return {"total": global_dense + global_norms + dense(local) + local_norms + embeddings + other,
            "global_dense": global_dense, "embeddings_hash": embeddings}
