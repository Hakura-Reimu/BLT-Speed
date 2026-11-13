from __future__ import annotations

"""Checkpoint-aware local surgery for the released Hugging Face Bolmo model.

The selection is deliberately a *decision*, not a ranking algorithm.  Layer
keys refer to teacher indices, for example ``encoder.0`` and ``decoder.2``::

    {"version": 1, "mode": "slice", "decoder_layers": [0, 2],
     "layers": {"encoder.0": {"heads": [...], "ffn": [...]},
                "decoder.0": {"heads": [...], "ffn": [...]},
                "decoder.2": {"heads": [...], "ffn": [...]}},
     "router": {"dims": [...]}}

``mode`` may also be ``random`` or ``svd``.  For the latter, use
``router: {"rank": r}`` to project Q and K into one joint left-singular
subspace.  An optional ``linear_svd_rank`` factorizes each of the encoder
output and decoder input projections individually, without crossing a
nonlinearity.  A random control can use the same rank and shapes.
"""

import copy
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import threading
import time
from typing import Any, Iterable
import weakref

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .atomic import atomic_json
from .data import sample_window_index
from .trackA import bolmo_ids, folded_log_probs
from .train import _atomic_torch_save, _truncate_uncommitted_log, git_revision


def _indices(values: Any, size: int, name: str) -> list[int]:
    if not isinstance(values, list) or not values or any(type(x) is not int for x in values):
        raise ValueError(f"{name} must be a nonempty list of integer indices")
    if len(set(values)) != len(values) or any(x < 0 or x >= size for x in values):
        raise ValueError(f"{name} contains duplicate or out-of-range indices")
    return values


def _expect_shape(value: torch.Tensor | None, shape: tuple[int, ...], name: str) -> None:
    if value is None or tuple(value.shape) != shape:
        actual = None if value is None else tuple(value.shape)
        raise ValueError(f"{name}: expected {shape}, got {actual}")


def _linear(source: nn.Linear, rows: list[int] | None = None,
            cols: list[int] | None = None) -> nn.Linear:
    if not isinstance(source, nn.Linear):
        raise ValueError(f"expected nn.Linear, got {type(source).__name__}")
    rows = list(range(source.out_features)) if rows is None else rows
    cols = list(range(source.in_features)) if cols is None else cols
    _expect_shape(source.weight, (source.out_features, source.in_features), "linear.weight")
    if source.bias is not None:
        _expect_shape(source.bias, (source.out_features,), "linear.bias")
    target = nn.Linear(len(cols), len(rows), bias=source.bias is not None,
                       device=source.weight.device, dtype=source.weight.dtype)
    row_index = torch.tensor(rows, device=source.weight.device)
    col_index = torch.tensor(cols, device=source.weight.device)
    with torch.no_grad():
        target.weight.copy_(source.weight.index_select(0, row_index).index_select(1, col_index))
        if source.bias is not None:
            target.bias.copy_(source.bias.index_select(0, row_index))
    return target


def _head_coordinates(heads: list[int], size_per_head: int) -> list[int]:
    return [head * size_per_head + i for head in heads for i in range(size_per_head)]


def _thin_xlstm(source: nn.Module, heads: list[int], width: int) -> nn.Module:
    cfg = source.config
    old_heads = int(cfg.num_heads)
    qdim, vdim = int(source.qk_dim), int(source.v_dim)
    if (int(cfg.embedding_dim) != width or qdim != int(width * cfg.qk_dim_factor)
            or vdim != int(width * cfg.v_dim_factor) or qdim % old_heads or vdim % old_heads):
        raise ValueError("inconsistent xLSTM embedding, head, Q/K or V dimensions")
    qsel = _head_coordinates(heads, qdim // old_heads)
    vsel = _head_coordinates(heads, vdim // old_heads)
    replacements: dict[int, Any] = {}
    expected: set[str] = set()

    def replace(name: str, shape: tuple[int, int], rows: list[int] | None = None,
                cols: list[int] | None = None) -> None:
        layer = getattr(source, name, None)
        if not isinstance(layer, nn.Linear):
            raise ValueError(f"xLSTM is missing linear {name}")
        _expect_shape(layer.weight, shape, f"xlstm.{name}.weight")
        expected.add(name + ".weight")
        if layer.bias is not None:
            _expect_shape(layer.bias, (shape[0],), f"xlstm.{name}.bias")
            expected.add(name + ".bias")
        replacements[id(layer)] = _linear(layer, rows, cols)

    if cfg.weight_mode == "single":
        for name in ("q", "k"):
            replace(name, (qdim, width), qsel)
        for name in ("v", "ogate_preact"):
            replace(name, (vdim, width), vsel)
        for name in ("igate_preact", "fgate_preact"):
            replace(name, (old_heads, width), heads)
            if getattr(source, name).bias is None:
                raise ValueError(f"xlstm.{name} must have its checkpoint bias")
    elif cfg.weight_mode == "fused":
        rows = qsel + [qdim + i for i in qsel] + [2 * qdim + i for i in vsel]
        rows += [2 * qdim + vdim + i for i in vsel]
        replace("qkv_opreact", (2 * qdim + 2 * vdim, width), rows)
        replace("ifgate_preact", (2 * old_heads, width), heads + [old_heads + i for i in heads])
        if source.ifgate_preact.bias is None:
            raise ValueError("xlstm.ifgate_preact must have its checkpoint bias")
    else:
        raise ValueError(f"unsupported xLSTM weight mode: {cfg.weight_mode}")

    replace("out_proj", (width, vdim), cols=vsel)
    norm = source.multihead_norm
    if (int(norm.num_heads) != old_heads or int(norm.head_dim) != vdim // old_heads
            or int(norm.num_features) != vdim):
        raise ValueError("unexpected xLSTM multihead_norm metadata")
    new_norm = copy.deepcopy(norm)
    index = torch.tensor(vsel, device=source.out_proj.weight.device)
    for name in ("weight", "bias"):
        value = getattr(norm, name, None)
        if value is not None:
            _expect_shape(value, (vdim,), f"xlstm.multihead_norm.{name}")
            expected.add("multihead_norm." + name)
            setattr(new_norm, name, nn.Parameter(value.detach().index_select(0, index).clone()))
    new_norm.num_heads = len(heads)
    new_norm.num_features = len(vsel)
    replacements[id(norm)] = new_norm
    actual = set(dict(source.named_parameters()))
    if actual != expected:
        raise ValueError(f"unexpected xLSTM parameter names: missing {sorted(expected-actual)}, extra {sorted(actual-expected)}")
    target = copy.deepcopy(source, replacements)
    target.config.num_heads = len(heads)
    target.config.qk_dim_factor = len(qsel) / width
    target.config.v_dim_factor = len(vsel) / width
    target.qk_dim, target.v_dim = len(qsel), len(vsel)
    return target


def _thin_mlp(source: nn.Module, channels: list[int], width: int) -> nn.Module:
    gate, up, down = source.gate_proj, source.up_proj, source.down_proj
    if not all(isinstance(layer, nn.Linear) for layer in (gate, up, down)):
        raise ValueError("local FFN projections must be linear")
    old_hidden = gate.out_features
    for name, layer, shape in (("gate", gate, (old_hidden, width)), ("up", up, (old_hidden, width)),
                               ("down", down, (width, old_hidden))):
        _expect_shape(layer.weight, shape, f"mlp.{name}_proj.weight")
    expected = {f"{name}.weight" for name in ("gate_proj", "up_proj", "down_proj")}
    expected.update(f"{name}.bias" for name in ("gate_proj", "up_proj", "down_proj")
                    if getattr(source, name).bias is not None)
    if set(dict(source.named_parameters())) != expected:
        raise ValueError("unexpected local FFN parameter names")
    target = copy.deepcopy(source, {id(gate): _linear(gate, channels), id(up): _linear(up, channels),
                                    id(down): _linear(down, cols=channels)})
    target.intermediate_size = len(channels)
    if hasattr(target, "config"):
        target.config.intermediate_size = len(channels)
    return target


def _thin_layer(source: nn.Module, chosen: dict[str, Any], width: int, name: str) -> nn.Module:
    if not isinstance(chosen, dict) or set(chosen) != {"heads", "ffn"}:
        raise ValueError(f"{name} must contain exactly heads and ffn")
    heads = _indices(chosen["heads"], int(source.xlstm.config.num_heads), name + ".heads")
    channels = _indices(chosen["ffn"], int(source.mlp.gate_proj.out_features), name + ".ffn")
    for norm_name in ("pre_xlstm_layernorm", "pre_feedforward_layernorm"):
        _expect_shape(getattr(source, norm_name).weight, (width,), f"{name}.{norm_name}.weight")
    expected = {"pre_xlstm_layernorm.weight", "pre_feedforward_layernorm.weight"}
    expected.update("xlstm." + key for key in dict(source.xlstm.named_parameters()))
    expected.update("mlp." + key for key in dict(source.mlp.named_parameters()))
    if set(dict(source.named_parameters())) != expected:
        raise ValueError(f"{name} has unexpected local parameter names")
    target = copy.deepcopy(source, {id(source.xlstm): _thin_xlstm(source.xlstm, heads, width),
                                    id(source.mlp): _thin_mlp(source.mlp, channels, width)})
    if hasattr(target, "config"):
        target.config.num_local_heads = len(heads)
        target.config.local_intermediate_size = len(channels)
    return target


class FactorizedLinear(nn.Module):
    """Two linear factors with no intervening activation and one output bias."""
    def __init__(self, source: nn.Linear, rank: int, *, random_init: bool = False):
        super().__init__()
        if not 1 <= rank <= min(source.in_features, source.out_features):
            raise ValueError("linear_svd_rank exceeds a projection dimension")
        self.down = nn.Linear(source.in_features, rank, bias=False, device=source.weight.device, dtype=source.weight.dtype)
        self.up = nn.Linear(rank, source.out_features, bias=source.bias is not None,
                            device=source.weight.device, dtype=source.weight.dtype)
        with torch.no_grad():
            if random_init:
                nn.init.normal_(self.down.weight, std=0.02)
                nn.init.normal_(self.up.weight, std=0.02)
            else:
                u, s, vh = torch.linalg.svd(source.weight.detach().float(), full_matrices=False)
                self.down.weight.copy_((s[:rank, None].sqrt() * vh[:rank]).to(source.weight.dtype))
                self.up.weight.copy_((u[:, :rank] * s[:rank].sqrt()).to(source.weight.dtype))
            if source.bias is not None:
                self.up.bias.copy_(torch.zeros_like(source.bias) if random_init else source.bias)

    @property
    def weight(self) -> torch.Tensor:
        # HF's cache initializer asks only for weight.dtype.
        return self.up.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


def _router(source: nn.Module, router: Any, mode: str, width: int) -> nn.Module:
    q, k = source.q_proj_layer, source.k_proj_layer
    if not isinstance(q, nn.Linear) or not isinstance(k, nn.Linear):
        raise ValueError("router Q/K must be linear")
    _expect_shape(q.weight, (width, width), "router.q_proj_layer.weight")
    _expect_shape(k.weight, (width, width), "router.k_proj_layer.weight")
    if q.bias is not None or k.bias is not None or set(dict(source.named_parameters())) != {
            "q_proj_layer.weight", "k_proj_layer.weight"}:
        raise ValueError("unexpected router parameter names or biases")
    if not isinstance(router, dict):
        raise ValueError("router must specify dims or rank")
    if set(router) == {"dims"}:
        dims = _indices(router["dims"], width, "router.dims")
        qnew, knew = _linear(q, dims), _linear(k, dims)
    elif set(router) == {"rank"} and type(router["rank"]) is int and 1 <= router["rank"] <= width:
        rank = router["rank"]
        if mode not in {"svd", "random"}:
            raise ValueError("a router rank requires svd or random mode")
        qnew, knew = _linear(q, list(range(rank))), _linear(k, list(range(rank)))
        if mode == "svd" and rank < width:
            with torch.no_grad():
                u, _, _ = torch.linalg.svd(torch.cat((q.weight.detach().float(), k.weight.detach().float()), dim=1),
                                           full_matrices=False)
                basis = u[:, :rank].T
                qnew.weight.copy_((basis @ q.weight.float()).to(q.weight.dtype))
                knew.weight.copy_((basis @ k.weight.float()).to(k.weight.dtype))
    else:
        raise ValueError("router must have exactly dims or a valid rank")
    return copy.deepcopy(source, {id(q): qnew, id(k): knew})


def _randomize_locals(student: nn.Module) -> None:
    encoder, decoder = student.model.local_encoder, student.model.local_decoder
    suffix = getattr(encoder, "subword_embedding", None)
    with torch.no_grad():
        for root in (encoder, decoder):
            for module in root.modules():
                if module is suffix:
                    continue
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if getattr(module, "bias", None) is not None:
                        nn.init.zeros_(module.bias)
                elif "norm" in type(module).__name__.lower():
                    if getattr(module, "weight", None) is not None:
                        nn.init.ones_(module.weight)
                    if getattr(module, "bias", None) is not None:
                        nn.init.zeros_(module.bias)


def _support_training_lengths(student: nn.Module) -> None:
    """Pad mLSTM training internally; never use padded final recurrent states.

    Bolmo's HF wrapper unconditionally chooses the unpadded train kernel, but
    byte windows include a BOS and therefore are usually not a multiple of
    xLSTM's 64-position chunk.  Inference and cached generation remain on the
    untouched backend path.  A training forward never requests a final state.
    """
    groups = (student.model.local_encoder.layers, student.model.local_decoder.layers)
    for group in groups:
        for layer in group:
            backend = getattr(layer.xlstm, "mlstm_backend", None)
            if backend is None:
                continue
            if not hasattr(backend, "_train_fn") or not hasattr(backend, "config"):
                raise ValueError("unsupported xLSTM backend; expected the pinned mLSTM backend")
            from mlstm_kernels.torch.kernel_wrappers import wrap_chunkwise__pad_zeros
            backend_ref = weakref.ref(backend)
            original_ref = weakref.WeakMethod(backend.forward)

            def forward(*args: Any, _backend_ref=backend_ref, _original_ref=original_ref, **kwargs: Any):
                current = _backend_ref()
                original = _original_ref()
                if current is None or original is None:
                    raise RuntimeError("mLSTM backend was released")
                if current.config.mode != "train":
                    return original(*args, **kwargs)
                if args:
                    raise ValueError("unexpected positional mLSTM backend inputs")
                if kwargs.get("return_last_states") or any(kwargs.get(key) is not None
                                                             for key in ("c_initial", "n_initial", "m_initial")):
                    raise ValueError("training with padded local states is unsupported")
                kwargs.pop("return_last_states", None)
                output = wrap_chunkwise__pad_zeros(mlstm_chunkwise_kernel=current._train_fn,
                                                     return_last_states=False, chunk_size=current.config.chunk_size,
                                                     eps=current.config.eps,
                                                     autocast_kernel_dtype=getattr(torch, current.config.autocast_kernel_dtype),
                                                     **kwargs)
                return output, None

            backend.forward = forward


def build_student(teacher: nn.Module, selection: dict[str, Any]) -> nn.Module:
    """Make independent thin locals while retaining the teacher's frozen tensors.

    This freezes the teacher in place.  It never changes its parameter values.
    The returned model uses ordinary HF forward/generation methods.
    """
    if not isinstance(selection, dict) or selection.get("version") != 1:
        raise ValueError("selection version must be 1")
    allowed = {"version", "mode", "decoder_layers", "layers", "router", "linear_svd_rank"}
    if set(selection) - allowed:
        raise ValueError(f"unknown selection keys: {sorted(set(selection)-allowed)}")
    mode = selection.get("mode")
    if mode not in {"slice", "svd", "random"}:
        raise ValueError("mode must be slice, svd, or random")
    backbone = teacher.model
    encoder, decoder = backbone.local_encoder, backbone.local_decoder
    width = int(teacher.config.hidden_size)
    if int(teacher.config.vocab_size) != 520:
        raise ValueError("expected the released 520-way Bolmo vocabulary")
    _expect_shape(teacher.lm_head.weight, (520, width), "lm_head.weight")
    _expect_shape(encoder.byte_embedding.weight, (520, width), "local_encoder.byte_embedding.weight")
    suffix = getattr(encoder, "subword_embedding", None)
    if bool(encoder.add_expanded_embeddings) != (suffix is not None):
        raise ValueError("inconsistent suffix embedding configuration")
    if suffix is not None:
        _expect_shape(suffix.weight, (int(teacher.config.subword_vocab_size), width), "subword_embedding.weight")
    if (len(encoder.layers) != int(getattr(teacher.config, "num_local_encoder_layers", len(encoder.layers)))
            or len(decoder.layers) != int(teacher.config.num_local_decoder_layers)):
        raise ValueError("local layer counts differ from the checkpoint config")
    for layer in list(encoder.layers) + list(decoder.layers):
        if (int(layer.xlstm.config.num_heads) != int(teacher.config.num_local_heads)
                or int(layer.mlp.gate_proj.out_features) != int(teacher.config.local_intermediate_size)):
            raise ValueError("local head or FFN dimensions differ from the checkpoint config")
    for name, norm in (("encoder.post_last_block_norm", encoder.post_last_block_norm),
                       ("decoder.initial_norm", decoder.initial_norm)):
        _expect_shape(norm.weight, (width,), name + ".weight")
    for name, layer in (("encoder.out_projection", encoder.out_projection),
                        ("decoder.in_projection", decoder.in_projection)):
        if not isinstance(layer, nn.Linear):
            raise ValueError(f"{name} must be linear")
        _expect_shape(layer.weight, (width, width), name + ".weight")
        _expect_shape(layer.bias, (width,), name + ".bias")
    expected_encoder = {"byte_embedding.weight", "post_last_block_norm.weight", "out_projection.weight",
                        "out_projection.bias", "boundary_predictor_module.q_proj_layer.weight",
                        "boundary_predictor_module.k_proj_layer.weight"}
    if suffix is not None:
        expected_encoder.add("subword_embedding.weight")
    expected_encoder.update(f"layers.{i}.{key}" for i, layer in enumerate(encoder.layers)
                            for key in dict(layer.named_parameters()))
    expected_decoder = {"initial_norm.weight", "in_projection.weight", "in_projection.bias"}
    expected_decoder.update(f"layers.{i}.{key}" for i, layer in enumerate(decoder.layers)
                            for key in dict(layer.named_parameters()))
    if set(dict(encoder.named_parameters())) != expected_encoder or set(dict(decoder.named_parameters())) != expected_decoder:
        raise ValueError("unexpected local encoder/decoder parameter names")
    predictor = encoder.boundary_predictor_module
    if predictor.boundary_predictor_lookahead != 1 or predictor.boundary_threshold != "sample:0":
        raise ValueError("expected the released deterministic lookahead-one boundary predictor")
    kept = _indices(selection.get("decoder_layers"), len(decoder.layers), "decoder_layers")
    if kept != sorted(kept):
        raise ValueError("decoder layers must retain their original forward order")
    choices = selection.get("layers")
    expected_keys = {f"encoder.{i}" for i in range(len(encoder.layers))} | {f"decoder.{i}" for i in kept}
    if not isinstance(choices, dict) or set(choices) != expected_keys:
        raise ValueError(f"layers must select exactly {sorted(expected_keys)}")
    teacher.requires_grad_(False)
    memo: dict[int, Any] = {id(backbone.layers): backbone.layers, id(backbone.norm): backbone.norm,
                            id(backbone.rotary_embs): backbone.rotary_embs, id(teacher.lm_head): teacher.lm_head}
    if suffix is not None:
        memo[id(suffix)] = suffix
    if getattr(backbone, "_tokenizer", None) is not None:
        memo[id(backbone._tokenizer)] = backbone._tokenizer
    for i, layer in enumerate(encoder.layers):
        memo[id(layer)] = _thin_layer(layer, choices[f"encoder.{i}"], width, f"encoder.{i}")
    for i, layer in enumerate(decoder.layers):
        memo[id(layer)] = (_thin_layer(layer, choices[f"decoder.{i}"], width, f"decoder.{i}")
                           if i in kept else layer)
    memo[id(predictor)] = _router(predictor, selection.get("router"), mode, width)
    student = copy.deepcopy(teacher, memo)
    student.model.local_decoder.layers = nn.ModuleList([memo[id(decoder.layers[i])] for i in kept])
    student.config.num_local_decoder_layers = len(kept)
    student.model.config.num_local_decoder_layers = len(kept)
    selected_layers = list(student.model.local_encoder.layers) + list(student.model.local_decoder.layers)
    head_counts = {int(layer.xlstm.config.num_heads) for layer in selected_layers}
    ffn_sizes = {int(layer.mlp.gate_proj.out_features) for layer in selected_layers}
    if len(head_counts) == 1:
        student.config.num_local_heads = student.model.config.num_local_heads = head_counts.pop()
    if len(ffn_sizes) == 1:
        student.config.local_intermediate_size = student.model.config.local_intermediate_size = ffn_sizes.pop()
    ranks = selection.get("linear_svd_rank")
    if ranks is not None:
        if type(ranks) is not int or mode not in {"svd", "random"}:
            raise ValueError("linear_svd_rank requires an integer and svd or random mode")
        student.model.local_encoder.out_projection = FactorizedLinear(encoder.out_projection, ranks, random_init=mode == "random")
        student.model.local_decoder.in_projection = FactorizedLinear(decoder.in_projection, ranks, random_init=mode == "random")
    if mode == "random":
        _randomize_locals(student)
    for root in (student.model.local_encoder, student.model.local_decoder):
        for name, parameter in root.named_parameters():
            if not name.startswith("subword_embedding."):
                parameter.requires_grad_(True)
    if student.model.layers is not backbone.layers or student.lm_head is not teacher.lm_head:
        raise AssertionError("global trunk/head sharing failed")
    if suffix is not None and student.model.local_encoder.subword_embedding is not suffix:
        raise AssertionError("suffix table sharing failed")
    _support_training_lengths(student)
    student._thin_selection = copy.deepcopy(selection)
    return student


def full_selection(teacher: nn.Module) -> dict[str, Any]:
    """Construct the identity selection for parity checks, never a thin ranking."""
    encoder, decoder = teacher.model.local_encoder, teacher.model.local_decoder
    layers = {}
    for prefix, group in (("encoder", encoder.layers), ("decoder", decoder.layers)):
        for i, layer in enumerate(group):
            layers[f"{prefix}.{i}"] = {"heads": list(range(layer.xlstm.config.num_heads)),
                                        "ffn": list(range(layer.mlp.gate_proj.out_features))}
    return {"version": 1, "mode": "slice", "decoder_layers": list(range(len(decoder.layers))),
            "layers": layers, "router": {"dims": list(range(teacher.config.hidden_size))}}


def dense_local_cost(student: nn.Module) -> dict[str, float | str]:
    """Count dense multiply/adds, separating per-byte and per-patch maps.

    This is an arithmetic estimate, not a timing measurement.  It excludes
    recurrent kernel arithmetic, norms, embeddings, padding, and the stock
    generation path's extra encoder pass.
    """
    encoder, decoder = student.model.local_encoder, student.model.local_decoder

    def matrix_parameters(module: nn.Module) -> int:
        return sum(child.weight.numel() for child in module.modules() if isinstance(child, nn.Linear))

    blocks = sum(matrix_parameters(layer.xlstm) + matrix_parameters(layer.mlp)
                 for layer in list(encoder.layers) + list(decoder.layers))
    byte_macs = (blocks + matrix_parameters(encoder.boundary_predictor_module)
                 + matrix_parameters(decoder.in_projection) + matrix_parameters(student.lm_head))
    patch_macs = matrix_parameters(encoder.out_projection)
    return {"dense_gflops_per_byte": 2 * byte_macs / 1e9,
            "encoder_projection_gflops_per_patch": 2 * patch_macs / 1e9,
            "description": "two FLOPs per dense multiply-add; head counted per byte; excludes recurrent arithmetic"}


@torch.inference_mode()
def importance_stats(teacher: nn.Module, windows: Iterable[Any], output: str | Path | None = None) -> dict[str, Any]:
    """Measure calibration activations and leave selection to the researcher.

    Head importance is the mean L2 norm of each head's contribution *after*
    its slice of out_proj.  FFN importance is mean absolute post-gate
    activation times the L2 norm of its down-projection column.  Ablations
    bypass one complete decoder block with the original teacher boundaries;
    deltas are in folded byte NLL (nats/byte).  Router statistics are
    per-output-coordinate means of absolute value and squared value.
    """
    encoder, decoder = teacher.model.local_encoder, teacher.model.local_decoder
    predictor = encoder.boundary_predictor_module
    if predictor.boundary_predictor_lookahead != 1:
        raise ValueError("calibration expects the released lookahead-one router")
    device = next(teacher.parameters()).device
    old_training = teacher.training
    teacher.eval()
    groups = {f"encoder.{i}": layer for i, layer in enumerate(encoder.layers)}
    groups.update({f"decoder.{i}": layer for i, layer in enumerate(decoder.layers)})
    sums: dict[str, dict[str, Any]] = {}
    router_sums: dict[str, dict[str, Any]] = {}
    handles: list[Any] = []
    enabled = [True]
    boundary: list[torch.Tensor] = []
    hashes, byte_count, baseline = [], 0, 0.0
    ablated = [0.0] * len(decoder.layers)

    def head_hook(key: str, block: nn.Module):
        def observe(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not enabled[0]:
                return
            x = inputs[0].detach()
            heads = int(block.xlstm.config.num_heads)
            vdim = int(block.xlstm.v_dim)
            if x.shape[-1] != vdim or vdim % heads:
                raise ValueError(f"{key}: unexpected out_proj activation shape")
            size = vdim // heads
            weight = block.xlstm.out_proj.weight.detach()
            contributions = []
            for i in range(heads):
                part = F.linear(x[..., i * size:(i + 1) * size].float(),
                                weight[:, i * size:(i + 1) * size].float())
                contributions.append(part.norm(dim=-1).sum())
            entry = sums.setdefault(key, {"head_sum": torch.zeros(heads, dtype=torch.float64),
                                          "head_count": 0, "ffn_sum": None, "ffn_count": 0})
            entry["head_sum"] += torch.stack(contributions).cpu().double()
            entry["head_count"] += x.numel() // vdim
        return observe

    def ffn_hook(key: str, block: nn.Module):
        def observe(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not enabled[0]:
                return
            x = inputs[0].detach().float()
            channels = block.mlp.down_proj.weight.shape[1]
            if x.shape[-1] != channels:
                raise ValueError(f"{key}: unexpected FFN activation shape")
            weighted = x.abs().reshape(-1, channels).sum(0) * block.mlp.down_proj.weight.detach().float().norm(dim=0)
            entry = sums.setdefault(key, {"head_sum": None, "head_count": 0,
                                          "ffn_sum": torch.zeros(channels, dtype=torch.float64), "ffn_count": 0})
            if entry["ffn_sum"] is None:
                entry["ffn_sum"] = torch.zeros(channels, dtype=torch.float64)
            entry["ffn_sum"] += weighted.cpu().double()
            entry["ffn_count"] += x.numel() // channels
        return observe

    def router_hook(key: str):
        def observe(_module: nn.Module, _args: tuple[Any, ...], value: torch.Tensor) -> None:
            if not enabled[0]:
                return
            x = value.detach().float().reshape(-1, value.shape[-1])
            entry = router_sums.setdefault(key, {"abs": torch.zeros(x.shape[-1], dtype=torch.float64),
                                                  "square": torch.zeros(x.shape[-1], dtype=torch.float64),
                                                  "count": 0})
            entry["abs"] += x.abs().sum(0).cpu().double()
            entry["square"] += x.square().sum(0).cpu().double()
            entry["count"] += x.shape[0]
        return observe

    def capture_boundary(_module: nn.Module, _args: tuple[Any, ...], result: tuple[Any, ...]) -> None:
        if enabled[0]:
            boundary.append(result[3].detach().clone())

    for key, block in groups.items():
        handles.append(block.xlstm.out_proj.register_forward_pre_hook(head_hook(key, block)))
        handles.append(block.mlp.down_proj.register_forward_pre_hook(ffn_hook(key, block)))
    handles.append(predictor.q_proj_layer.register_forward_hook(router_hook("q")))
    handles.append(predictor.k_proj_layer.register_forward_hook(router_hook("k")))
    handles.append(encoder.register_forward_hook(capture_boundary))
    try:
        if encoder.has_cache or decoder.has_cache:
            raise ValueError("clear Bolmo inference caches before collecting statistics")
        for window in windows:
            raw = window if isinstance(window, bytes) else window.data
            if not isinstance(raw, bytes) or not raw:
                raise ValueError("calibration windows must contain nonempty bytes")
            ids = bolmo_ids(raw, device)
            boundary.clear()
            enabled[0] = True
            logits = teacher(input_ids=ids, use_cache=False).logits[:, :-1]
            if len(boundary) != 1 or boundary[0].shape != ids.shape:
                raise RuntimeError("failed to capture teacher prefill boundaries")
            targets = torch.tensor(list(raw), dtype=torch.long, device=device)[None, :, None]
            baseline += float(-folded_log_probs(logits).gather(-1, targets).sum().item())
            fixed = boundary[0]
            enabled[0] = False
            for i, layer in enumerate(decoder.layers):
                handle = layer.register_forward_hook(lambda _m, args, _out: args[0])
                try:
                    logits = teacher(input_ids=ids, use_cache=False, boundary_mask=fixed).logits[:, :-1]
                    ablated[i] += float(-folded_log_probs(logits).gather(-1, targets).sum().item())
                finally:
                    handle.remove()
            hashes.append(hashlib.sha256(raw).hexdigest())
            byte_count += len(raw)
        if not byte_count:
            raise ValueError("no calibration windows")
        layer_stats = {}
        for key, entry in sums.items():
            if not entry["head_count"] or not entry["ffn_count"]:
                raise RuntimeError(f"missing calibration activations for {key}")
            layer_stats[key] = {"head_output_norm_mean": (entry["head_sum"] / entry["head_count"]).tolist(),
                                "ffn_activation_times_down_norm_mean": (entry["ffn_sum"] / entry["ffn_count"]).tolist()}
        if set(layer_stats) != set(groups) or set(router_sums) != {"q", "k"}:
            raise RuntimeError("not all local layers and router projections were observed")
        result = {"version": 1, "window_sha256": hashes, "bytes": byte_count,
                  "definitions": {"head": "mean L2 norm of the head's out_proj contribution",
                                  "ffn": "mean absolute SwiGLU activation times L2 norm of down column",
                                  "decoder": "folded byte NLL delta (nats/byte), fixed teacher layout",
                                  "router": "coordinate mean absolute value and root mean square"},
                  "layers": layer_stats,
                  "decoder_ablation_delta_nats_per_byte": [(value - baseline) / byte_count for value in ablated],
                  "teacher_nats_per_byte": baseline / byte_count,
                  "router": {key: {"mean_abs": (entry["abs"] / entry["count"]).tolist(),
                                   "rms": (entry["square"] / entry["count"]).sqrt().tolist()}
                             for key, entry in router_sums.items()}}
        if output is not None:
            atomic_json(Path(output), result)
        return result
    finally:
        for handle in handles:
            handle.remove()
        teacher.train(old_training)


class LoRALinear(nn.Module):
    """A zero-initialized additive update around an unchanged shared linear."""
    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        if not isinstance(base, nn.Linear) or not 0 < rank <= min(base.in_features, base.out_features):
            raise ValueError("invalid LoRA linear or rank")
        self.base = base
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.normal_(self.a, std=0.02)
        self.scale = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * F.linear(F.linear(x, self.a), self.b)


def configure_trainable(student: nn.Module, *, unfreeze_suffix: bool = False, unfreeze_head: bool = False,
                        lora_rank: int = 0, float32: bool = True) -> None:
    """Configure deltas without ever modifying the teacher's shared tensors."""
    if unfreeze_suffix:
        suffix = student.model.local_encoder.subword_embedding
        if suffix is None:
            raise ValueError("the model has no suffix embedding")
        student.model.local_encoder.subword_embedding = copy.deepcopy(suffix)
        student.model.local_encoder.subword_embedding.requires_grad_(True)
    if unfreeze_head:
        student.lm_head = copy.deepcopy(student.lm_head)
        student.lm_head.requires_grad_(True)
    if lora_rank:
        # Copy only the module shells. Every base parameter/buffer remains shared.
        global_layers = student.model.layers
        memo = {id(value): value for value in list(global_layers.parameters()) + list(global_layers.buffers())}
        student.model.layers = copy.deepcopy(global_layers, memo)
        for layer in student.model.layers:
            attention = getattr(layer, "self_attn", None)
            if attention is None:
                raise ValueError("LoRA requires global attention layers with q_proj and v_proj")
            for name in ("q_proj", "v_proj"):
                base = getattr(attention, name, None)
                if not isinstance(base, nn.Linear):
                    raise ValueError(f"LoRA requires self_attn.{name} to be a linear")
                setattr(attention, name, LoRALinear(base, lora_rank))
    if float32:
        # Trainable parameters are independent locals/deltas; frozen shared
        # tensors must never be cast in place.
        for parameter in student.parameters():
            if parameter.requires_grad and parameter.is_floating_point():
                parameter.data = parameter.data.float()


def _autocast(device: torch.device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _inputs(teacher: nn.Module, raw: bytes, device: torch.device) -> dict[str, torch.Tensor]:
    ids = bolmo_ids(raw, device)
    result = {"input_ids": ids}
    if getattr(teacher.model.local_encoder, "add_expanded_embeddings", False):
        tokenizer = getattr(teacher.model, "tokenizer", None)
        if tokenizer is not None:
            expanded = tokenizer.expand_byte_ids([1, *(byte + 4 for byte in raw)])
            if len(expanded) != ids.shape[1]:
                raise ValueError("Bolmo suffix tokenizer returned a misaligned sequence")
            result["expanded_input_ids"] = torch.tensor([expanded], dtype=torch.long, device=device)
    return result


def _forward_capture(model: nn.Module, inputs: dict[str, torch.Tensor], boundary_mask: torch.Tensor | None,
                     *, capture_patches: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    encoder = model.model.local_encoder
    captured: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []

    def capture(_module: nn.Module, _args: tuple[Any, ...], output: tuple[Any, ...]) -> None:
        # The HF forward may later rearrange the patch tensor in place.
        patch = output[1].clone() if capture_patches else None
        captured.append((output[2], output[3].detach().clone(), patch))

    handle = encoder.register_forward_hook(capture)
    try:
        logits = model(**inputs, use_cache=False, boundary_mask=boundary_mask).logits
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0][0] is None or captured[0][1].shape != inputs["input_ids"].shape:
        raise RuntimeError("failed to capture Bolmo prefill router outputs")
    logp, layout, patches = captured[0]
    return logits[:, :-1], logp, layout, patches


def distillation_loss(teacher: nn.Module, student: nn.Module, raw: bytes, *, mse_weight: float = 0.0,
                      ce_weight: float = 1.0, boundary_weight: float = 1.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Teacher-forced 520-way KL + folded byte CE + boundary BCE.

    The layout is the teacher's full one-lookahead prefill mask.  BOS and the
    last lookahead sentinel are excluded from router supervision.
    """
    if min(mse_weight, ce_weight, boundary_weight) < 0:
        raise ValueError("loss weights cannot be negative")
    device = next(student.parameters()).device
    inputs = _inputs(teacher, raw, device)
    teacher.eval()
    with torch.no_grad(), _autocast(device):
        teacher_logits, _, layout, teacher_patches = _forward_capture(teacher, inputs, None,
                                                                      capture_patches=mse_weight > 0)
    student.train()
    with _autocast(device):
        logits, logp, _, student_patches = _forward_capture(student, inputs, layout,
                                                             capture_patches=mse_weight > 0)
    target = torch.tensor(list(raw), dtype=torch.long, device=device)[None, :, None]
    teacher_logp = teacher_logits.float().log_softmax(-1)
    student_logp = logits.float().log_softmax(-1)
    kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1).mean()
    ce = -torch.logaddexp(student_logp[..., 4:260], student_logp[..., 264:520]).gather(-1, target).mean()
    teacher_ce = -torch.logaddexp(teacher_logp[..., 4:260], teacher_logp[..., 264:520]).gather(-1, target).mean()
    router_logp = logp[:, 1:-1].float()
    boundary_target = layout[:, 1:-1].float()
    if router_logp.numel():
        safe_logp = router_logp.clamp(min=-30.0, max=-1e-6)
        boundary_ce = -(boundary_target * safe_logp + (1 - boundary_target) * torch.log(-torch.expm1(safe_logp))).mean()
        accuracy = ((router_logp > math.log(0.5)) == boundary_target.bool()).float().mean()
    else:
        boundary_ce = logits.sum() * 0
        accuracy = logits.new_tensor(1.0)
    if mse_weight:
        if teacher_patches is None or student_patches is None or teacher_patches.shape != student_patches.shape:
            raise RuntimeError("teacher and student pooled representations do not align")
        mse = F.mse_loss(student_patches.float(), teacher_patches.float())
    else:
        mse = logits.new_zeros((), dtype=torch.float32)
    loss = kl + ce_weight * ce + boundary_weight * boundary_ce + mse_weight * mse
    values = torch.stack((kl.detach(), ce.detach(), boundary_ce.detach(), accuracy.detach(), mse.detach(),
                          (ce.detach() / math.log(2)), (teacher_ce.detach() / math.log(2)))).cpu().tolist()
    names = ("kl", "ce", "boundary_bce", "boundary_accuracy", "patch_mse", "student_bpb", "teacher_bpb")
    return loss, dict(zip(names, values))


@torch.no_grad()
def evaluate_pair(teacher: nn.Module, student: nn.Module, raw: bytes, *, own_routing: bool = False) -> dict[str, Any]:
    device = next(student.parameters()).device
    inputs = _inputs(teacher, raw, device)
    teacher.eval()
    student.eval()
    with _autocast(device):
        teacher_logits, _, layout, _ = _forward_capture(teacher, inputs, None)
        student_logits, logp, student_layout, _ = _forward_capture(student, inputs, None if own_routing else layout)
    target = torch.tensor(list(raw), dtype=torch.long, device=device)[None, :, None]
    teacher_nats = float(-folded_log_probs(teacher_logits).gather(-1, target).sum().item())
    student_nats = float(-folded_log_probs(student_logits).gather(-1, target).sum().item())
    valid = logp[:, 1:-1]
    accuracy = float(((valid > math.log(0.5)) == layout[:, 1:-1]).float().mean().item()) if valid.numel() else 1.0
    teacher_patches = int(layout.sum().item())
    student_patches = int(student_layout.sum().item())
    if not teacher_patches or not student_patches or not math.isfinite(teacher_nats + student_nats):
        raise RuntimeError("invalid routed evaluation result")
    return {"bytes": len(raw), "teacher_nats": teacher_nats, "student_nats": student_nats,
            "teacher_patches": teacher_patches, "student_patches": student_patches,
            "boundary_accuracy": accuracy}


@dataclass(frozen=True)
class DistillConfig:
    window_bytes: int = 2048
    windows_per_step: int = 1
    init_seed: int = 0
    data_seed: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 100
    schedule_steps: int = 10000
    val_interval_steps: int = 500
    mse_weight: float = 0.0
    ce_weight: float = 1.0
    boundary_weight: float = 1.0
    lora_rank: int = 0
    unfreeze_suffix: bool = False
    unfreeze_head: bool = False
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        if min(self.window_bytes, self.windows_per_step, self.warmup_steps, self.schedule_steps,
               self.val_interval_steps) <= 0 or self.learning_rate <= 0 or self.grad_clip <= 0:
            raise ValueError("invalid distillation training settings")
        if min(self.weight_decay, self.mse_weight, self.ce_weight, self.boundary_weight, self.lora_rank) < 0:
            raise ValueError("negative distillation setting")


def _checkpoint_local_layers(student: nn.Module) -> None:
    from torch.utils.checkpoint import checkpoint

    for group in (student.model.local_encoder.layers, student.model.local_decoder.layers):
        for layer in group:
            layer_ref = weakref.ref(layer)
            original_ref = weakref.WeakMethod(layer.forward)

            def forward(*args: Any, _original_ref=original_ref, _layer_ref=layer_ref, **kwargs: Any):
                current, original = _layer_ref(), _original_ref()
                if current is None or original is None:
                    raise RuntimeError("local layer was released")
                if current.training and torch.is_grad_enabled() and not kwargs.get("use_cache", False):
                    return checkpoint(original, *args, use_reentrant=False, **kwargs)
                return original(*args, **kwargs)

            layer.forward = forward


def load_student_checkpoint(teacher: nn.Module, checkpoint_path: str | Path,
                            teacher_identity: dict[str, Any]) -> tuple[nn.Module, dict[str, Any]]:
    """Recreate a student from a delta checkpoint and its exact teacher."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("version") != 1 or checkpoint.get("teacher_identity") != teacher_identity:
        raise ValueError("student checkpoint version or teacher identity mismatch")
    selection, cfg = checkpoint["selection"], checkpoint["config"]
    student = build_student(teacher, selection)
    configure_trainable(student, unfreeze_suffix=cfg["unfreeze_suffix"], unfreeze_head=cfg["unfreeze_head"],
                        lora_rank=cfg["lora_rank"])
    params = {name: value for name, value in student.named_parameters() if value.requires_grad}
    saved = checkpoint["trainable"]
    if set(params) != set(saved):
        raise ValueError("checkpoint trainable tensor names differ from the reconstructed student")
    with torch.no_grad():
        for name, parameter in params.items():
            value = saved[name]
            if parameter.shape != value.shape or parameter.dtype != value.dtype:
                raise ValueError(f"checkpoint tensor shape or dtype mismatch: {name}")
            parameter.copy_(value.to(parameter.device))
    student.eval()
    # Optimizer states and RNG do not belong in an evaluation result.
    metadata = {key: checkpoint[key] for key in ("version", "selection", "teacher_identity", "config", "step",
                                                 "consumed_bytes", "data_manifest_hash", "window_index_sha256",
                                                 "validation_window_sha256", "git_revision")}
    return student, metadata


class DistillTrainer:
    """Deterministic single-window-microbatch training with delta checkpoints."""
    def __init__(self, teacher: nn.Module, student: nn.Module, selection: dict[str, Any],
                 teacher_identity: dict[str, Any], windows: Any, manifest_hash: str,
                 val_windows: Iterable[Any], run_dir: str | Path, config: DistillConfig):
        self.teacher, self.student = teacher, student
        self.selection, self.teacher_identity = copy.deepcopy(selection), copy.deepcopy(teacher_identity)
        self.windows, self.manifest_hash, self.config = windows, manifest_hash, config
        self.run_dir = Path(run_dir)
        self.device = next(student.parameters()).device
        if next(teacher.parameters()).device != self.device:
            raise ValueError("teacher and student must reside on the same device")
        if getattr(student, "_thin_selection", None) != selection:
            raise ValueError("student selection does not match the checkpoint selection")
        if not len(windows):
            raise ValueError("empty training window index")
        self.val_windows = [window if isinstance(window, bytes) else window.data for window in val_windows]
        if not self.val_windows or any(not isinstance(raw, bytes) or not raw for raw in self.val_windows):
            raise ValueError("the fixed validation subset must contain nonempty byte windows")
        self.val_hashes = [hashlib.sha256(raw).hexdigest() for raw in self.val_windows]
        self.index_hash = getattr(windows, "index_sha256", None)
        if self.index_hash is None:
            raise ValueError("training window index must expose index_sha256")
        configure_trainable(student, unfreeze_suffix=config.unfreeze_suffix, unfreeze_head=config.unfreeze_head,
                            lora_rank=config.lora_rank)
        if config.gradient_checkpointing:
            _checkpoint_local_layers(student)
            if hasattr(student, "gradient_checkpointing_enable"):
                student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.trainables = {name: value for name, value in student.named_parameters() if value.requires_grad}
        if not self.trainables:
            raise ValueError("student has no trainable tensors")
        embedding_ids = {id(module.weight) for module in student.modules() if isinstance(module, nn.Embedding)}
        decay = [p for p in self.trainables.values() if p.ndim >= 2 and id(p) not in embedding_ids]
        no_decay = [p for p in self.trainables.values() if p.ndim < 2 or id(p) in embedding_ids]
        self.optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": config.weight_decay},
                                           {"params": no_decay, "weight_decay": 0.0}],
                                          lr=config.learning_rate, betas=(0.9, 0.95))

        def lr_factor(step: int) -> float:
            if step < config.warmup_steps:
                return (step + 1) / config.warmup_steps
            progress = min(1.0, (step - config.warmup_steps) / max(1, config.schedule_steps - config.warmup_steps))
            return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_factor)
        self.step = self.consumed_bytes = self.checkpoint_serial = 0
        self.run_dir.mkdir(parents=True, exist_ok=True)
        frozen = {"config": asdict(config), "selection": self.selection, "teacher": self.teacher_identity,
                  "data_manifest_hash": manifest_hash, "window_index_sha256": self.index_hash,
                  "validation_window_sha256": self.val_hashes}
        config_path = self.run_dir / "config.json"
        if config_path.exists():
            if json.loads(config_path.read_text(encoding="utf-8")) != frozen:
                raise ValueError("run configuration or data/teacher identity differs")
        else:
            atomic_json(config_path, frozen)

    def sample_indices(self, step: int) -> list[int]:
        return [sample_window_index(self.config.data_seed, step, i, self.config.windows_per_step, len(self.windows))
                for i in range(self.config.windows_per_step)]

    def sample_audit(self, step: int) -> dict[str, Any]:
        indices = self.sample_indices(step) if step >= 0 else []
        return {"indices": indices, "window_sha256": [hashlib.sha256(self.windows[i].data).hexdigest() for i in indices]}

    def train_step(self) -> dict[str, Any]:
        indexes = self.sample_indices(self.step)
        batch = [self.windows[index].data for index in indexes]
        if any(not raw or len(raw) > self.config.window_bytes for raw in batch):
            raise ValueError("invalid training window")
        total_bytes = sum(map(len, batch))
        total_decisions = sum(max(0, len(raw) - 1) for raw in batch)
        self.optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        for raw in batch:
            loss, metrics = distillation_loss(self.teacher, self.student, raw, mse_weight=self.config.mse_weight,
                                               ce_weight=self.config.ce_weight, boundary_weight=self.config.boundary_weight)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("non-finite distillation loss")
            (loss * (len(raw) / total_bytes)).backward()
            for key, value in metrics.items():
                if key == "boundary_accuracy":
                    weight = max(0, len(raw) - 1) / total_decisions if total_decisions else 1 / len(batch)
                else:
                    weight = len(raw) / total_bytes
                sums[key] = sums.get(key, 0.0) + value * weight
        norm = torch.nn.utils.clip_grad_norm_(self.trainables.values(), self.config.grad_clip, error_if_nonfinite=True)
        lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        self.consumed_bytes += total_bytes
        return {**sums, "step": self.step, "consumed_bytes": self.consumed_bytes, "lr": lr,
                "grad_norm": float(norm.item()), "sample_indices": indexes}

    def validate(self, stop: threading.Event | None = None) -> dict[str, float] | None:
        records = []
        for raw in self.val_windows:
            if stop is not None and stop.is_set():
                return None
            records.append(evaluate_pair(self.teacher, self.student, raw))
        count = sum(row["bytes"] for row in records)
        decisions = sum(max(0, row["bytes"] - 1) for row in records)
        accuracy = (sum(row["boundary_accuracy"] * max(0, row["bytes"] - 1) for row in records) / decisions
                    if decisions else 1.0)
        return {"student_bpb": sum(row["student_nats"] for row in records) / (math.log(2) * count),
                "teacher_bpb": sum(row["teacher_nats"] for row in records) / (math.log(2) * count),
                "boundary_accuracy": accuracy}

    def save(self) -> Path:
        self.checkpoint_serial += 1
        payload = {"version": 1, "trainable": {name: p.detach().cpu().clone() for name, p in self.trainables.items()},
                   "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                   "selection": self.selection, "teacher_identity": self.teacher_identity, "config": asdict(self.config),
                   "data_manifest_hash": self.manifest_hash, "window_index_sha256": self.index_hash,
                   "validation_window_sha256": self.val_hashes, "git_revision": git_revision(),
                   "step": self.step, "consumed_bytes": self.consumed_bytes,
                   "sampler_cursor": self.step * self.config.windows_per_step, "checkpoint_serial": self.checkpoint_serial,
                   "sample_audit": self.sample_audit(self.step - 1),
                   "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                   "rng_torch": torch.get_rng_state(),
                   "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        directory = self.run_dir / "ckpt"
        path = directory / f"rolling-{self.checkpoint_serial % 2}.pt"
        _atomic_torch_save(path, payload)
        atomic_json(directory / "latest.json", {"checkpoint": path.name, "step": self.step,
                                                "consumed_bytes": self.consumed_bytes, "serial": self.checkpoint_serial})
        return path

    def load(self, path: str | Path | None = None) -> None:
        if path is None:
            directory = self.run_dir / "ckpt"
            pointer = json.loads((directory / "latest.json").read_text(encoding="utf-8"))
            path = directory / pointer["checkpoint"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        expected = {"version": 1, "selection": self.selection, "teacher_identity": self.teacher_identity,
                    "config": asdict(self.config), "data_manifest_hash": self.manifest_hash,
                    "window_index_sha256": self.index_hash, "validation_window_sha256": self.val_hashes}
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise ValueError("checkpoint configuration or base/data identity mismatch")
        if checkpoint["sampler_cursor"] != checkpoint["step"] * self.config.windows_per_step:
            raise ValueError("invalid checkpoint sampler cursor")
        if checkpoint["sample_audit"] != self.sample_audit(checkpoint["step"] - 1):
            raise ValueError("checkpoint sample audit differs from the current data")
        saved = checkpoint["trainable"]
        if set(saved) != set(self.trainables):
            raise ValueError("checkpoint trainable tensor names differ")
        for name, parameter in self.trainables.items():
            if saved[name].shape != parameter.shape or saved[name].dtype != parameter.dtype:
                raise ValueError(f"checkpoint trainable tensor shape or dtype mismatch: {name}")
        with torch.no_grad():
            for name, parameter in self.trainables.items():
                parameter.copy_(saved[name].to(parameter.device))
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.step, self.consumed_bytes = checkpoint["step"], checkpoint["consumed_bytes"]
        self.checkpoint_serial = checkpoint["checkpoint_serial"]
        random.setstate(checkpoint["rng_python"])
        np.random.set_state(checkpoint["rng_numpy"])
        torch.set_rng_state(checkpoint["rng_torch"].cpu())
        if checkpoint["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([value.cpu() for value in checkpoint["rng_cuda"]])
        _truncate_uncommitted_log(self.run_dir / "log.jsonl", self.step)

    def run(self, stop_bytes: int, pause_file: str | Path, *, checkpoint_seconds: float = 900.0,
            poll_seconds: float = 2.0, stop_event: threading.Event | None = None) -> None:
        if stop_bytes <= 0 or checkpoint_seconds <= 0 or not 0 < poll_seconds <= 10:
            raise ValueError("invalid stop or polling settings")
        stop = stop_event if stop_event is not None else threading.Event()
        done, pause = threading.Event(), Path(pause_file)
        previous_handler = signal.getsignal(signal.SIGTERM)

        def terminate(_signum: int, _frame: Any) -> None:
            stop.set()

        def watch() -> None:
            while not done.is_set():
                if pause.exists():
                    stop.set()
                done.wait(poll_seconds)

        signal.signal(signal.SIGTERM, terminate)
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        last_save = last_log = time.monotonic()
        previous_bytes = self.consumed_bytes
        dirty = False
        try:
            if not self.checkpoint_serial:
                if (self.run_dir / "ckpt/latest.json").exists():
                    raise ValueError("existing run requires loading its checkpoint before training")
                self.save()
            if pause.exists():
                stop.set()
            while self.consumed_bytes < stop_bytes and not stop.is_set():
                metrics = self.train_step()
                dirty = True
                if self.step % self.config.val_interval_steps == 0 and not stop.is_set():
                    metrics["validation"] = self.validate(stop)
                now = time.monotonic()
                metrics["bytes_per_second"] = (self.consumed_bytes - previous_bytes) / max(now - last_log, 1e-9)
                metrics["peak_memory_bytes"] = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
                with open(self.run_dir / "log.jsonl", "a", encoding="utf-8", newline="\n") as log:
                    log.write(json.dumps(metrics, sort_keys=True) + "\n")
                    log.flush()
                    os.fsync(log.fileno())
                previous_bytes, last_log = self.consumed_bytes, now
                if now - last_save >= checkpoint_seconds:
                    self.save()
                    dirty, last_save = False, time.monotonic()
                atomic_json(self.run_dir / "status.json", {"status": "running", "step": self.step,
                                                            "consumed_bytes": self.consumed_bytes})
            state = "paused" if stop.is_set() else "complete"
            if dirty:
                self.save()
            atomic_json(self.run_dir / "status.json", {"status": state, "step": self.step,
                                                        "consumed_bytes": self.consumed_bytes})
            outcome = os.environ.get("BLT_QUEUE_OUTCOME_FILE")
            if outcome:
                atomic_json(Path(outcome), {"status": state})
        except BaseException:
            atomic_json(self.run_dir / "status.json", {"status": "failed", "step": self.step,
                                                        "consumed_bytes": self.consumed_bytes})
            raise
        finally:
            done.set()
            watcher.join()
            signal.signal(signal.SIGTERM, previous_handler)
