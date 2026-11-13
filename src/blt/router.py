from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import torch
import torch.nn.functional as F


Q_NAME = "model.local_encoder.boundary_predictor_module.q_proj_layer.weight"
K_NAME = "model.local_encoder.boundary_predictor_module.k_proj_layer.weight"
HEAD_NAME = "lm_head.weight"
BYTE_START, BOUNDARY_START, BYTE_COUNT = 4, 264, 256


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


@dataclass
class Snapshot:
    tensors: dict[str, torch.Tensor]
    hashes: dict[str, str]


def snapshot(model: torch.nn.Module) -> Snapshot:
    tensors = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    return Snapshot(tensors, {name: tensor_sha256(value) for name, value in tensors.items()})


def restore(model: torch.nn.Module, reference: Snapshot) -> None:
    current = model.state_dict()
    if set(current) != set(reference.tensors):
        raise ValueError("model tensor names changed since the snapshot")
    for name, value in current.items():
        saved = reference.tensors[name]
        if value.shape != saved.shape or value.dtype != saved.dtype:
            raise ValueError(f"shape or dtype changed: {name}")
    model.load_state_dict(reference.tensors, strict=True)


def audit(model: torch.nn.Module, reference: Snapshot) -> list[dict[str, Any]]:
    current = model.state_dict()
    if set(current) != set(reference.tensors):
        raise ValueError("model tensor names changed since the snapshot")
    changed = []
    for name, tensor in current.items():
        old = reference.tensors[name]
        if tensor.shape != old.shape or tensor.dtype != old.dtype:
            raise ValueError(f"shape or dtype changed: {name}")
        now = tensor.detach().cpu().contiguous()
        digest = tensor_sha256(now)
        if digest != reference.hashes[name]:
            left, right = old.reshape(-1), now.reshape(-1)
            maximum = 0.0
            for start in range(0, left.numel(), 1_000_000):
                delta = (left[start:start + 1_000_000].double() - right[start:start + 1_000_000].double()).abs()
                maximum = max(maximum, float(delta.max().item()))
            changed.append({"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                            "sha256_before": reference.hashes[name], "sha256_after": digest, "max_abs_delta": maximum})
    return changed


def _parameters(model: torch.nn.Module) -> tuple[torch.nn.Parameter, torch.nn.Parameter, torch.nn.Parameter]:
    params = dict(model.named_parameters())
    missing = [name for name in (Q_NAME, K_NAME, HEAD_NAME) if name not in params]
    if missing:
        raise ValueError(f"Bolmo tensors missing: {missing}")
    q, k, head = params[Q_NAME], params[K_NAME], params[HEAD_NAME]
    if q.ndim != 2 or q.shape != k.shape or q.shape[0] != q.shape[1] or head.shape != (520, q.shape[1]):
        raise ValueError("unexpected Bolmo router or output-head tensor shapes")
    return q, k, head


@torch.no_grad()
def symmetrise_qk(model: torch.nn.Module, gamma: float) -> None:
    if not math.isfinite(gamma):
        raise ValueError("gamma must be finite")
    q, k, _ = _parameters(model)
    q_old, k_old = q.float(), k.float()
    mean, delta = (q_old + k_old) / 2, (q_old - k_old) / 2
    q.copy_((mean + gamma * delta).to(q.dtype))
    k.copy_((mean - gamma * delta).to(k.dtype))


@torch.no_grad()
def edit_paired_rows(model: torch.nn.Module, direction: torch.Tensor, eta: torch.Tensor) -> None:
    _, _, head = _parameters(model)
    if direction.shape != (head.shape[1],) or eta.shape != (BYTE_COUNT,):
        raise ValueError("direction must be [width] and eta must be [256]")
    if not torch.isfinite(direction).all() or not torch.isfinite(eta).all():
        raise ValueError("direction and eta must be finite")
    delta = eta.to(device=head.device, dtype=torch.float32)[:, None] * direction.to(device=head.device, dtype=torch.float32)[None, :] / 2
    low = head[BYTE_START:BYTE_START + BYTE_COUNT].float() - delta
    high = head[BOUNDARY_START:BOUNDARY_START + BYTE_COUNT].float() + delta
    head[BYTE_START:BYTE_START + BYTE_COUNT].copy_(low.to(head.dtype))
    head[BOUNDARY_START:BOUNDARY_START + BYTE_COUNT].copy_(high.to(head.dtype))


def adjust_joint_logits(logits: torch.Tensor, prior: torch.Tensor | None = None, *, beta: float = 0.0,
                        strength: float = 0.0) -> torch.Tensor:
    if logits.shape[-1] != 520 or not math.isfinite(beta) or not math.isfinite(strength):
        raise ValueError("expected 520 joint logits and finite offsets")
    source = logits.float()
    if beta == 0 and (prior is None or strength == 0):
        return source.clone()
    low = source[..., BYTE_START:BYTE_START + BYTE_COUNT]
    high = source[..., BOUNDARY_START:BOUNDARY_START + BYTE_COUNT]
    pair_mass = torch.logaddexp(low, high)
    odds = high - low + beta
    if prior is not None:
        odds = odds + strength * prior.to(device=logits.device, dtype=torch.float32)
    output = source.clone()
    output[..., BYTE_START:BYTE_START + BYTE_COUNT] = pair_mass + F.logsigmoid(-odds)
    output[..., BOUNDARY_START:BOUNDARY_START + BYTE_COUNT] = pair_mass + F.logsigmoid(odds)
    return output


def adjust_boundary_logprobs(logprobs: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    # Keep exact forced and padded sentinel probabilities unchanged.
    valid = (logprobs < 0) & (logprobs > -1000)
    safe = logprobs.float().clamp(min=-80, max=-torch.finfo(torch.float32).eps)
    log_odds = safe - torch.log(-torch.expm1(safe))
    return torch.where(valid, F.logsigmoid(log_odds + shift), logprobs.float())


class RouterProgram:
    """Temporary in-memory Arm P hooks; the model's remote source files remain untouched."""
    def __init__(self, model: torch.nn.Module, table: torch.Tensor | None = None, *,
                 beta_in: float = 0.0, beta_out: float = 0.0, strength: float = 0.0) -> None:
        _parameters(model)
        if not all(math.isfinite(value) for value in (beta_in, beta_out, strength)):
            raise ValueError("router offsets must be finite")
        if table is not None and (table.numel() != 65536 or not torch.isfinite(table).all()):
            raise ValueError("prior table must contain 65536 finite entries")
        self.model, self.beta_in, self.beta_out, self.strength = model, beta_in, beta_out, strength
        self.encoder = model.model.local_encoder
        self.predictor = self.encoder.boundary_predictor_module
        if self.predictor.boundary_predictor_lookahead != 1 or self.predictor.boundary_threshold != "sample:0":
            raise ValueError("router hooks require the released deterministic one-byte-lookahead predictor")
        self.table = None if table is None else table.detach().float().reshape(256, 256).to(next(model.parameters()).device)
        self._encoder_ids = self._forward_ids = self._boundary_state = None
        self._logits_to_keep: int | torch.Tensor = 0
        self._handles: list[Any] = []
        self._active = False

    def _prefill_hook(self, _module: torch.nn.Module, _args: tuple, output: tuple[torch.Tensor, torch.Tensor]):
        if self.beta_in == 0 and (self.table is None or self.strength == 0):
            return output
        logprobs, original_mask = output
        ids = self._encoder_ids
        if ids is None or ids.shape != logprobs.shape:
            raise RuntimeError("cannot align Bolmo prefill byte IDs with boundary logits")
        current = (ids >= BYTE_START) & (ids < BYTE_START + BYTE_COUNT)
        prior = torch.zeros_like(logprobs, dtype=torch.float32)
        if self.table is not None and ids.shape[1] > 1:
            previous = (ids[:, :-1] >= BYTE_START) & (ids[:, :-1] < BYTE_START + BYTE_COUNT)
            pair_valid = previous & current[:, 1:]
            before = (ids[:, :-1] - BYTE_START).clamp(0, 255)
            after = (ids[:, 1:] - BYTE_START).clamp(0, 255)
            prior[:, 1:] = torch.where(pair_valid, self.table[before, after], 0)
        shifted = adjust_boundary_logprobs(logprobs, self.beta_in + self.strength * prior)
        eligible = current & (logprobs < 0) & (logprobs > -1000)
        shifted = torch.where(eligible, shifted, logprobs.float())
        mask = torch.where(eligible, shifted > math.log(0.5), original_mask)
        return shifted, mask

    def _head_hook(self, _module: torch.nn.Module, _args: tuple, logits: torch.Tensor):
        if self.beta_out == 0 and (self.table is None or self.strength == 0):
            return logits
        ids = self._forward_ids
        if ids is None:
            raise RuntimeError("cannot align Bolmo head logits without forward input IDs")
        if ids.shape[0] != logits.shape[0]:
            if self._boundary_state is None:
                raise RuntimeError("cannot align partial decode batch with input IDs")
            ids = self._boundary_state.selective_get(ids, inv=True)
        keep = self._logits_to_keep
        if isinstance(keep, int):
            ids = ids[:, -keep:] if keep else ids
        else:
            ids = ids[:, keep]
        if ids.shape != logits.shape[:-1]:
            raise RuntimeError("cannot align Bolmo decode history with head logits")
        prior = None
        if self.table is not None:
            valid = (ids >= BYTE_START) & (ids < BYTE_START + BYTE_COUNT)
            last = (ids - BYTE_START).clamp(0, 255)
            prior = torch.where(valid.unsqueeze(-1), self.table[last], 0)
        return adjust_joint_logits(logits, prior, beta=self.beta_out, strength=self.strength)

    def __enter__(self) -> RouterProgram:
        if self._active or getattr(self.model, "_blt_router_program_active", False):
            raise RuntimeError("router program is already installed")
        self._active = True
        self.model._blt_router_program_active = True
        self._encoder_forward = self.encoder.forward
        self._model_forward = self.model.forward
        def encoder_forward(*args: Any, **kwargs: Any):
            previous = self._encoder_ids
            self._encoder_ids = kwargs.get("input_ids", args[0] if args else None)
            try:
                return self._encoder_forward(*args, **kwargs)
            finally:
                self._encoder_ids = previous
        def model_forward(*args: Any, **kwargs: Any):
            previous = (self._forward_ids, self._boundary_state, self._logits_to_keep)
            self._forward_ids = kwargs.get("input_ids", args[0] if args else None)
            self._boundary_state = kwargs.get("boundary_state")
            self._logits_to_keep = kwargs.get("logits_to_keep", 0)
            try:
                return self._model_forward(*args, **kwargs)
            finally:
                self._forward_ids, self._boundary_state, self._logits_to_keep = previous
        self.encoder.forward = encoder_forward
        self.model.forward = model_forward
        try:
            self._handles.append(self.predictor.register_forward_hook(self._prefill_hook))
            self._handles.append(self.model.lm_head.register_forward_hook(self._head_hook))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args: object) -> None:
        for handle in self._handles:
            handle.remove()
        self.encoder.forward = self._encoder_forward
        self.model.forward = self._model_forward
        self._handles.clear()
        self._active = False
        del self.model._blt_router_program_active


@torch.inference_mode()
def _prefill_boundaries(model: Any, raw: bytes) -> list[bool]:
    from .trackA import bolmo_ids, free_bolmo_cache
    free_bolmo_cache(model)
    ids = bolmo_ids(raw, next(model.parameters()).device)
    expanded = None
    if model.model.local_encoder.add_expanded_embeddings:
        expanded = torch.tensor([model.model.tokenizer.expand_byte_ids(ids[0].tolist())], device=ids.device)
    try:
        mask = model.model.prefill_boundary_prediction_forward(ids, expanded_input_ids=expanded)
        return mask[0].cpu().tolist()
    finally:
        free_bolmo_cache(model)


@torch.inference_mode()
def _replay_boundaries(model: Any, raw: bytes) -> list[bool]:
    """Teacher-force bytes through stock generate, including its rollback and cache updates."""
    from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
    from .trackA import bolmo_ids, free_bolmo_cache
    if len(raw) < 3:
        raise ValueError("calibration documents need at least three bytes")

    class Replay(LogitsProcessor):
        def __init__(self):
            self.decisions: list[bool] = []
        def __call__(self, _input_ids, scores):
            index = len(self.decisions)
            if index >= len(raw) or scores.shape[0] != 1:
                raise RuntimeError("unexpected Bolmo replay step")
            low = raw[index] + BYTE_START
            high = raw[index] + BOUNDARY_START
            self.decisions.append(bool((scores[0, high] > scores[0, low]).item()))
            allowed = torch.full_like(scores, -torch.inf)
            allowed[:, low], allowed[:, high] = scores[:, low], scores[:, high]
            return allowed

    class Stop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return torch.full((input_ids.shape[0],), input_ids.shape[1] >= len(raw) + 1,
                              device=input_ids.device, dtype=torch.bool)

    replay = Replay()
    ids = bolmo_ids(raw[:1], next(model.parameters()).device)
    predictor = model.model.local_encoder.boundary_predictor_module
    original_predict = predictor.forward
    def one_token_bos(hidden_states, *args, **kwargs):
        # The stock predictor indexes an empty lookahead tensor for a one-token BOS.
        # Its result is immediately overwritten by generate's true BOS mask, so
        # supplying the forced BOS value here does not change the decoder state.
        if hidden_states.shape[1] == 1:
            return (torch.zeros(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.float32),
                    torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool))
        return original_predict(hidden_states, *args, **kwargs)
    predictor.forward = one_token_bos
    try:
        output = model.generate(ids, do_sample=False, max_new_tokens=len(raw) + 4,
                                logits_processor=LogitsProcessorList([replay]),
                                stopping_criteria=StoppingCriteriaList([Stop()]))
        expected = bolmo_ids(raw, ids.device)
        if len(replay.decisions) != len(raw) or not torch.equal(output, expected):
            raise RuntimeError("teacher-forced replay did not reproduce the calibration bytes")
        return replay.decisions
    finally:
        predictor.forward = original_predict
        free_bolmo_cache(model)


def calibrate(model: Any, docs: Any, target_p_in: float, target_p_out: float, *, table: torch.Tensor | None = None,
              strength: float = 0.0, tolerance: float = 0.03) -> dict[str, Any]:
    """Calibrate on supplied documents only; decode offsets are measured by actual replay."""
    raw_docs = [item.encode("utf-8") if isinstance(item, str) else bytes(item) for item in docs]
    if not raw_docs or any(len(item) < 3 for item in raw_docs):
        raise ValueError("provide calibration documents with at least three bytes each")
    if not (target_p_in >= 1 and target_p_out >= 1 and 0 < tolerance < 1):
        raise ValueError("invalid compression targets or tolerance")
    total_bytes = sum(len(item) for item in raw_docs)

    def measure(beta_in: float, beta_out: float, replay: bool):
        ins, outs = [], []
        with RouterProgram(model, table, beta_in=beta_in, beta_out=beta_out, strength=strength):
            for raw in raw_docs:
                # The final byte has no prefill lookahead and is excluded on both paths.
                ins.append(_prefill_boundaries(model, raw)[1:-1])
                if replay:
                    outs.append(_replay_boundaries(model, raw)[:-1])
        in_count = len(raw_docs) + sum(sum(row) for row in ins)
        result = {"p_in": total_bytes / in_count}
        if replay:
            out_count = len(raw_docs) + sum(sum(row) for row in outs)
            compared = sum(len(row) for row in ins)
            disagreements = sum(a != b for left, right in zip(ins, outs) for a, b in zip(left, right))
            result.update({"p_out": total_bytes / out_count, "compared_boundaries": compared,
                           "boundary_disagreement_rate": disagreements / compared if compared else 0.0})
        return result

    left, right = -20.0, 20.0
    candidates = []
    for _ in range(24):
        middle = (left + right) / 2
        value = measure(middle, 0, False)["p_in"]
        candidates.append((abs(value / target_p_in - 1), middle, value))
        if value > target_p_in:
            left = middle
        else:
            right = middle
    _, beta_in, pin = min(candidates)
    if abs(pin / target_p_in - 1) > tolerance:
        raise ValueError(f"prefill target unreachable within tolerance: target={target_p_in}, nearest={pin}")

    trials: dict[float, dict[str, Any]] = {}
    def try_offset(beta: float):
        if beta not in trials:
            trials[beta] = measure(beta_in, beta, True)
        return abs(trials[beta]["p_out"] / target_p_out - 1)
    coarse = [float(i) for i in range(-12, 13, 3)]
    best = min(coarse, key=try_offset)
    fine = [best + i * 0.25 for i in range(-12, 13)]
    best = min(fine + coarse, key=try_offset)
    result = trials[best]
    if abs(result["p_out"] / target_p_out - 1) > tolerance:
        raise ValueError(f"decode target not reached by replay search: target={target_p_out}, nearest={result['p_out']}")
    return {"beta_in": beta_in, "beta_out": best, "target_p_in": target_p_in, "target_p_out": target_p_out,
            "calibration_documents": len(raw_docs), "calibration_bytes": total_bytes,
            "boundary_convention": "one BOS patch per document; compare byte boundaries before the final lookahead sentinel",
            "decode_search_trials": len(trials), **result}


DEFAULT_SENTINELS = ("0123456789", "user_id_123", "{[()]}", "name=value;path/to/file", ":: -> != <=")


def evaluate_edited(bundle: Any, chunks: Any, *, sentinels: Any = DEFAULT_SENTINELS) -> dict[str, Any]:
    from .trackA import score_bolmo_chunk, score_reference
    result = score_reference(bundle, chunks)
    result["sentinels"] = [{"text": text, **score_bolmo_chunk(bundle, text.encode("utf-8"), byte_nll=True)}
                           for text in sentinels]
    return result
