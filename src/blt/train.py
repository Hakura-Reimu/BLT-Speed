from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .atomic import atomic_json, sync_directory
from .config import BPEConfig, ByteConfig, ModelConfig, normalized_model_dict, parameter_counts
from .data import ByteWindow, WindowIndex, iter_eval_chunks, sample_window_index, sha256_file, tokenize_window
from .model import BPELM, ByteLatentLM
from .patcher import PatchBatch, patch_batch
from .score import NUMERICS_POLICY, configure_matmul_precision
from .token_cache import MemoryTokenCache, TokenCache


MILESTONES = (375_000_000, 750_000_000, 1_500_000_000, 3_000_000_000, 3_750_000_000)
# Equal-time view of the 3 GB trajectories: active seconds inside run(), including compilation, validation and
# checkpointing (and recompilation after a resume), excluding paused time and process start-up.
TIME_MILESTONES = (3600,)
ACTIVE_TIME_LEDGER = "active_time.json"
BPE_BUCKET_SIZE = 256
PATCH_BUCKET_SIZE = 128


def bucket_length(length: int, quantum: int, limit: int) -> int:
    """Smallest bucket that fits, capped only by the model's position limit."""
    if length < 0 or quantum <= 0 or limit < 0 or length > limit:
        raise ValueError("length exceeds the configured position limit")
    if length == 0:
        return 0
    return min(((length + quantum - 1) // quantum) * quantum, limit)


def bucketed_patches(values: np.ndarray, lengths: np.ndarray, config: ByteConfig) -> PatchBatch:
    """Scan once, then pad completed patch ends; global BOS occupies one more position."""
    if config.min_len is None:
        raise ValueError("byte patcher must be calibrated before training")
    patches = patch_batch(values, config.min_len, config.max_len, config.block_size, lengths=lengths)
    required = int(patches.counts.max(initial=0))
    capacity = bucket_length(required, PATCH_BUCKET_SIZE, config.trunk.max_positions - 1)
    if capacity == patches.ends.shape[1]:
        return patches
    ends = np.full((values.shape[0], capacity), -1, dtype=np.int64)
    ends[:, :required] = patches.ends
    return replace(patches, ends=ends)


def configure_compile_cache() -> None:
    """Keep static bucket graphs resident instead of falling back after Dynamo's small default limit."""
    import torch._dynamo.config as dynamo_config
    for name, minimum in (("cache_size_limit", 1024), ("accumulated_cache_size_limit", 4096)):
        if hasattr(dynamo_config, name):
            setattr(dynamo_config, name, max(int(getattr(dynamo_config, name)), minimum))
    if hasattr(dynamo_config, "fail_on_recompile_limit_hit"):
        dynamo_config.fail_on_recompile_limit_hit = True


@dataclass(frozen=True)
class TrainingConfig:
    model: ModelConfig
    init_seed: int
    data_seed: int
    windows_per_step: int
    micro_batch_size: int
    window_bytes: int
    peak_lr: float
    schedule_steps: int
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    val_interval_steps: int = 500
    val_windows: int = 16
    compile_model: bool = False

    def __post_init__(self) -> None:
        if min(self.windows_per_step, self.micro_batch_size, self.window_bytes, self.schedule_steps, self.warmup_steps,
               self.val_interval_steps, self.val_windows) <= 0 or self.peak_lr <= 0 or self.grad_clip <= 0:
            raise ValueError("training settings must be positive")
        if self.micro_batch_size > self.windows_per_step:
            raise ValueError("micro batch cannot exceed global batch")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(config: ModelConfig) -> BPELM | ByteLatentLM:
    model = BPELM(config) if isinstance(config, BPEConfig) else ByteLatentLM(config)
    actual = sum(p.numel() for p in model.parameters())
    expected = parameter_counts(config)["total"]
    if actual != expected:
        raise AssertionError(f"model parameter count {actual} differs from config count {expected}")
    return model


def make_optimizer(model: nn.Module, config: TrainingConfig, device: torch.device) -> torch.optim.Optimizer:
    embeddings = {id(module.weight) for module in model.modules() if isinstance(module, nn.Embedding)}
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if parameter.ndim < 2 or id(parameter) in embeddings else decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": config.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=config.peak_lr, betas=(0.9, 0.95), fused=device.type == "cuda")


def make_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < config.warmup_steps:
            return (step + 1) / config.warmup_steps
        progress = min(1.0, (step - config.warmup_steps) / max(1, config.schedule_steps - config.warmup_steps))
        return 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _link_checkpoint(source: Path, target: Path) -> None:
    temporary = target.with_name(target.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
        os.replace(temporary, target)
        sync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _truncate_uncommitted_log(path: Path, committed_step: int) -> None:
    if not path.exists():
        return
    with open(path, "rb") as original, tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            previous = 0
            for line in original:
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    if not line.endswith(b"\n") and not original.read(1):
                        break
                    raise ValueError("training log has a corrupt committed record")
                step = record["step"]
                if step <= committed_step:
                    if step <= previous:
                        raise ValueError("training log contains duplicate or unordered committed steps")
                    handle.write(line if line.endswith(b"\n") else line + b"\n")
                    previous = step
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _batch(windows: list[ByteWindow], config: ModelConfig, tokenizer: Any, device: torch.device,
           *, byte_length: int | None = None) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if not windows:
        raise ValueError("cannot batch an empty collection of windows")
    byte_count = sum(len(window.data) for window in windows)
    if isinstance(config, BPEConfig):
        sequences = [tokenize_window(tokenizer, window.data) for window in windows]
        if any(not ids for ids in sequences):
            raise ValueError("empty token sequence")
        inputs = [[config.bos_id] + ids[:-1] for ids in sequences]
        pad = config.bos_id
        length = bucket_length(max(map(len, sequences)), BPE_BUCKET_SIZE, config.trunk.max_positions)
    else:
        sequences = [list(window.data) for window in windows]
        inputs, pad = sequences, 0
        length = max(map(len, sequences)) if byte_length is None else byte_length
        if length < max(map(len, sequences)):
            raise ValueError("byte window exceeds the configured sequence length")
    x = torch.full((len(windows), length), pad, dtype=torch.long, device=device)
    target = torch.full_like(x, -100)
    for i, (input_ids, ids) in enumerate(zip(inputs, sequences)):
        x[i, :len(input_ids)] = torch.tensor(input_ids, device=device)
        target[i, :len(ids)] = torch.tensor(ids, device=device)
    units = sum(map(len, sequences))
    if isinstance(config, ByteConfig):
        values = np.zeros((len(windows), length), dtype=np.uint8)
        for i, window in enumerate(windows):
            values[i, :len(window.data)] = np.frombuffer(window.data, dtype=np.uint8)
        boundaries = patch_batch(values, config.min_len, config.max_len, config.block_size,
                                 lengths=np.asarray([len(window.data) for window in windows])).boundaries
        units = sum(int(boundaries[i, :len(window.data)].sum()) for i, window in enumerate(windows))
    return x, target, byte_count, units


def _chunk_cross_entropy(hidden: torch.Tensor, weight: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = F.linear(hidden, weight)
    return F.cross_entropy(logits.float(), targets, ignore_index=-100, reduction="sum")


def _loss(model: nn.Module, inputs: torch.Tensor | dict[str, Any], targets: torch.Tensor,
          *, chunk_size: int = 1024) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError("loss chunk size must be positive")
    # Request hidden states through forward so torch.compile wraps the trunk.
    hidden = model(inputs, return_hidden=True)
    if hidden.shape[:-1] != targets.shape:
        raise ValueError("hidden states and targets must have matching batch and sequence dimensions")
    flat_hidden = hidden.reshape(-1, hidden.shape[-1])
    flat_targets = targets.reshape(-1)
    weight = model.output_weight
    total = None
    for start in range(0, flat_hidden.shape[0], chunk_size):
        args = (flat_hidden[start:start + chunk_size], weight, flat_targets[start:start + chunk_size])
        if torch.is_grad_enabled():
            # Head activations are recomputed in backward; the head contains no randomness.
            chunk = checkpoint(_chunk_cross_entropy, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            chunk = _chunk_cross_entropy(*args)
        total = chunk if total is None else total + chunk
    if total is None:
        # Preserve the sum-reduction result and gradients for an empty batch.
        return flat_hidden.sum() * 0 + weight.sum() * 0
    return total


class Trainer:
    def __init__(self, config: TrainingConfig, model: nn.Module, windows: WindowIndex, run_dir: str | Path,
                 device: torch.device | str, manifest_hash: str, tokenizer_hash: str | None = None,
                 tokenizer: Any = None, token_cache: TokenCache | MemoryTokenCache | None = None) -> None:
        self.config, self.model, self.windows = config, model, windows
        self.run_dir, self.device = Path(run_dir), torch.device(device)
        self.manifest_hash, self.tokenizer_hash, self.tokenizer = manifest_hash, tokenizer_hash, tokenizer
        if isinstance(config.model, BPEConfig) and tokenizer is None:
            raise ValueError("BPE training requires a tokenizer")
        self.token_cache = token_cache
        if isinstance(config.model, BPEConfig):
            self.token_cache = token_cache if token_cache is not None else MemoryTokenCache(windows, tokenizer)
            if len(self.token_cache) != len(windows):
                raise ValueError("token cache and window counts differ")
            if isinstance(self.token_cache, TokenCache):
                expected = {"window_index_sha256": windows.index_sha256, "manifest_sha256": manifest_hash,
                            "tokenizer_sha256": tokenizer_hash, "vocab_size": config.model.vocab_size,
                            "window_bytes": config.window_bytes}
                if any(self.token_cache.meta.get(key) != value for key, value in expected.items()):
                    raise ValueError("token cache does not match this training run")
            # Keep this legacy 64-aligned value for existing frozen configs/checkpoints;
            # actual microbatch allocation is selected from its own 256-token bucket.
            self.sequence_length = ((self.token_cache.max_tokens + 63) // 64) * 64
            if self.token_cache.max_tokens > config.model.trunk.max_positions:
                raise ValueError("BPE window exceeds the configured position limit")
        else:
            self.sequence_length = config.window_bytes
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future | None = None
        self._future_key: tuple[int, int] | None = None
        self._compiled_steps = 0
        self._compile_graphs = 0
        self._compile_recompiles = 0
        self._padding_buckets_seen: set[tuple[int, int, int]] = set()
        self._padding_history_complete = True
        self.model.to(self.device)
        if any(p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError("model parameters must remain in fp32")
        configure_matmul_precision()
        if config.compile_model:
            configure_compile_cache()
        self.train_model = torch.compile(model, fullgraph=True, dynamic=False) if config.compile_model else model
        self.optimizer = make_optimizer(model, config, self.device)
        self.scheduler = make_scheduler(self.optimizer, config)
        self.step = 0
        self.consumed_bytes = 0
        self.sampler_origin = 0
        self.checkpoint_serial = 0
        self.saved_milestones: set[int] = set()
        self.active_seconds = 0.0
        self.sessions = 0
        self.saved_time_milestones: set[int] = set()
        self.initialization: dict[str, Any] | None = None
        # Read once: checkpoints record the code this process imported, not whatever is deployed when they are saved.
        self.code_revision = git_revision()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cache_meta_hash = sha256_file(token_cache.path / "meta.json") if isinstance(token_cache, TokenCache) else None
        frozen = {"training": asdict(config), "parameter_counts": parameter_counts(config.model),
                  "data_manifest_sha256": manifest_hash, "tokenizer_sha256": tokenizer_hash,
                  "padded_sequence_length": self.sequence_length,
                  "token_cache_meta_sha256": self.cache_meta_hash}
        config_path = self.run_dir / "config.json"
        if config_path.exists():
            if json.loads(config_path.read_text(encoding="utf-8")) != frozen:
                raise ValueError("run configuration or data/tokenizer hash differs from the frozen configuration")
        else:
            atomic_json(config_path, frozen)
        self.padding_policy = {"version": 1, "bpe_bucket_size": BPE_BUCKET_SIZE,
                               "patch_bucket_size": PATCH_BUCKET_SIZE, "global_bos_positions": 1,
                               "position_limit": config.model.trunk.max_positions,
                               "last_bucket": "capped at the model position limit"}
        policy_path = self.run_dir / "padding_policy.json"
        if policy_path.exists():
            if json.loads(policy_path.read_text(encoding="utf-8")) != self.padding_policy:
                raise ValueError("run padding policy differs from the frozen configuration")
        else:
            atomic_json(policy_path, self.padding_policy)
        numerics_path = self.run_dir / "numerics_policy.json"
        if numerics_path.exists():
            if json.loads(numerics_path.read_text(encoding="utf-8")) != NUMERICS_POLICY:
                raise ValueError("run numerics policy differs from the frozen configuration")
        else:
            atomic_json(numerics_path, NUMERICS_POLICY)

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()

    def sample_indices(self, step: int) -> list[int]:
        cfg = self.config
        return [sample_window_index(cfg.data_seed, self.sampler_origin + step * cfg.windows_per_step + i, 0, 1, len(self.windows))
                for i in range(cfg.windows_per_step)]

    def sample_audit(self, step: int) -> dict[str, Any]:
        indexes = self.sample_indices(step) if step >= 0 else []
        return {"sample_indices": indexes,
                "window_sha256": [hashlib.sha256(self.windows[index].data).hexdigest() for index in indexes]}

    def _prepare_microbatch(self, indexes: tuple[int, ...]) -> tuple[dict[str, Any], torch.Tensor, int, int]:
        cfg = self.config
        batch_size = cfg.micro_batch_size
        if not indexes or len(indexes) > batch_size:
            raise ValueError("invalid microbatch indexes")
        if isinstance(cfg.model, BPEConfig):
            assert self.token_cache is not None
            sequences = [np.asarray(self.token_cache[index], dtype=np.int64) for index in indexes]
            for ids in sequences:
                if len(ids) == 0 or ids.min() < 0 or ids.max() >= cfg.model.vocab_size:
                    raise ValueError("invalid cached token sequence")
            length = bucket_length(max(map(len, sequences)), BPE_BUCKET_SIZE, cfg.model.trunk.max_positions)
            targets = np.full((batch_size, length), -100, dtype=np.int64)
            inputs = np.full((batch_size, length), cfg.model.bos_id, dtype=np.int64)
            units = 0
            for row, ids in enumerate(sequences):
                inputs[row, 1:len(ids)] = ids[:-1]
                targets[row, :len(ids)] = ids
                units += len(ids)
            batch: dict[str, Any] = {"tokens": torch.from_numpy(inputs)}
            units_with_tail = units
        else:
            c = cfg.model
            length = cfg.window_bytes
            targets = np.full((batch_size, length), -100, dtype=np.int64)
            values = np.zeros((batch_size, length), dtype=np.uint8)
            lengths = np.zeros(batch_size, dtype=np.int64)
            for row, index in enumerate(indexes):
                raw = self.windows[index].data
                if len(raw) > length:
                    raise ValueError("byte window exceeds fixed training shape")
                values[row, :len(raw)] = np.frombuffer(raw, dtype=np.uint8)
                targets[row, :len(raw)] = values[row, :len(raw)]
                lengths[row] = len(raw)
            assert c.min_len is not None
            patches = bucketed_patches(values, lengths, c)
            units = int(patches.counts.sum())
            ends_at_tail = patches.boundaries[np.arange(batch_size), np.maximum(lengths - 1, 0)]
            units_with_tail = units + int(((lengths > 0) & ~ends_at_tail).sum())
            batch = {"bytes": torch.from_numpy(values),
                     "patches": {"boundaries": torch.from_numpy(patches.boundaries),
                                 "ends": torch.from_numpy(patches.ends), "patch_ids": torch.from_numpy(patches.patch_ids),
                                 "counts": torch.from_numpy(patches.counts)}}
        target = torch.from_numpy(targets)
        if self.device.type == "cuda":
            def pin(value: Any) -> Any:
                return {key: pin(item) for key, item in value.items()} if isinstance(value, dict) else value.pin_memory()
            batch, target = pin(batch), target.pin_memory()
        return batch, target, units, units_with_tail

    def start_prefetch(self) -> None:
        if self._executor is not None:
            raise RuntimeError("prefetch is already running")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="blt-data")

    def stop_prefetch(self) -> None:
        if self._executor is not None:
            if self._future is not None:
                self._future.cancel()
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
            self._future = None
            self._future_key = None

    def _schedule(self, step: int, offset: int) -> None:
        assert self._executor is not None
        indices = self.sample_indices(step)[offset:offset + self.config.micro_batch_size]
        self._future = self._executor.submit(self._prepare_microbatch, tuple(indices))
        self._future_key = (step, offset)

    def _next_microbatch(self, step: int, offset: int) -> tuple[dict[str, Any], torch.Tensor, int, int]:
        if self._executor is None:
            indices = self.sample_indices(step)[offset:offset + self.config.micro_batch_size]
            return self._prepare_microbatch(tuple(indices))
        if self._future_key != (step, offset):
            if self._future is not None:
                self._future.cancel()
            self._schedule(step, offset)
        assert self._future is not None
        result = self._future.result()
        next_offset = offset + self.config.micro_batch_size
        self._schedule(step if next_offset < self.config.windows_per_step else step + 1,
                       next_offset if next_offset < self.config.windows_per_step else 0)
        return result

    def _to_device(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self._to_device(item) for key, item in value.items()}
        return value.to(self.device, non_blocking=self.device.type == "cuda")

    def train_step(self) -> dict[str, Any]:
        cfg = self.config
        indexes = self.sample_indices(self.step)
        total_bytes = sum(self.windows.windows[index][1] - self.windows.windows[index][0] for index in indexes)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_nll, units, units_with_tail, data_wait = None, 0, 0, 0.0
        padded_positions = 0
        bucket_shapes: list[tuple[int, int, int]] = []
        seen_before = len(self._padding_buckets_seen)
        lr = self.optimizer.param_groups[0]["lr"]
        if cfg.compile_model:
            from torch._dynamo.utils import counters
            graphs_before = counters["stats"].get("unique_graphs", 0)
        for start in range(0, cfg.windows_per_step, cfg.micro_batch_size):
            wait_start = time.monotonic()
            batch, targets, count, count_with_tail = self._next_microbatch(self.step, start)
            data_wait += time.monotonic() - wait_start
            if isinstance(cfg.model, BPEConfig):
                batch_rows, capacity = batch["tokens"].shape
                bucket_shape = (batch_rows, 0, capacity)
            else:
                batch_rows, byte_length = batch["bytes"].shape
                capacity = batch["patches"]["ends"].shape[1]
                bucket_shape = (batch_rows, byte_length, capacity)
            bucket_shapes.append(bucket_shape)
            self._padding_buckets_seen.add(bucket_shape)
            padded_positions += batch_rows * capacity
            inputs, targets = self._to_device(batch), self._to_device(targets)
            with self._autocast():
                nll = _loss(self.train_model, inputs, targets)
            (nll / total_bytes).backward()
            total_nll = nll.detach() if total_nll is None else total_nll + nll.detach()
            units += count
            units_with_tail += count_with_tail
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        # One device-to-host synchronization per optimizer update, after all microbatches.
        assert total_nll is not None
        loss_value, norm_value = torch.stack((total_nll.float() / total_bytes, norm.float())).cpu().tolist()
        self.step += 1
        self.consumed_bytes += total_bytes
        metrics = {"step": self.step, "consumed_bytes": self.consumed_bytes, "loss_nats_per_byte": loss_value,
                "lr": lr, "grad_norm": norm_value, "units_per_window": units / len(indexes),
                "units_with_tail_per_window": units_with_tail / len(indexes),
                "unit": "tokens" if isinstance(cfg.model, BPEConfig) else "completed_patches",
                "sample_indices": indexes, "data_wait_seconds": data_wait,
                "padding_real_positions": units, "padding_padded_positions": padded_positions,
                "padding_efficiency": units / padded_positions if padded_positions else 1.0,
                "padding_bucket_shapes": [list(shape) for shape in bucket_shapes],
                "padding_new_buckets": len(self._padding_buckets_seen) - seen_before,
                "padding_distinct_buckets": len(self._padding_buckets_seen),
                "padding_history_complete": self._padding_history_complete,
                "compile_new_graphs": 0, "compile_recompiles": 0, "compile_recompiled": False,
                "compile_new_graphs_since_start": 0, "compile_recompiles_since_start": 0}
        if cfg.compile_model:
            new_graphs = counters["stats"].get("unique_graphs", 0) - graphs_before
            new_recompiles = max(0, new_graphs - int(self._compiled_steps == 0 and new_graphs > 0))
            self._compile_graphs += new_graphs
            self._compile_recompiles += new_recompiles
            metrics.update(compile_new_graphs=new_graphs, compile_recompiles=new_recompiles,
                           compile_recompiled=new_recompiles > 0,
                           compile_new_graphs_since_start=self._compile_graphs,
                           compile_recompiles_since_start=self._compile_recompiles)
            self._compiled_steps += 1
        return metrics

    @torch.no_grad()
    def validate(self, supply: str | Path, stop: threading.Event | None = None) -> float | None:
        self.model.eval()
        count, nll, num_bytes = 0, None, 0
        pending: list[ByteWindow] = []
        def evaluate(group: list[ByteWindow]) -> tuple[torch.Tensor, int]:
            if isinstance(self.config.model, ByteConfig):
                c = self.config.model
                x, y, size, _ = _batch(group, c, None, torch.device("cpu"), byte_length=self.config.window_bytes)
                patches = bucketed_patches(x.numpy(), np.asarray([len(window.data) for window in group]), c)
                inputs = {"bytes": x, "patches": {"ends": torch.from_numpy(patches.ends),
                          "patch_ids": torch.from_numpy(patches.patch_ids), "counts": torch.from_numpy(patches.counts),
                          "boundaries": torch.from_numpy(patches.boundaries)}}
                inputs, y = self._to_device(inputs), self._to_device(y)
            else:
                inputs, y, size, _ = _batch(group, self.config.model, self.tokenizer, self.device)
            with self._autocast():
                return _loss(self.model, inputs, y).detach(), size
        for window in iter_eval_chunks(supply, "val", self.config.window_bytes):
            pending.append(window)
            count += 1
            if len(pending) == self.config.micro_batch_size or count == self.config.val_windows:
                value, size = evaluate(pending)
                nll = value if nll is None else nll + value
                num_bytes += size
                pending.clear()
                if stop is not None and stop.is_set():
                    return None
            if count == self.config.val_windows:
                break
        if pending:
            value, size = evaluate(pending)
            nll = value if nll is None else nll + value
            num_bytes += size
        if not num_bytes or nll is None:
            raise ValueError("validation split is empty")
        return (nll / (math.log(2) * num_bytes)).item()

    def save(self, milestones: tuple[int, ...] = (), time_milestones: tuple[int, ...] = ()) -> Path:
        self.checkpoint_serial += 1
        self.saved_milestones.update(milestones)
        self.saved_time_milestones.update(time_milestones)
        audit = self.sample_audit(self.step - 1)
        payload = {"version": 2, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                   "scheduler": self.scheduler.state_dict(), "step": self.step, "consumed_bytes": self.consumed_bytes,
                   "sampler_cursor": self.sampler_origin + self.step * self.config.windows_per_step,
                   "checkpoint_serial": self.checkpoint_serial, "saved_milestones": sorted(self.saved_milestones),
                   "rng_python": random.getstate(), "rng_numpy": np.random.get_state(), "rng_torch": torch.get_rng_state(),
                   "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                   "config": asdict(self.config), "git_revision": self.code_revision, "data_manifest_hash": self.manifest_hash,
                   "tokenizer_hash": self.tokenizer_hash, "sample_audit": audit,
                   "token_cache_meta_sha256": self.cache_meta_hash, "padded_sequence_length": self.sequence_length}
        payload.update(self._checkpoint_extra())
        directory = self.run_dir / "ckpt"
        path = directory / f"rolling-{self.checkpoint_serial % 2}.pt"
        _atomic_torch_save(path, payload)
        for milestone in milestones:
            _link_checkpoint(path, directory / f"bytes-{milestone}.pt")
        for milestone in time_milestones:
            _link_checkpoint(path, directory / f"active-{milestone}s.pt")
        atomic_json(directory / "latest.json", {"checkpoint": path.name, "step": self.step,
                                                "consumed_bytes": self.consumed_bytes, "serial": self.checkpoint_serial,
                                                "sample_audit": audit})
        return path

    def _checkpoint_extra(self) -> dict[str, Any]:
        return {"sampler_origin": self.sampler_origin, "initialization": self.initialization,
                "padding_bucket_shapes_seen": [list(shape) for shape in sorted(self._padding_buckets_seen)],
                "padding_history_complete": self._padding_history_complete, "padding_policy": self.padding_policy,
                "active_seconds": self.active_seconds, "sessions": self.sessions,
                "saved_time_milestones": sorted(self.saved_time_milestones)}

    def _load_checkpoint_extra(self, checkpoint: dict[str, Any]) -> None:
        if checkpoint.get("padding_policy", self.padding_policy) != self.padding_policy:
            raise ValueError("checkpoint padding policy differs")
        # Checkpoints written before the equal-time milestone have no active-time record. The ledger, written after
        # each checkpoint and at exit, also holds the final write and shutdown of the previous session.
        self.active_seconds = float(checkpoint.get("active_seconds", 0.0))
        self.sessions = int(checkpoint.get("sessions", 0))
        ledger = self.run_dir / ACTIVE_TIME_LEDGER
        if ledger.exists():
            recorded = json.loads(ledger.read_text(encoding="utf-8"))
            self.active_seconds = max(self.active_seconds, float(recorded["active_seconds"]))
            self.sessions = max(self.sessions, int(recorded["sessions"]))
        self.saved_time_milestones = set(checkpoint.get("saved_time_milestones", []))
        shapes = checkpoint.get("padding_bucket_shapes_seen")
        if shapes is None:
            self._padding_buckets_seen = set()
            self._padding_history_complete = checkpoint["step"] == 0
        else:
            if (not isinstance(shapes, list) or any(not isinstance(shape, list) or len(shape) != 3 or
                    any(not isinstance(value, int) or value < 0 for value in shape) or shape[0] == 0 for shape in shapes)):
                raise ValueError("invalid checkpoint padding bucket history")
            complete = checkpoint.get("padding_history_complete")
            if not isinstance(complete, bool):
                raise ValueError("invalid checkpoint padding history status")
            self._padding_buckets_seen = {tuple(shape) for shape in shapes}
            self._padding_history_complete = complete
        self.initialization = checkpoint.get("initialization")
        path = self.run_dir / "initialization.json"
        if self.initialization is None:
            if path.exists():
                raise ValueError("run initialization metadata differs from its checkpoint")
        elif path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != self.initialization:
                raise ValueError("run initialization metadata differs from its checkpoint")
        else:
            atomic_json(path, self.initialization)

    def initialize_from(self, path: str | Path) -> None:
        """Start a new run from weights, Adam moments and schedule of a compatible checkpoint.

        The new run counts its own continuation bytes and steps from zero. The source
        optimizer and scheduler continue from their saved state, including RNG states.
        """
        source = Path(path).resolve()
        if source.is_relative_to(self.run_dir.resolve()):
            raise ValueError("--init-from must refer to a different run")
        if self.step or self.checkpoint_serial or (self.run_dir / "ckpt/latest.json").exists():
            raise ValueError("initialization is only permitted before a new run's first checkpoint")
        checkpoint = torch.load(source, map_location=self.device, weights_only=False)
        if checkpoint.get("version") != 2:
            raise ValueError("unsupported initialization checkpoint")
        old = {**checkpoint["config"], "model": normalized_model_dict(checkpoint["config"]["model"])}
        new = asdict(self.config)
        same_fields = ("model", "init_seed", "data_seed", "peak_lr", "schedule_steps", "warmup_steps",
                       "weight_decay", "grad_clip")
        if (any(old.get(key) != new[key] for key in same_fields)
                or old["window_bytes"] * old["windows_per_step"] != self.config.window_bytes * self.config.windows_per_step
                or checkpoint["data_manifest_hash"] != self.manifest_hash
                or checkpoint["tokenizer_hash"] != self.tokenizer_hash):
            raise ValueError("initialization checkpoint has incompatible model, data, optimizer or schedule")
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.sampler_origin = checkpoint["sampler_cursor"]
        if not isinstance(self.sampler_origin, int) or self.sampler_origin < 0:
            raise ValueError("invalid source sampler cursor")
        random.setstate(checkpoint["rng_python"])
        np.random.set_state(checkpoint["rng_numpy"])
        torch.set_rng_state(checkpoint["rng_torch"].cpu())
        if checkpoint["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["rng_cuda"]])
        prior = checkpoint.get("initialization") or {}
        prior_actual = prior.get("source_total_consumed_bytes", 0)
        prior_nominal = prior.get("source_nominal_total_bytes", 0)
        filename = source.name
        milestone = filename[6:-3] if filename.startswith("bytes-") and filename.endswith(".pt") else ""
        nominal = prior_nominal + int(milestone) if milestone.isdigit() and isinstance(prior_nominal, int) else None
        self.initialization = {"checkpoint": str(source), "checkpoint_sha256": sha256_file(source),
                               "source_step": checkpoint["step"], "source_consumed_bytes": checkpoint["consumed_bytes"],
                               "source_total_consumed_bytes": prior_actual + checkpoint["consumed_bytes"],
                               "source_nominal_total_bytes": nominal}
        metadata = self.run_dir / "initialization.json"
        if metadata.exists():
            if json.loads(metadata.read_text(encoding="utf-8")) != self.initialization:
                raise ValueError("new run already has different initialization metadata")
        else:
            atomic_json(metadata, self.initialization)

    def load(self, path: str | Path | None = None) -> None:
        if path is None:
            directory = self.run_dir / "ckpt"
            pointer = json.loads((directory / "latest.json").read_text(encoding="utf-8"))
            path = directory / pointer["checkpoint"]
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("version") != 2:
            raise ValueError("checkpoint predates the hash and data-pipeline changes")
        saved = {**checkpoint["config"], "model": normalized_model_dict(checkpoint["config"]["model"])}
        if saved != asdict(self.config) or checkpoint["data_manifest_hash"] != self.manifest_hash or checkpoint["tokenizer_hash"] != self.tokenizer_hash:
            raise ValueError("checkpoint configuration or dataset/tokenizer identity mismatch")
        if checkpoint["token_cache_meta_sha256"] != self.cache_meta_hash or checkpoint["padded_sequence_length"] != self.sequence_length:
            raise ValueError("checkpoint token cache or padded shape differs")
        self.sampler_origin = checkpoint.get("sampler_origin", 0)
        if (not isinstance(self.sampler_origin, int) or self.sampler_origin < 0
                or checkpoint["sampler_cursor"] != self.sampler_origin + checkpoint["step"] * self.config.windows_per_step):
            raise ValueError("invalid committed sampler cursor")
        if checkpoint["sample_audit"] != self.sample_audit(checkpoint["step"] - 1):
            raise ValueError("checkpoint sample audit differs from the current data")
        _truncate_uncommitted_log(self.run_dir / "log.jsonl", checkpoint["step"])
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.step, self.consumed_bytes = checkpoint["step"], checkpoint["consumed_bytes"]
        self.checkpoint_serial = checkpoint["checkpoint_serial"]
        self.saved_milestones = set(checkpoint["saved_milestones"])
        random.setstate(checkpoint["rng_python"])
        np.random.set_state(checkpoint["rng_numpy"])
        torch.set_rng_state(checkpoint["rng_torch"].cpu())
        if checkpoint["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["rng_cuda"]])
        self._load_checkpoint_extra(checkpoint)

    def run(self, supply: str | Path, stop_bytes: int, pause_file: str | Path, *, checkpoint_seconds: float = 900.0,
            poll_seconds: float = 2.0, stop_event: threading.Event | None = None) -> None:
        if stop_bytes <= 0 or not 0 < poll_seconds <= 10 or checkpoint_seconds <= 0:
            raise ValueError("invalid stop or polling settings")
        stop = stop_event if stop_event is not None else threading.Event()
        done = threading.Event()
        pause = Path(pause_file)
        previous_handler = signal.getsignal(signal.SIGTERM)
        def terminate(_signum: int, _frame: Any) -> None:
            stop.set()
        signal.signal(signal.SIGTERM, terminate)
        def watch() -> None:
            while not done.is_set():
                if pause.exists():
                    stop.set()
                done.wait(poll_seconds)
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        last_save = time.monotonic()
        last_log = last_save
        session_start, active_before = last_save, self.active_seconds
        self.sessions += 1
        def record_active_time() -> None:
            # The ledger also covers checkpoint writes and shutdown, which a checkpoint cannot contain.
            self.active_seconds = active_before + (time.monotonic() - session_start)
            atomic_json(self.run_dir / ACTIVE_TIME_LEDGER, {"active_seconds": self.active_seconds,
                                                            "sessions": self.sessions, "step": self.step})
        previous_bytes = self.consumed_bytes
        dirty = False
        try:
            # The queue can safely resume even if a job is interrupted before its first update.
            if self.checkpoint_serial == 0:
                if (self.run_dir / "ckpt/latest.json").exists():
                    raise ValueError("existing run requires loading its checkpoint before training")
                self.save()
            if pause.exists():
                stop.set()
            self.start_prefetch()
            while self.consumed_bytes < stop_bytes and not stop.is_set():
                metrics = self.train_step()
                dirty = True
                now = time.monotonic()
                metrics["bytes_per_second"] = (self.consumed_bytes - previous_bytes) / max(now - last_log, 1e-9)
                metrics["peak_memory_bytes"] = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
                if self.step % self.config.val_interval_steps == 0 and not stop.is_set():
                    metrics["val_bpb"] = self.validate(supply, stop)
                self.active_seconds = active_before + (time.monotonic() - session_start)
                metrics["active_seconds"] = self.active_seconds
                with open(self.run_dir / "log.jsonl", "a", encoding="utf-8", newline="\n") as log:
                    log.write(json.dumps(metrics, sort_keys=True) + "\n")
                    log.flush()
                    os.fsync(log.fileno())
                previous_bytes, last_log = self.consumed_bytes, now
                milestones = tuple(value for value in MILESTONES if value <= self.consumed_bytes and value not in self.saved_milestones)
                timed = tuple(value for value in TIME_MILESTONES
                              if value <= self.active_seconds and value not in self.saved_time_milestones)
                if milestones or timed or now - last_save >= checkpoint_seconds:
                    self.save(milestones, timed)
                    for target in timed:
                        atomic_json(self.run_dir / "ckpt" / f"active-{target}s.json",
                                    {"target_seconds": target, "active_seconds": self.active_seconds,
                                     "overshoot_seconds": self.active_seconds - target, "step": self.step,
                                     "consumed_bytes": self.consumed_bytes, "sessions": self.sessions})
                    record_active_time()
                    dirty = False
                    last_save = time.monotonic()
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
            self.stop_prefetch()
            done.set()
            watcher.join()
            signal.signal(signal.SIGTERM, previous_handler)
            record_active_time()


class LongContextTrainer(Trainer):
    """Continuation with one fixed-shape length bucket per optimizer step.

    Each step has the same nominal raw-byte batch. The four-step cycle therefore
    assigns 50%, 25%, 25% of nominal bytes to the three lengths. UTF-8 boundary
    trimming is recorded in the actual per-bucket byte counters.
    """

    def __init__(self, config: TrainingConfig, model: nn.Module, windows: dict[int, WindowIndex],
                 run_dir: str | Path, device: torch.device | str, manifest_hash: str,
                 tokenizer_hash: str | None = None, tokenizer: Any = None,
                 token_caches: dict[int, TokenCache | MemoryTokenCache] | None = None) -> None:
        sizes = sorted(windows)
        if len(sizes) != 3 or sizes != [config.window_bytes, config.window_bytes * 2, config.window_bytes * 4]:
            raise ValueError("long-context buckets must be the base length, twice and four times that length")
        batch_bytes = config.window_bytes * config.windows_per_step
        if any(batch_bytes % size or windows[size].window_bytes != size for size in sizes):
            raise ValueError("each long-context bucket must divide the nominal global byte batch")
        self.bucket_windows = windows
        self.bucket_caches = token_caches or {}
        self.bucket_sizes = sizes
        self.bucket_cycle = (sizes[0], sizes[1], sizes[0], sizes[2])
        self.bucket_batch_bytes = batch_bytes
        self.bucket_bytes = {size: 0 for size in sizes}
        self.base_config = config
        super().__init__(config, model, windows[sizes[0]], run_dir, device, manifest_hash, tokenizer_hash,
                         tokenizer, self.bucket_caches.get(sizes[0]))
        self.bucket_sequence_lengths: dict[int, int] = {}
        bucket_records = []
        for size in sizes:
            window = windows[size]
            if isinstance(config.model, BPEConfig):
                if size not in self.bucket_caches:
                    raise ValueError(f"missing pretokenized long-context cache for {size} bytes")
                cache = self.bucket_caches[size]
                if len(cache) != len(window):
                    raise ValueError("long-context token cache and window counts differ")
                if isinstance(cache, TokenCache):
                    expected = {"window_index_sha256": window.index_sha256, "manifest_sha256": manifest_hash,
                                "tokenizer_sha256": tokenizer_hash, "vocab_size": config.model.vocab_size,
                                "window_bytes": size}
                    if any(cache.meta.get(key) != value for key, value in expected.items()):
                        raise ValueError("long-context token cache identity differs")
                    cache_hash = sha256_file(cache.path / "meta.json")
                else:
                    cache_hash = None
                # Long-context checkpoint metadata keeps its historical length.
                length = ((cache.max_tokens + 63) // 64) * 64
                if cache.max_tokens > config.model.trunk.max_positions:
                    raise ValueError("long-context token window exceeds the position limit")
            else:
                length, cache_hash = size, None
            self.bucket_sequence_lengths[size] = length
            bucket_records.append({"window_bytes": size, "window_count": len(window),
                                   "window_index_sha256": window.index_sha256,
                                   "token_cache_meta_sha256": cache_hash, "padded_sequence_length": length,
                                   "windows_per_step": batch_bytes // size,
                                   "micro_batch_size": 1 if size == sizes[-1] else min(config.micro_batch_size, batch_bytes // size)})
        frozen = {"version": 1, "sampling": "nominal-byte-cycle-v1", "cycle": list(self.bucket_cycle),
                  "global_batch_bytes": batch_bytes, "buckets": bucket_records}
        path = self.run_dir / "long_context.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != frozen:
                raise ValueError("long-context bucket layout or cache identity differs from the frozen run")
        else:
            atomic_json(path, frozen)
        self.long_context_config = frozen

    def _bucket_at(self, step: int) -> int:
        if step < 0:
            raise ValueError("invalid long-context step")
        return self.bucket_cycle[step % len(self.bucket_cycle)]

    def _bucket_step(self, step: int, size: int) -> int:
        cycles, offset = divmod(step, len(self.bucket_cycle))
        return cycles * self.bucket_cycle.count(size) + self.bucket_cycle[:offset].count(size)

    def sample_indices(self, step: int) -> list[int]:
        size = self._bucket_at(step)
        per_step = self.bucket_batch_bytes // size
        local_step = self._bucket_step(step, size)
        origin = self.sampler_origin * self.base_config.window_bytes // size
        return [sample_window_index(self.base_config.data_seed, origin + local_step * per_step + i, 0, 1, len(self.bucket_windows[size]))
                for i in range(per_step)]

    def sample_audit(self, step: int) -> dict[str, Any]:
        if step < 0:
            return {"bucket_bytes": None, "sample_indices": [], "window_sha256": []}
        size = self._bucket_at(step)
        indexes = self.sample_indices(step)
        return {"bucket_bytes": size, "sample_indices": indexes,
                "window_sha256": [hashlib.sha256(self.bucket_windows[size][index].data).hexdigest() for index in indexes]}

    @contextmanager
    def _using_bucket(self, size: int):
        saved = self.config, self.windows, self.token_cache, self.sequence_length
        count = self.bucket_batch_bytes // size
        micro = 1 if size == self.bucket_sizes[-1] else min(self.base_config.micro_batch_size, count)
        self.config = replace(self.base_config, window_bytes=size, windows_per_step=count, micro_batch_size=micro)
        self.windows = self.bucket_windows[size]
        self.token_cache = self.bucket_caches.get(size)
        self.sequence_length = self.bucket_sequence_lengths[size]
        try:
            yield
        finally:
            self.config, self.windows, self.token_cache, self.sequence_length = saved

    def start_prefetch(self) -> None:
        # Cross-bucket read-ahead would otherwise capture mutable bucket state.
        # Synchronous preparation preserves the committed sampler and correct window bucket.
        return None

    def stop_prefetch(self) -> None:
        return None

    def train_step(self) -> dict[str, Any]:
        size = self._bucket_at(self.step)
        before = self.consumed_bytes
        with self._using_bucket(size):
            metrics = super().train_step()
        self.bucket_bytes[size] += self.consumed_bytes - before
        metrics.update(bucket_bytes=size, bucket_consumed_bytes=self.bucket_bytes[size],
                       bucket_bytes_by_length={str(key): value for key, value in self.bucket_bytes.items()})
        return metrics

    def _checkpoint_extra(self) -> dict[str, Any]:
        return {**super()._checkpoint_extra(), "long_context": {"config": self.long_context_config,
                "bucket_consumed_bytes": {str(key): value for key, value in self.bucket_bytes.items()},
                "next_bucket_steps": {str(size): self._bucket_step(self.step, size) for size in self.bucket_sizes}}}

    def _load_checkpoint_extra(self, checkpoint: dict[str, Any]) -> None:
        super()._load_checkpoint_extra(checkpoint)
        extra = checkpoint.get("long_context")
        if not isinstance(extra, dict) or extra.get("config") != self.long_context_config:
            raise ValueError("checkpoint long-context configuration differs")
        expected_steps = {str(size): self._bucket_step(self.step, size) for size in self.bucket_sizes}
        counts = extra.get("bucket_consumed_bytes", {})
        if (extra.get("next_bucket_steps") != expected_steps or set(counts) != set(expected_steps)
                or any(not isinstance(value, int) or value < 0 for value in counts.values())
                or sum(counts.values()) != self.consumed_bytes):
            raise ValueError("invalid long-context committed sampler or byte counters")
        self.bucket_bytes = {size: counts[str(size)] for size in self.bucket_sizes}
