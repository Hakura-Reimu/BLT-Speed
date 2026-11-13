from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from blt.thin_bolmo import (DistillConfig, DistillTrainer, build_student, configure_trainable, dense_local_cost,
                            distillation_loss, evaluate_pair, full_selection, importance_stats, load_student_checkpoint)
from blt.trackA import _cuda_home, bolmo_ids, load_reference


class ToyMultiHeadNorm(nn.Module):
    def __init__(self, heads, head_dim):
        super().__init__()
        self.num_heads, self.head_dim = heads, head_dim
        self.num_features = heads * head_dim
        self.weight = nn.Parameter(torch.ones(self.num_features))
        self.bias = None

    def forward(self, x):
        assert x.shape[-2:] == (self.num_heads, self.head_dim)
        return F.layer_norm(x, (self.head_dim,)).flatten(-2) * self.weight


class ToyXLSTM(nn.Module):
    """Small differentiable stand-in with the pinned xLSTM tensor layout."""
    def __init__(self, width, heads):
        super().__init__()
        self.config = SimpleNamespace(embedding_dim=width, num_heads=heads, qk_dim_factor=0.5,
                                      v_dim_factor=1.0, weight_mode="single")
        self.qk_dim, self.v_dim = width // 2, width
        self.q = nn.Linear(width, self.qk_dim, bias=False)
        self.k = nn.Linear(width, self.qk_dim, bias=False)
        self.v = nn.Linear(width, self.v_dim, bias=False)
        self.ogate_preact = nn.Linear(width, self.v_dim, bias=False)
        self.igate_preact = nn.Linear(width, heads)
        self.fgate_preact = nn.Linear(width, heads)
        self.multihead_norm = ToyMultiHeadNorm(heads, self.v_dim // heads)
        self.out_proj = nn.Linear(self.v_dim, width, bias=False)

    def forward(self, x):
        b, s, _ = x.shape
        heads = self.config.num_heads
        q = self.q(x).reshape(b, s, heads, -1)
        k = self.k(x).reshape(b, s, heads, -1)
        v = self.v(x).reshape(b, s, heads, -1)
        gates = torch.sigmoid(self.igate_preact(x)) * torch.sigmoid(self.fgate_preact(x))
        memory = torch.cumsum((q * k).sum(-1, keepdim=True) * v * gates[..., None], dim=1)
        return self.out_proj(self.multihead_norm(memory) * torch.sigmoid(self.ogate_preact(x)))


class ToyMLP(nn.Module):
    def __init__(self, width, hidden):
        super().__init__()
        self.hidden_size, self.intermediate_size = width, hidden
        self.config = SimpleNamespace(intermediate_size=hidden)
        self.gate_proj = nn.Linear(width, hidden, bias=False)
        self.up_proj = nn.Linear(width, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ToyLayer(nn.Module):
    def __init__(self, width, heads, hidden):
        super().__init__()
        self.xlstm, self.mlp = ToyXLSTM(width, heads), ToyMLP(width, hidden)
        self.pre_xlstm_layernorm = nn.LayerNorm(width, bias=False)
        self.pre_feedforward_layernorm = nn.LayerNorm(width, bias=False)

    def forward(self, x):
        x = x + self.xlstm(self.pre_xlstm_layernorm(x))
        return x + self.mlp(self.pre_feedforward_layernorm(x))


class ToyPredictor(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.q_proj_layer = nn.Linear(width, width, bias=False)
        self.k_proj_layer = nn.Linear(width, width, bias=False)
        self.boundary_predictor_lookahead = 1
        self.boundary_threshold = "sample:0"

    def forward(self, h):
        cosine = (F.normalize(self.q_proj_layer(h[:, :-1]), dim=-1)
                  * F.normalize(self.k_proj_layer(h[:, 1:]), dim=-1)).sum(-1)
        logp = torch.log1p(-cosine.float().clamp(max=0.999)) - math.log(2)
        logp = torch.cat((torch.zeros_like(logp[:, :1]), logp[:, 1:],
                          torch.full_like(logp[:, :1], -100000.0)), dim=1)
        return logp, logp > math.log(0.5)


class ToyEncoder(nn.Module):
    def __init__(self, width, heads, hidden, suffix_size):
        super().__init__()
        self.add_expanded_embeddings = True
        self.byte_embedding = nn.Embedding(520, width)
        self.subword_embedding = nn.Embedding(suffix_size, width)
        self.layers = nn.ModuleList([ToyLayer(width, heads, hidden)])
        self.post_last_block_norm = nn.LayerNorm(width, bias=False)
        self.out_projection = nn.Linear(width, width)
        self.boundary_predictor_module = ToyPredictor(width)
        self.has_cache = False

    def free_inference_cache(self):
        self.has_cache = False

    def forward(self, input_ids, true_boundary_mask=None, **kwargs):
        h = self.byte_embedding(input_ids) + self.subword_embedding(input_ids % self.subword_embedding.num_embeddings)
        for layer in self.layers:
            h = layer(h)
        h = self.post_last_block_norm(h)
        logp, predicted = self.boundary_predictor_module(h)
        mask = predicted if true_boundary_mask is None else true_boundary_mask
        assert h.shape[0] == 1 and mask[:, 0].all()
        patch = self.out_projection(h[:, mask[0]])
        return h, patch, logp, mask


class ToyDecoder(nn.Module):
    def __init__(self, width, heads, hidden, count):
        super().__init__()
        self.in_projection = nn.Linear(width, width)
        self.initial_norm = nn.LayerNorm(width, bias=False)
        self.layers = nn.ModuleList([ToyLayer(width, heads, hidden) for _ in range(count)])
        self.has_cache = False

    def free_inference_cache(self):
        self.has_cache = False

    def forward(self, embeds, patch_embeds, boundary_mask, **kwargs):
        index = (boundary_mask.long().cumsum(-1) - 1).clamp(min=0)
        h = self.in_projection(embeds) + self.initial_norm(patch_embeds).gather(
            1, index[..., None].expand(-1, -1, embeds.shape[-1]))
        for layer in self.layers:
            h = layer(h)
        return h


class ToyCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        w, h, f = config.hidden_size, config.num_local_heads, config.local_intermediate_size
        self.config = config
        self.local_encoder = ToyEncoder(w, h, f, config.subword_vocab_size)
        self.local_decoder = ToyDecoder(w, h, f, config.num_local_decoder_layers)
        self.layers = nn.ModuleList([nn.Linear(w, w, bias=False)])
        self.norm = nn.LayerNorm(w, bias=False)
        self.rotary_embs = nn.ModuleDict()
        self._tokenizer = None

    def forward(self, input_ids, boundary_mask=None, **kwargs):
        h, patches, _, boundary = self.local_encoder(input_ids, true_boundary_mask=boundary_mask)
        for layer in self.layers:
            patches = patches + layer(patches)
        h = self.local_decoder.forward(h, patches, boundary_mask=boundary)
        return self.norm(h)


class ToyBolmo(nn.Module):
    def __init__(self, width=16, heads=4, hidden=12, suffix_size=32, decoder_layers=4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=width, vocab_size=520, subword_vocab_size=suffix_size,
                                      num_local_heads=heads, local_intermediate_size=hidden,
                                      num_local_decoder_layers=decoder_layers)
        self.model = ToyCore(self.config)
        self.lm_head = nn.Linear(width, 520, bias=False)

    def forward(self, input_ids, boundary_mask=None, **kwargs):
        return SimpleNamespace(logits=self.lm_head(self.model(input_ids, boundary_mask=boundary_mask)))


def toy_selection():
    return {"version": 1, "mode": "slice", "decoder_layers": [0, 2],
            "layers": {key: {"heads": [3, 1], "ffn": [8, 2, 5, 0]}
                       for key in ("encoder.0", "decoder.0", "decoder.2")},
            "router": {"dims": [9, 1, 6, 3]}}


def test_full_selection_preserves_function_and_shares_frozen_weights():
    torch.manual_seed(12)
    teacher = ToyBolmo().eval()
    student = build_student(teacher, full_selection(teacher)).eval()
    ids = bolmo_ids("A résumé, 42!".encode(), "cpu")
    with torch.no_grad():
        torch.testing.assert_close(student(ids).logits, teacher(ids).logits, rtol=0, atol=0)
    assert student.model.layers is teacher.model.layers
    assert student.model.local_encoder.subword_embedding.weight is teacher.model.local_encoder.subword_embedding.weight
    assert student.lm_head.weight is teacher.lm_head.weight
    assert not any(p.requires_grad for p in teacher.parameters())
    assert student.model.local_encoder.layers[0].xlstm.q.weight.requires_grad
    assert student.model.local_encoder.layers[0].xlstm.q.weight is not teacher.model.local_encoder.layers[0].xlstm.q.weight


def test_surgery_preserves_selected_head_and_channel_order():
    torch.manual_seed(7)
    teacher = ToyBolmo()
    student = build_student(teacher, toy_selection())
    old = teacher.model.local_decoder.layers[2]
    new = student.model.local_decoder.layers[1]
    q = torch.tensor([6, 7, 2, 3])
    v = torch.tensor([12, 13, 14, 15, 4, 5, 6, 7])
    channels = torch.tensor([8, 2, 5, 0])
    torch.testing.assert_close(new.xlstm.q.weight, old.xlstm.q.weight[q])
    torch.testing.assert_close(new.xlstm.k.weight, old.xlstm.k.weight[q])
    torch.testing.assert_close(new.xlstm.v.weight, old.xlstm.v.weight[v])
    torch.testing.assert_close(new.xlstm.ogate_preact.weight, old.xlstm.ogate_preact.weight[v])
    torch.testing.assert_close(new.xlstm.igate_preact.bias, old.xlstm.igate_preact.bias[[3, 1]])
    torch.testing.assert_close(new.xlstm.fgate_preact.weight, old.xlstm.fgate_preact.weight[[3, 1]])
    torch.testing.assert_close(new.xlstm.multihead_norm.weight, old.xlstm.multihead_norm.weight[v])
    torch.testing.assert_close(new.xlstm.out_proj.weight, old.xlstm.out_proj.weight[:, v])
    torch.testing.assert_close(new.mlp.gate_proj.weight, old.mlp.gate_proj.weight[channels])
    torch.testing.assert_close(new.mlp.up_proj.weight, old.mlp.up_proj.weight[channels])
    torch.testing.assert_close(new.mlp.down_proj.weight, old.mlp.down_proj.weight[:, channels])
    for projection in ("q_proj_layer", "k_proj_layer"):
        a = getattr(student.model.local_encoder.boundary_predictor_module, projection)
        b = getattr(teacher.model.local_encoder.boundary_predictor_module, projection)
        torch.testing.assert_close(a.weight, b.weight[[9, 1, 6, 3]])
    assert (new.xlstm.config.num_heads, new.xlstm.qk_dim, new.xlstm.v_dim) == (2, 4, 8)
    assert new.xlstm.config.qk_dim_factor == 0.25 and new.xlstm.config.v_dim_factor == 0.5
    assert student(bolmo_ids(b"Hello world", "cpu")).logits.shape == (1, 12, 520)


def test_plan_shapes_and_local_layer_parameter_count():
    with torch.device("meta"):
        teacher = ToyBolmo(width=2048, heads=16, hidden=2816, suffix_size=100278)
        chosen = {"heads": [0, 5, 9, 15], "ffn": list(range(512))}
        selection = {"version": 1, "mode": "slice", "decoder_layers": [1, 3],
                     "layers": {key: chosen for key in ("encoder.0", "decoder.1", "decoder.3")},
                     "router": {"dims": list(range(256))}}
        student = build_student(teacher, selection)
    layer = student.model.local_encoder.layers[0]
    assert (layer.xlstm.qk_dim, layer.xlstm.v_dim, layer.xlstm.config.num_heads) == (256, 512, 4)
    assert layer.mlp.gate_proj.weight.shape == (512, 2048)
    assert student.model.local_encoder.boundary_predictor_module.q_proj_layer.weight.shape == (256, 2048)
    # Two q/k, two v/output-gate, two per-head gates, out projection,
    # three FFN matrices, two residual norms, gate biases and multihead norm.
    expected = (2*256*2048 + 2*512*2048 + 2*4*2048 + 2*4 + 512
                + 2048*512 + 3*512*2048 + 2*2048)
    assert sum(p.numel() for p in layer.parameters()) == expected
    assert len(student.model.local_decoder.layers) == 2
    assert student.config.num_local_heads == 4 and student.config.local_intermediate_size == 512
    assert teacher.config.num_local_heads == 16 and teacher.config.local_intermediate_size == 2816
    cost = dense_local_cost(student)
    assert cost["dense_gflops_per_byte"] == pytest.approx(0.056754176)
    assert cost["encoder_projection_gflops_per_patch"] == pytest.approx(0.008388608)


def test_selection_round_trip_validation_and_svd_router():
    chosen = json.loads(json.dumps(toy_selection()))
    assert chosen == toy_selection()
    teacher = ToyBolmo()
    chosen["layers"]["encoder.0"]["heads"] = [1, 1]
    with pytest.raises(ValueError, match="duplicate"):
        build_student(teacher, chosen)
    chosen = toy_selection()
    chosen["mode"] = "svd"
    chosen["router"] = {"rank": 4}
    student = build_student(teacher, chosen)
    router = student.model.local_encoder.boundary_predictor_module
    q, k = teacher.model.local_encoder.boundary_predictor_module.q_proj_layer.weight, teacher.model.local_encoder.boundary_predictor_module.k_proj_layer.weight
    u, _, _ = torch.linalg.svd(torch.cat((q, k), dim=1), full_matrices=False)
    # Singular-vector signs can differ; their joint row-space projection cannot.
    torch.testing.assert_close(router.q_proj_layer.weight.T @ router.k_proj_layer.weight,
                               q.T @ u[:, :4] @ u[:, :4].T @ k, rtol=1e-4, atol=1e-5)


def test_svd_projection_and_random_control_have_matching_shapes():
    torch.manual_seed(23)
    teacher = ToyBolmo().eval()
    selection = full_selection(teacher)
    selection.update({"mode": "svd", "router": {"rank": 16}, "linear_svd_rank": 16})
    svd_student = build_student(teacher, selection).eval()
    ids = bolmo_ids(b"full rank factors", "cpu")
    with torch.no_grad():
        torch.testing.assert_close(svd_student(ids).logits, teacher(ids).logits, rtol=1e-4, atol=1e-5)
    selection["mode"] = "random"
    control = build_student(teacher, selection)
    svd_shapes = {name: value.shape for name, value in svd_student.state_dict().items()}
    random_shapes = {name: value.shape for name, value in control.state_dict().items()}
    assert svd_shapes == random_shapes
    assert control.lm_head.weight is teacher.lm_head.weight
    assert not torch.equal(control.model.local_encoder.layers[0].xlstm.q.weight,
                           teacher.model.local_encoder.layers[0].xlstm.q.weight)


def test_calibration_measures_layers_and_does_not_modify_teacher(tmp_path):
    torch.manual_seed(88)
    teacher = ToyBolmo().eval()
    before = {name: p.detach().clone() for name, p in teacher.named_parameters()}
    output = tmp_path / "stats.json"
    stats = importance_stats(teacher, [b"First example.", b"Second example."], output)
    assert json.loads(output.read_text(encoding="utf-8")) == stats
    assert stats["bytes"] == len(b"First example.Second example.")
    assert len(stats["layers"]["decoder.2"]["head_output_norm_mean"]) == 4
    assert len(stats["layers"]["encoder.0"]["ffn_activation_times_down_norm_mean"]) == 12
    assert len(stats["decoder_ablation_delta_nats_per_byte"]) == 4
    assert len(stats["router"]["q"]["rms"]) == 16
    for name, p in teacher.named_parameters():
        assert torch.equal(p, before[name])


def test_distillation_common_layout_loss_and_routing():
    torch.manual_seed(51)
    teacher = ToyBolmo().eval()
    student = build_student(teacher, full_selection(teacher))
    loss, metrics = distillation_loss(teacher, student, b"training bytes", mse_weight=0.5)
    assert metrics["kl"] == pytest.approx(0, abs=1e-7)
    assert metrics["patch_mse"] == pytest.approx(0, abs=1e-7)
    assert metrics["boundary_accuracy"] == 1
    loss.backward()
    assert student.model.local_encoder.layers[0].xlstm.q.weight.grad is not None
    assert student.model.local_encoder.boundary_predictor_module.q_proj_layer.weight.grad is not None
    assert not any(p.grad is not None for p in teacher.parameters())
    common = evaluate_pair(teacher, student, b"training bytes")
    own = evaluate_pair(teacher, student, b"training bytes", own_routing=True)
    assert common["teacher_nats"] == pytest.approx(common["student_nats"], abs=1e-6)
    assert own["student_nats"] == pytest.approx(common["student_nats"], abs=1e-6)


def test_optional_unfreezing_keeps_online_teacher_unchanged():
    teacher = ToyBolmo().eval()
    student = build_student(teacher, toy_selection())
    original_head = teacher.lm_head.weight.detach().clone()
    original_suffix = teacher.model.local_encoder.subword_embedding.weight.detach().clone()
    configure_trainable(student, unfreeze_head=True, unfreeze_suffix=True)
    assert student.lm_head.weight is not teacher.lm_head.weight
    assert student.model.local_encoder.subword_embedding.weight is not teacher.model.local_encoder.subword_embedding.weight
    loss, _ = distillation_loss(teacher, student, b"training bytes")
    loss.backward()
    assert student.lm_head.weight.grad is not None
    assert student.model.local_encoder.subword_embedding.weight.grad is not None
    with torch.no_grad():
        student.lm_head.weight.add_(0.1)
        student.model.local_encoder.subword_embedding.weight.add_(0.1)
    assert torch.equal(teacher.lm_head.weight, original_head)
    assert torch.equal(teacher.model.local_encoder.subword_embedding.weight, original_suffix)


class TinyWindows:
    def __init__(self):
        self.raw = [b"abcdefgh", b"hijklmn!", b"12345678", b"question"]
        self.index_sha256 = "synthetic-index-v1"

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        return SimpleNamespace(data=self.raw[index])


def test_distillation_checkpoint_resume_and_delta_contents(tmp_path):
    torch.manual_seed(61)
    initial = ToyBolmo().eval()
    config = DistillConfig(window_bytes=8, windows_per_step=2, init_seed=61, data_seed=3,
                           learning_rate=1e-3, warmup_steps=2, schedule_steps=10,
                           val_interval_steps=2, gradient_checkpointing=True)
    selection = toy_selection()

    def make(run_dir):
        teacher = copy.deepcopy(initial)
        student = build_student(teacher, selection)
        return DistillTrainer(teacher, student, selection, {"checkpoint": "fixture"}, TinyWindows(), "data-hash",
                              [b"val text"], run_dir, config)

    uninterrupted = make(tmp_path / "full")
    expected_samples = [uninterrupted.train_step()["sample_indices"] for _ in range(3)]
    interrupted = make(tmp_path / "resume")
    first = interrupted.train_step()
    path = interrupted.save()
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert "model" not in saved and "trainable" in saved
    assert not any("subword_embedding" in key or key.startswith("lm_head") or key.startswith("model.layers")
                   for key in saved["trainable"])
    assert saved["sampler_cursor"] == 2 and saved["sample_audit"]["indices"] == first["sample_indices"]
    eval_student, metadata = load_student_checkpoint(copy.deepcopy(initial), path, {"checkpoint": "fixture"})
    assert metadata["step"] == 1
    for name, parameter in eval_student.named_parameters():
        if name in saved["trainable"]:
            torch.testing.assert_close(parameter, saved["trainable"][name])
    with pytest.raises(ValueError, match="teacher identity"):
        load_student_checkpoint(copy.deepcopy(initial), path, {"checkpoint": "different"})
    resumed = make(tmp_path / "resume")
    resumed.load()
    remaining = [resumed.train_step()["sample_indices"] for _ in range(2)]
    assert [first["sample_indices"], *remaining] == expected_samples
    assert resumed.consumed_bytes == uninterrupted.consumed_bytes
    for name, value in uninterrupted.trainables.items():
        torch.testing.assert_close(value, resumed.trainables[name], rtol=0, atol=0)
    assert resumed.validate()["teacher_bpb"] > 0


def test_pause_file_creates_initial_checkpoint_without_update(tmp_path):
    torch.manual_seed(2)
    teacher = ToyBolmo()
    selection = toy_selection()
    student = build_student(teacher, selection)
    config = DistillConfig(window_bytes=8, gradient_checkpointing=False)
    trainer = DistillTrainer(teacher, student, selection, {"id": "toy"}, TinyWindows(), "data",
                             [b"val text"], tmp_path / "run", config)
    pause = tmp_path / "pause"
    pause.touch()
    trainer.run(8, pause)
    status = json.loads((tmp_path / "run" / "status.json").read_text())
    assert status["status"] == "paused" and trainer.step == 0
    assert (tmp_path / "run" / "ckpt" / "latest.json").exists()


def test_run_resume_discards_uncommitted_log_and_continues(tmp_path):
    torch.manual_seed(19)
    initial = ToyBolmo().eval()
    selection = toy_selection()
    cfg = DistillConfig(window_bytes=8, gradient_checkpointing=False, warmup_steps=1, schedule_steps=10)

    def make():
        teacher = copy.deepcopy(initial)
        student = build_student(teacher, selection)
        return DistillTrainer(teacher, student, selection, {"id": "run-test"}, TinyWindows(), "data",
                              [b"val text"], tmp_path / "run", cfg)

    trainer = make()
    trainer.run(8, tmp_path / "no-pause")
    assert trainer.step == 1
    log_path = tmp_path / "run" / "log.jsonl"
    with log_path.open("a", encoding="utf-8") as log:
        log.write('{"step": 2}\n{')
    resumed = make()
    resumed.load()
    assert [json.loads(line)["step"] for line in log_path.read_text().splitlines()] == [1]
    resumed.run(16, tmp_path / "no-pause")
    assert resumed.step == 2 and resumed.consumed_bytes == 16
    assert [json.loads(line)["step"] for line in log_path.read_text().splitlines()] == [1, 2]


@pytest.mark.parametrize("lora_rank", [0, 2])
def test_pinned_bolmo_tiny_cpu_parity_and_checkpointed_training(tmp_path, monkeypatch, lora_rank):
    pytest.importorskip("transformers")
    pytest.importorskip("mlstm_kernels")
    _cuda_home()
    try:
        from reference.bolmo_hf.modeling_bolmo import BolmoConfig, BolmoForCausalLM
    except ImportError as exc:
        pytest.skip(f"pinned Bolmo dependencies are unavailable: {exc}")
    # The HF constructor chooses its backend from global CUDA availability;
    # force the native backend for this deliberately CPU-sized check.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = BolmoConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, num_local_heads=4, local_intermediate_size=48,
                      num_local_decoder_layers=4, add_expanded_embeddings=False,
                      layer_types=["full_attention"] * 2, sliding_window=None, pad_token_id=0,
                      bos_token_id=1, eos_token_id=1, max_position_embeddings=256)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(42)
    teacher = BolmoForCausalLM(cfg).eval()
    student = build_student(teacher, full_selection(teacher)).eval()
    ids = bolmo_ids(b"real tiny Bolmo", "cpu")
    with torch.no_grad():
        torch.testing.assert_close(teacher(input_ids=ids, use_cache=False).logits,
                                   student(input_ids=ids, use_cache=False).logits, rtol=0, atol=0)
    stats = importance_stats(teacher, [b"calibration bytes"])
    assert len(stats["layers"]["encoder.0"]["head_output_norm_mean"]) == 4
    assert len(stats["decoder_ablation_delta_nats_per_byte"]) == 4
    # The BOS makes this sequence length non-divisible by xLSTM's 64-token
    # training chunk.  Exercise both the padding and checkpoint recomputation.
    selection = {"version": 1, "mode": "slice", "decoder_layers": [0, 2],
                 "layers": {key: {"heads": [0, 2], "ffn": list(range(16))}
                            for key in ("encoder.0", "decoder.0", "decoder.2")},
                 "router": {"dims": list(range(8))}}
    student = build_student(teacher, selection)
    config = DistillConfig(window_bytes=8, gradient_checkpointing=True, lora_rank=lora_rank,
                           warmup_steps=2, schedule_steps=10)
    trainer = DistillTrainer(teacher, student, selection, {"id": "real-tiny"}, TinyWindows(), "manifest",
                             [b"val text"], tmp_path / "tiny", config)
    before = teacher.model.layers[0].self_attn.q_proj.weight.detach().clone()
    metrics = trainer.train_step()
    assert math.isfinite(metrics["kl"]) and metrics["consumed_bytes"] == 8
    assert torch.equal(before, teacher.model.layers[0].self_attn.q_proj.weight)
    if lora_rank:
        assert trainer.student.model.layers[0].self_attn.q_proj.base.weight is teacher.model.layers[0].self_attn.q_proj.weight
        assert trainer.student.model.layers[0].self_attn.q_proj.b.grad is not None
        checkpoint_path = trainer.save()
        fresh_teacher = BolmoForCausalLM(copy.deepcopy(cfg)).eval()
        fresh_teacher.load_state_dict(teacher.state_dict())
        restored, _ = load_student_checkpoint(fresh_teacher, checkpoint_path, {"id": "real-tiny"})
        trainer.student.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(input_ids=ids, use_cache=False).logits,
                                       trainer.student(input_ids=ids, use_cache=False).logits, rtol=0, atol=0)
    else:
        assert trainer.student.model.layers is teacher.model.layers
    trainer.student.eval()
    trainer.student.model._tokenizer = SimpleNamespace(pad_token_id=0, offset=4, eos_token_id=1, bpe_token_end_id=3)
    prompt = bolmo_ids(b"prompt text", "cpu")
    try:
        generated = trainer.student.generate(prompt, max_new_tokens=3, do_sample=False)
        assert generated.shape[1] > prompt.shape[1]
        assert torch.equal(generated[:, :prompt.shape[1]], prompt)
    finally:
        trainer.student.model.local_encoder.free_inference_cache()
        trainer.student.model.local_decoder.free_inference_cache()


def test_cuda_real_checkpoint_function_preservation():
    path = Path(os.environ.get("TRACKA_BOLMO", "~/blt/models/Bolmo-1B")).expanduser()
    if not torch.cuda.is_available() or not path.is_dir():
        pytest.skip("requires CUDA and the local Bolmo-1B checkpoint")
    bundle = load_reference("bolmo", path, bolmo_subword_tokenizer=os.environ.get("TRACKA_BOLMO_TOKENIZER"))
    teacher = bundle.model.eval()
    student = build_student(teacher, full_selection(teacher)).eval()
    ids = bolmo_ids("Real bytes: café, 42!".encode(), bundle.device)
    with torch.inference_mode():
        reference = teacher(input_ids=ids, use_cache=False).logits
        actual = student(input_ids=ids, use_cache=False).logits
        torch.testing.assert_close(actual.float(), reference.float(), atol=2e-2, rtol=2e-2)
    del student, actual, reference
    torch.cuda.empty_cache()
    selection = {"version": 1, "mode": "slice", "decoder_layers": [0, 2],
                 "layers": {key: {"heads": [0, 3, 7, 15], "ffn": list(range(512))}
                            for key in ("encoder.0", "decoder.0", "decoder.2")},
                 "router": {"dims": list(range(256))}}
    student = build_student(teacher, selection)
    configure_trainable(student)
    layer = student.model.local_encoder.layers[0]
    assert (layer.xlstm.qk_dim, layer.xlstm.v_dim, layer.xlstm.config.num_heads) == (256, 512, 4)
    loss, metrics = distillation_loss(teacher, student, b"A CUDA training step.")
    assert math.isfinite(metrics["kl"])
    loss.backward()
    assert layer.xlstm.q.weight.grad is not None
    assert not any(parameter.grad is not None for parameter in teacher.parameters())
