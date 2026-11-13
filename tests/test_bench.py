import pytest
import copy
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from blt.bench import InferenceCase, benchmark, compile_inference, paired_speed_interval
from blt.config import BPEConfig, ByteConfig, TransformerConfig
from blt.eval_io import EvaluationModel
from blt.eval_tokens import TokenBytes
from blt.model import BPELM, ByteLatentLM


def test_cpu_inference_benchmark_structure():
    trunk = TransformerConfig(1, 16, 2, 1, 8, 32, max_positions=128)
    config = ByteConfig("tiny", trunk, 16, 2, 32, 3, 6, max_len=7, hash_enabled=False)
    bundle = EvaluationModel(ByteLatentLM(config).eval(), config, None, torch.device("cpu"), {"kind": "random"})
    prompts = [b"A short, ordinary prompt for timing."]
    prefill = benchmark([bundle], prompts, mode="prefill", batch=1, output_bytes=4, warmup=0, repetitions=1)
    assert set(prefill["variants"]) == {"gpu_resident", "gpu_resident_preprocessed", "end_to_end"}
    for variant in prefill["variants"].values():
        result = variant["models"][0]
        assert result["bytes"] == [len(prompts[0])]
        assert result["median_seconds"] > 0 and result["cache"][0]["logical_bytes"] > 0
    for mode in ("decode", "e2e"):
        result = benchmark([bundle], prompts, mode=mode, batch=1, output_bytes=4, warmup=0, repetitions=1)
        row = result["variants"][mode]["models"][0]
        assert row["bytes"] == [4]
        assert row["overshoot_bytes"] == [[0]]


def test_preprocessed_bench_inputs_match_host_prefill_for_both_families():
    torch.manual_seed(54)
    prompt = b"Ordinary text, with commas!"
    trunk = TransformerConfig(1, 16, 2, 1, 8, 32, max_positions=128)
    byte_config = ByteConfig("tiny", trunk, 16, 2, 32, 3, 6, max_len=7, hash_rows=32)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator([prompt.decode()], trainers.BpeTrainer(
        vocab_size=280, special_tokens=["<|bos|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    bpe_config = BPEConfig("tiny-bpe", trunk, tokenizer.get_vocab_size(), tokenizer.token_to_id("<|bos|>"))
    bundles = [EvaluationModel(ByteLatentLM(byte_config).eval(), byte_config, None, torch.device("cpu"), {"kind": "byte"}),
               EvaluationModel(BPELM(bpe_config).eval(), bpe_config, tokenizer, torch.device("cpu"), {"kind": "bpe"})]
    for bundle in bundles:
        case = InferenceCase(bundle, prompt, 2, output_bytes=4, prepare_preprocessed=True)
        _, expected, _ = case.prefill(resident=False)
        _, resident, _ = case.prefill(resident=True)
        _, preprocessed, _ = case.prefill(resident=True, preprocessed=True)
        torch.testing.assert_close(resident, expected, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(preprocessed, expected, rtol=2e-5, atol=2e-5)
    result = benchmark(bundles, [prompt], mode="prefill", batch=1, warmup=0, repetitions=1)
    for variant in result["variants"].values():
        assert len(variant["models"]) == 2 and "paired_bootstrap" in variant
    with pytest.raises(ValueError, match="requires CUDA"):
        benchmark(bundles, [prompt], mode="prefill", batch=1, warmup=0, repetitions=1, execution="graph")


def test_paired_bootstrap_uses_aligned_repetitions():
    result = paired_speed_interval([1, 2, 3], [2, 4, 6], seed=3, samples=100)
    assert result["speed_a_over_b"] == pytest.approx(2)
    assert result["ci95_low"] == pytest.approx(2)
    assert result["ci95_high"] == pytest.approx(2)
    overshoot = paired_speed_interval([1, 2, 3], [1, 2, 3], a_bytes=[256] * 3, b_bytes=[260] * 3, seed=3, samples=100)
    assert overshoot["speed_a_over_b"] == pytest.approx(256 / 260)


def test_bytelevel_token_output_preserves_partial_utf8_bytes():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(["café 🙂 testing"], trainers.BpeTrainer(vocab_size=270, special_tokens=["<|bos|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    raw = "café 🙂".encode("utf-8")
    ids = tokenizer.encode(raw.decode("utf-8"), add_special_tokens=False).ids
    vocabulary = TokenBytes(tokenizer, tokenizer.get_vocab_size())
    assert b"".join(vocabulary.values[index] for index in ids) == raw


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 inference compilation requires CUDA")
@pytest.mark.parametrize("family", ["byte", "bpe"])
def test_cuda_compiled_inference_matches_eager(family):
    trunk = TransformerConfig(1, 16, 2, 1, 8, 32, max_positions=128)
    if family == "byte":
        config = ByteConfig("tiny", trunk, 16, 2, 32, 3, 6, max_len=7, hash_rows=32)
        model = ByteLatentLM(config)
        prefix = torch.tensor([[1, 2, 32, 4, 5, 33, 7]], device="cuda")
        following = torch.tensor([32], device="cuda")
    else:
        config = BPEConfig("tiny", trunk, 32, 0)
        model = BPELM(config)
        prefix = torch.tensor([[0, 1, 2, 3, 4, 5, 6]], device="cuda")
        following = torch.tensor([7], device="cuda")
    eager, compiled = model.cuda().eval(), copy.deepcopy(model).cuda().eval()
    compile_inference(compiled)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        eager_state = eager.init_state(1, 32, "cuda", torch.bfloat16)
        compiled_state = compiled.init_state(1, 32, "cuda", torch.bfloat16)
        eager_logits = eager.prefill(eager_state, prefix)
        compiled_logits = compiled.prefill(compiled_state, prefix)
        torch.testing.assert_close(eager_logits, compiled_logits, atol=2e-2, rtol=2e-2)
        eager_logits = eager.step(eager_state, following)
        compiled_logits = compiled.step(compiled_state, following)
        torch.testing.assert_close(eager_logits, compiled_logits, atol=2e-2, rtol=2e-2)
