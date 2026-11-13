import copy
import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from blt.config import BPEConfig, ByteConfig, TransformerConfig
from blt.probe import (ExtractionBudget, ExtractionBudgetExceeded, aligned_states, cache_fingerprint, collect_features,
                       fit_probes, load_features, prepare_examples, probe_criteria, save_features)
from blt.train import make_model, seed_everything


def _models():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(["A. abcXYZ tail and digits 42. walking walked."], trainers.BpeTrainer(
        vocab_size=257, special_tokens=["<|bos|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    trunk = TransformerConfig(2, 16, 2, 1, 8, 32, max_positions=128)
    byte = ByteConfig("byte", trunk, 16, 2, 32, 2, 6, max_len=8, block_size=8, hash_rows=8)
    bpe = BPEConfig("bpe", trunk, tokenizer.get_vocab_size(), tokenizer.token_to_id("<|bos|>"))
    seed_everything(1)
    return make_model(byte).eval(), make_model(bpe).eval(), tokenizer


def test_alignment_is_causal_uses_global_path_and_flags_forced_closure():
    byte, bpe, tokenizer = _models()
    first = aligned_states(byte, bpe, tokenizer, "A. future", 2)
    second = aligned_states(byte, bpe, tokenizer, "A. altered", 2)
    assert first is not None and second is not None
    assert not first["forced_patch_closure"]
    assert first["prefix_sha256"] == second["prefix_sha256"]
    for family in ("byte", "bpe"):
        for layer in ("input", "middle", "final"):
            np.testing.assert_allclose(first[family][layer], second[family][layer], atol=1e-6)
    # Probe states must never depend on the byte decoder or its local skip path.
    with torch.no_grad():
        for parameter in byte.decoder.parameters():
            parameter.add_(100)
        byte.condition.weight.add_(100)
    after = aligned_states(byte, bpe, tokenizer, "A. future", 2)
    for layer in ("input", "middle", "final"):
        np.testing.assert_allclose(first["byte"][layer], after["byte"][layer], atol=1e-6)
    assert aligned_states(byte, bpe, tokenizer, "abcXYZ", 3) is None
    forced = aligned_states(byte, bpe, tokenizer, "abcXYZ", 3, allow_forced_patch=True)
    assert forced is not None and forced["forced_patch_closure"] is True


def test_numeric_and_suffix_annotations_are_validated_and_types_are_disjoint():
    records = []
    for value in list(range(10, 30)) + list(range(100, 120)):
        text = f"Value {value}."
        records.append({"text": text, "target_end_byte": len(text) - 1, "anchor_byte": len(text), "attribute": "numeric"})
    for lemma in ("walk", "talk", "play", "jump", "look", "help", "call", "work"):
        for suffix in ("ing", "ed"):
            text = f"We {lemma}{suffix}."
            records.append({"text": text, "target_end_byte": len(text) - 1, "anchor_byte": len(text),
                            "attribute": "suffix", "lemma": lemma, "suffix": suffix})
    examples = prepare_examples(records)
    assert {row["attribute"] for row in examples} == {"parity", "magnitude", "suffix"}
    rng = np.random.default_rng(2)
    features = {}
    for model in ("byte", "bpe", "untrained_byte", "untrained_bpe"):
        features[model] = {layer: rng.normal(size=(len(examples), 4)) for layer in ("input", "middle", "final")}
    results, fitted = fit_probes(examples, features, seed=3)
    for attribute in ("parity", "magnitude", "suffix"):
        result = results[attribute]
        assert result["status"] == "run"
        assert not (set(result["train_groups"]) & set(result["test_groups"]))
        assert "random_label_test_accuracy" in result["models"]["byte"]["final"]
        assert "untrained_byte" in result["models"] and (attribute, "byte") in fitted
    bad = [{"text": "Number 42.", "anchor_byte": 9, "attribute": "numeric", "value": 12}]
    with pytest.raises(ValueError, match="disagrees"):
        prepare_examples(bad)


def test_feature_cache_checks_identity_and_preserves_alignment(tmp_path):
    examples = prepare_examples([{"text": "7.", "target_end_byte": 1, "anchor_byte": 2, "attribute": "numeric"}])
    identities = {"byte": {"sha256": "b"}, "bpe": {"sha256": "t"}}
    fingerprint = cache_fingerprint(examples, identities, seed=1, forced=True, max_prefix_bytes=10)
    features = {model: {layer: np.ones((2, 3)) for layer in ("input", "middle", "final")}
                for model in ("byte", "bpe", "untrained_byte", "untrained_bpe")}
    alignment = {"retained_indices": [0, 1], "anchors": [{"forced_patch_closure": True}] * 2}
    path = tmp_path / "states.npz"
    save_features(path, fingerprint, features, alignment)
    kept, recovered, meta = load_features(path, fingerprint, examples)
    assert kept == examples and meta == alignment
    np.testing.assert_array_equal(recovered["byte"]["input"], features["byte"]["input"])
    with pytest.raises(ValueError, match="does not match"):
        load_features(path, "another fingerprint", examples)


def test_no_aligned_examples_reports_not_run_instead_of_inventing_states():
    byte, bpe, tokenizer = _models()
    examples = prepare_examples([{"text": "123xyz", "anchor_byte": 3, "attribute": "numeric"}])
    kept, features, alignment = collect_features(examples, byte, bpe, copy.deepcopy(byte), copy.deepcopy(bpe), tokenizer)
    assert kept == [] and alignment["skipped"]["not_aligned"] == 2
    result, _ = fit_probes(kept, features)
    assert all(value["status"] == "not run" for value in result.values())


def test_extraction_budget_stops_before_starting_another_model_call(monkeypatch):
    import blt.probe as module

    times = iter([0.0, 1.0])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(times, 1.0))
    budget = ExtractionBudget(1)
    assert budget.run(lambda: "done") == "done"
    with pytest.raises(ExtractionBudgetExceeded):
        budget.run(lambda: pytest.fail("the budget should prevent this call"))
    with pytest.raises(ValueError):
        ExtractionBudget(3601)


def test_probe_criteria_require_replicates_and_digit_sensitivity():
    probes = {name: {"status": "run", "models": {"byte": {"controlled_final_gain": gain}}}
              for name, gain in (("parity", .05), ("magnitude", .04), ("suffix", .01))}
    def side(excess, change=True):
        return {"excess_nats_per_byte": excess, "frozen_probes": {"parity": {
            "original_label": "odd", "edited_label": "even", "original_prediction": "odd",
            "edited_prediction": "even" if change else "odd"}}}
    pairs = [{"kind": kind, "byte": side(.08), "bpe": side(.1)}
             for kind in ("extra_space", "newline", "casing", "digit")]
    base = {"type": "representation_probes_v1", "dataset_sha256": "same", "split_seed": 0,
            "model_names": {"byte": "M-P6", "bpe": "M-BPE50K"}, "training_seeds": {"byte": 0, "bpe": 0},
            "checkpoints": {"byte": {"sha256": "byte-0"}, "bpe": {"sha256": "bpe-0"}},
            "probes": probes, "robustness": {"pairs": pairs, "groups": {
                kind: {"byte": {"mean_state_drift": .2}} for kind in ("extra_space", "newline", "casing", "digit")}}}
    assert probe_criteria(base)["overall"]["status"] == "not run"
    repeat = copy.deepcopy(base)
    repeat["training_seeds"] = {"byte": 1, "bpe": 1}
    repeat["checkpoints"] = {"byte": {"sha256": "byte-1"}, "bpe": {"sha256": "bpe-1"}}
    assert probe_criteria(base, repeat)["overall"]["status"] == "pass"
    base["robustness"]["pairs"][-1]["byte"] = side(.08, change=False)
    assert probe_criteria(base, repeat)["overall"]["status"] == "fail"
