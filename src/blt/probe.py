from __future__ import annotations

"""Causally aligned global-state probes and perturbation controls."""

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .eval_tokens import TokenBytes
from .hashes import hashes_at_ends
from .model import BPELM, ByteLatentLM
from .patcher import patch_batch
from .score import autocast_for
from .tasks import continuation_nll, robustness_pairs


ATTRIBUTES = ("parity", "magnitude", "suffix")
LAYERS = ("input", "middle", "final")
SUFFIXES = ("ing", "ed", "ly", "s")


class ExtractionBudgetExceeded(RuntimeError):
    pass


class ExtractionBudget:
    """Conservative wall-time cap on model extraction and scoring calls."""
    def __init__(self, seconds: float = 3600) -> None:
        if not 0 < seconds <= 3600:
            raise ValueError("representation extraction budget must be at most 3600 seconds")
        self.limit, self.elapsed = float(seconds), 0.0

    def run(self, function):
        if self.elapsed >= self.limit:
            raise ExtractionBudgetExceeded("representation extraction budget exhausted")
        start = time.monotonic()
        try:
            return function()
        finally:
            self.elapsed += time.monotonic() - start


def _budgeted(budget: ExtractionBudget | None, function):
    return function() if budget is None else budget.run(function)


def magnitude(value: int) -> str:
    if value < 0:
        raise ValueError("numeric probes use nonnegative integers")
    return "0-9" if value < 10 else "10-99" if value < 100 else "100-999" if value < 1000 else "1000+"


def prepare_examples(records: list[dict]) -> list[dict]:
    """Validate annotated UTF-8 target ends and derive labels and disjoint groups.

    A ``numeric`` record expands to both numeric tasks. Morphology requires a
    supplied lemma and suffix annotation; the tool does not guess lemmas.
    """
    examples = []
    suffix_forms: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(records):
        text, anchor = row.get("text"), row.get("anchor_byte")
        if not isinstance(text, str) or not isinstance(anchor, int) or isinstance(anchor, bool):
            raise ValueError(f"record {index}: text and integer anchor_byte are required")
        raw = text.encode("utf-8")
        target_end = row.get("target_end_byte", anchor)
        if (not isinstance(target_end, int) or isinstance(target_end, bool)
                or not 0 < target_end <= anchor <= len(raw)):
            raise ValueError(f"record {index}: invalid target or anchor byte offset")
        try:
            prefix = raw[:target_end].decode("utf-8")
            raw[:anchor].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"record {index}: offsets must be UTF-8 boundaries") from error
        attribute = row.get("attribute")
        base = {"record": index, "text": text, "anchor_byte": anchor, "target_end_byte": target_end}
        if attribute in {"numeric", "parity", "magnitude"}:
            match = re.search(r"[0-9]+$", prefix)
            if match is None:
                raise ValueError(f"record {index}: numeric target must end at target_end_byte")
            value = int(match.group())
            if "value" in row and str(row["value"]) != str(value):
                raise ValueError(f"record {index}: numeric value disagrees with the source text")
            for task in (("parity", "magnitude") if attribute == "numeric" else (attribute,)):
                label = ("even" if value % 2 == 0 else "odd") if task == "parity" else magnitude(value)
                examples.append({**base, "attribute": task, "label": label, "group": str(value), "value": value})
        elif attribute == "suffix":
            match = re.search(r"[A-Za-z]+$", prefix)
            suffix, lemma = row.get("suffix"), row.get("lemma")
            if (match is None or not isinstance(suffix, str) or suffix.casefold() not in SUFFIXES
                    or not isinstance(lemma, str) or not lemma.strip()
                    or not match.group().casefold().endswith(suffix.casefold())):
                raise ValueError(f"record {index}: suffix annotation and lemma must match the target word")
            form, annotation = match.group().casefold(), (lemma.strip().casefold(), suffix.casefold())
            if form in suffix_forms and suffix_forms[form] != annotation:
                raise ValueError(f"record {index}: the same word has conflicting lemma or suffix annotations")
            suffix_forms[form] = annotation
            examples.append({**base, "attribute": "suffix", "label": suffix.casefold(),
                             "group": lemma.strip().casefold(), "word": form})
        else:
            raise ValueError(f"record {index}: unknown attribute")
    return examples


def _stages(module: Any, x: torch.Tensor) -> dict[str, torch.Tensor]:
    if x.shape[1] > module.config.max_positions:
        raise ValueError("probe prefix exceeds the model position limit")
    result = {"input": x[:, -1, :]}
    middle = (len(module.blocks) + 1) // 2
    for index, block in enumerate(module.blocks, 1):
        x, _ = block(x)
        if index == middle:
            result["middle"] = x[:, -1, :]
    result["final"] = module.norm(x)[:, -1, :]
    return result


def _token_prefix(tokenizer: Any, token_bytes: TokenBytes, text: str, anchor: int) -> list[int] | None:
    raw = text.encode("utf-8")
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    pieces = [token_bytes.values[token] if 0 <= token < len(token_bytes.values) else None for token in ids]
    if any(piece is None for piece in pieces) or b"".join(pieces) != raw:
        raise ValueError("BPE tokens do not exactly reconstruct the original UTF-8 bytes")
    position = 0
    for index, piece in enumerate(pieces):
        position += len(piece)
        if position == anchor:
            prefix_ids = ids[:index + 1]
            # The observed token end must also survive tokenizing the causal
            # prefix alone; otherwise suffix-dependent segmentation leaks in.
            independent = tokenizer.encode(raw[:anchor].decode("utf-8"), add_special_tokens=False).ids
            return prefix_ids if prefix_ids == independent else None
        if position > anchor:
            return None
    return [] if anchor == 0 else None


@torch.inference_mode()
def aligned_states(byte_model: ByteLatentLM, bpe_model: BPELM, tokenizer: Any, text: str, anchor: int, *,
                   allow_forced_patch: bool = False, token_bytes: TokenBytes | None = None) -> dict | None:
    """Extract states only at the same raw prefix and an actual BPE token end.

    If the byte patch has not naturally closed, an optional analysis-only final
    closure is recorded. No byte after the anchor is fed to either model.
    """
    if not isinstance(byte_model, ByteLatentLM) or not isinstance(bpe_model, BPELM):
        raise ValueError("alignment requires one byte model and one BPE model")
    raw = text.encode("utf-8")
    if not 0 < anchor <= len(raw):
        raise ValueError("anchor is outside the text")
    raw[:anchor].decode("utf-8")
    token_bytes = token_bytes or TokenBytes(tokenizer, bpe_model.config.vocab_size)
    ids = _token_prefix(tokenizer, token_bytes, text, anchor)
    if ids is None:
        return None
    c = byte_model.config
    prefix = raw[:anchor]
    patches = patch_batch(np.frombuffer(prefix, dtype=np.uint8)[None, :], c.min_len, c.max_len, c.block_size)
    forced = not bool(patches.boundaries[0, -1])
    if forced and not allow_forced_patch:
        return None
    ends = list(patches.ends[0, :int(patches.counts[0])])
    if forced:
        ends.append(anchor - 1)
    device_b = next(byte_model.parameters()).device
    device_t = next(bpe_model.parameters()).device
    raw_tensor = torch.tensor(list(prefix), dtype=torch.long, device=device_b)[None, :]
    end_tensor = torch.tensor(ends, dtype=torch.long, device=device_b)[None, :]
    byte_model.eval()
    bpe_model.eval()
    with autocast_for(device_b):
        enc, _ = byte_model._local(byte_model.byte_embedding(raw_tensor), byte_model.encoder)
        selected = enc.gather(1, end_tensor.unsqueeze(-1).expand(-1, -1, c.local_width))
        pooled = byte_model.pool(selected)
        if byte_model.hash_embedding is not None:
            pooled = pooled + byte_model._hash_vectors(hashes_at_ends(
                raw_tensor, end_tensor, c.hash_rows, c.hash_seed, parameters=byte_model.hash_parameters))
        global_input = torch.cat((byte_model.global_bos.expand(1, 1, -1), pooled), dim=1)
        byte_states = _stages(byte_model.global_trunk, global_input)
    with autocast_for(device_t):
        tokens = torch.tensor([bpe_model.config.bos_id, *ids], device=device_t, dtype=torch.long)[None, :]
        bpe_states = _stages(bpe_model.trunk, bpe_model.embedding(tokens))
    return {"byte": {layer: value[0].float().cpu().numpy().copy() for layer, value in byte_states.items()},
            "bpe": {layer: value[0].float().cpu().numpy().copy() for layer, value in bpe_states.items()},
            "forced_patch_closure": forced, "anchor_byte": anchor,
            "prefix_sha256": hashlib.sha256(prefix).hexdigest(), "bpe_tokens": len(ids), "byte_patches": len(ends)}


def group_split(examples: list[dict], indices: list[int], *, seed: int, test_fraction: float = .25) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import GroupShuffleSplit

    if not 0 < test_fraction < 1:
        raise ValueError("test fraction must be between zero and one")
    groups = np.asarray([examples[i]["group"] for i in indices])
    labels = np.asarray([examples[i]["label"] for i in indices])
    if len(set(groups)) < 4 or len(set(labels)) < 2:
        raise ValueError("at least four disjoint groups and two classes are needed")
    for attempt in range(128):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed + attempt)
        train, test = next(splitter.split(np.zeros(len(indices)), labels, groups))
        if set(labels[train]) == set(labels[test]) == set(labels) and not (set(groups[train]) & set(groups[test])):
            return np.asarray(indices)[train], np.asarray(indices)[test]
    raise ValueError("could not create a value/lemma-disjoint split containing every class")


def _new_probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))


def fit_probes(examples: list[dict], features: dict[str, dict[str, np.ndarray]], *, seed: int = 0,
               test_fraction: float = .25) -> tuple[dict, dict]:
    """Fit fixed probes and shuffled-label controls on disjoint types."""
    needed = {"byte", "bpe", "untrained_byte", "untrained_bpe"}
    if set(features) != needed:
        raise ValueError("trained and untrained controls for both model families are required")
    if any(set(layers) != set(LAYERS) or any(len(array) != len(examples) for array in layers.values())
           for layers in features.values()):
        raise ValueError("feature arrays must align with every example and layer")
    labels = np.asarray([row["label"] for row in examples])
    result, fitted = {}, {}
    for attribute in ATTRIBUTES:
        indices = [i for i, row in enumerate(examples) if row["attribute"] == attribute]
        try:
            train, test = group_split(examples, indices, seed=seed, test_fraction=test_fraction)
        except ValueError as error:
            result[attribute] = {"status": "not run", "reason": str(error)}
            continue
        record = {"status": "run", "train_count": len(train), "test_count": len(test),
                  "train_groups": sorted({examples[i]["group"] for i in train}),
                  "test_groups": sorted({examples[i]["group"] for i in test}), "models": {}}
        rng = np.random.default_rng(seed)
        random_labels = rng.permutation(labels[train])
        for name, layers in features.items():
            record["models"][name] = {}
            for layer, x in layers.items():
                probe = _new_probe()
                probe.fit(x[train], labels[train])
                random_probe = _new_probe()
                random_probe.fit(x[train], random_labels)
                record["models"][name][layer] = {
                    "train_accuracy": float(np.mean(probe.predict(x[train]) == labels[train])),
                    "test_accuracy": float(np.mean(probe.predict(x[test]) == labels[test])),
                    "random_label_test_accuracy": float(np.mean(random_probe.predict(x[test]) == labels[test]))}
                if name in {"byte", "bpe"} and layer == "final":
                    fitted[(attribute, name)] = probe
        for family in ("byte", "bpe"):
            trained = record["models"][family]
            untrained = record["models"][f"untrained_{family}"]
            baseline = max(trained["input"]["test_accuracy"], trained["final"]["random_label_test_accuracy"],
                           untrained["final"]["test_accuracy"])
            record["models"][family]["controlled_final_gain"] = trained["final"]["test_accuracy"] - baseline
            record["models"][family]["control_baseline"] = baseline
        result[attribute] = record
    return result, fitted


def collect_features(examples: list[dict], byte_model: ByteLatentLM, bpe_model: BPELM,
                     untrained_byte: ByteLatentLM, untrained_bpe: BPELM, tokenizer: Any, *,
                     allow_forced_patch: bool = False, max_prefix_bytes: int = 512,
                     budget: ExtractionBudget | None = None) -> tuple[list[dict], dict, dict]:
    if max_prefix_bytes <= 0:
        raise ValueError("maximum prefix length must be positive")
    if byte_model.config != untrained_byte.config or bpe_model.config != untrained_bpe.config:
        raise ValueError("untrained controls must use the identical model configurations")
    token_bytes = TokenBytes(tokenizer, bpe_model.config.vocab_size)
    arrays = {name: {layer: [] for layer in LAYERS} for name in ("byte", "bpe", "untrained_byte", "untrained_bpe")}
    kept, alignment, retained_indices = [], [], []
    counts = {"too_long": 0, "not_aligned": 0, "budget_exhausted": 0}
    cached: dict[tuple[str, int], tuple[dict | None, dict | None]] = {}
    for index, example in enumerate(examples):
        anchor = example["anchor_byte"]
        if anchor > max_prefix_bytes:
            counts["too_long"] += 1
            continue
        key = (example["text"], anchor)
        if key not in cached:
            try:
                trained = _budgeted(budget, lambda: aligned_states(byte_model, bpe_model, tokenizer, key[0], anchor,
                                                                   allow_forced_patch=allow_forced_patch, token_bytes=token_bytes))
                control = (_budgeted(budget, lambda: aligned_states(untrained_byte, untrained_bpe, tokenizer, key[0], anchor,
                                                                    allow_forced_patch=allow_forced_patch, token_bytes=token_bytes))
                           if trained is not None else None)
            except ExtractionBudgetExceeded:
                counts["budget_exhausted"] = len(examples) - index
                break
            cached[key] = trained, control
        trained, control = cached[key]
        if trained is None or control is None:
            counts["not_aligned"] += 1
            continue
        if (trained["prefix_sha256"] != control["prefix_sha256"]
                or trained["forced_patch_closure"] != control["forced_patch_closure"]):
            raise ValueError("control model alignment differs from the trained models")
        for family in ("byte", "bpe"):
            for layer in LAYERS:
                arrays[family][layer].append(trained[family][layer])
                arrays[f"untrained_{family}"][layer].append(control[family][layer])
        kept.append(example)
        retained_indices.append(index)
        alignment.append({key: trained[key] for key in ("anchor_byte", "prefix_sha256", "bpe_tokens",
                                                         "byte_patches", "forced_patch_closure")})
    widths = {"byte": byte_model.config.trunk.width, "bpe": bpe_model.config.trunk.width,
              "untrained_byte": untrained_byte.config.trunk.width, "untrained_bpe": untrained_bpe.config.trunk.width}
    features = {name: {layer: np.stack(values) if values else np.empty((0, widths[name]), dtype=np.float32)
                       for layer, values in stages.items()} for name, stages in arrays.items()}
    return kept, features, {"examples_requested": len(examples), "examples_used": len(kept),
                            "skipped": counts, "retained_indices": retained_indices, "anchors": alignment,
                            "forced_patch_closure_count": sum(item["forced_patch_closure"] for item in alignment),
                            "forced_closure_is_analysis_only": True, "partial": counts["budget_exhausted"] > 0}


def cache_fingerprint(examples: list[dict], identities: dict, *, seed: int, forced: bool, max_prefix_bytes: int) -> str:
    encoded = json.dumps({"examples": examples, "identities": identities, "control_seed": seed,
                          "forced": forced, "max_prefix_bytes": max_prefix_bytes},
                         sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_features(path: str | Path, fingerprint: str, features: dict, alignment: dict) -> None:
    arrays = {f"{name}__{layer}": value for name, layers in features.items() for layer, value in layers.items()}
    arrays["metadata"] = np.asarray(json.dumps({"version": 1, "fingerprint": fingerprint, "alignment": alignment}))
    with Path(path).open("wb") as handle:
        np.savez_compressed(handle, **arrays)


def load_features(path: str | Path, fingerprint: str, examples: list[dict]) -> tuple[list[dict], dict, dict]:
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"].item()))
        if meta.get("version") != 1 or meta.get("fingerprint") != fingerprint:
            raise ValueError("state cache does not match the examples, checkpoints or extraction settings")
        alignment = meta["alignment"]
        retained = alignment["retained_indices"]
        if any(not isinstance(i, int) or i < 0 or i >= len(examples) for i in retained):
            raise ValueError("invalid state-cache example indices")
        kept = [examples[i] for i in retained]
        features = {name: {layer: archive[f"{name}__{layer}"].copy() for layer in LAYERS}
                    for name in ("byte", "bpe", "untrained_byte", "untrained_bpe")}
    if any(len(value) != len(kept) for layers in features.values() for value in layers.values()):
        raise ValueError("state cache has inconsistent array lengths")
    return kept, features, alignment


def _cosine_drift(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return None if denominator == 0 else float(1 - np.dot(left, right) / denominator)


def _probe_labels(text: str, examples: list[dict]) -> dict[str, tuple[str, str]]:
    labels: dict[str, tuple[str, str]] = {}
    numeric = re.search(r"[0-9]+", text)
    if numeric:
        value = int(numeric.group())
        labels["parity"] = ("even" if value % 2 == 0 else "odd", str(value))
        labels["magnitude"] = (magnitude(value), str(value))
    lexicon: dict[str, set[tuple[str, str]]] = {}
    for row in examples:
        if row["attribute"] == "suffix":
            lexicon.setdefault(row["word"], set()).add((row["label"], row["group"]))
    for word in reversed(re.findall(r"[A-Za-z]+", text)):
        matches = lexicon.get(word.casefold(), set())
        if len(matches) == 1:
            labels["suffix"] = next(iter(matches))
            break
    return labels


def perturbation_analysis(byte_model: ByteLatentLM, bpe_model: BPELM, tokenizer: Any, corpus: str,
                          examples: list[dict], fit_results: dict, fitted: dict, *, limit: int = 1000,
                          allow_forced_patch: bool = False, max_prefix_bytes: int = 512,
                          budget: ExtractionBudget | None = None) -> dict:
    """Use the repository's edit pairs and apply previously fitted probes unchanged."""
    token_bytes = TokenBytes(tokenizer, bpe_model.config.vocab_size)
    rows = []
    skipped = {"too_long": 0, "not_aligned": 0}
    budget_exhausted = False
    for pair in robustness_pairs(corpus, limit):
        original = pair["context"] + pair["original"]
        edited = pair["context"] + pair["edited"]
        offsets = [len(original.encode("utf-8")), len(edited.encode("utf-8"))]
        if max(offsets) > max_prefix_bytes:
            skipped["too_long"] += 1
            continue
        try:
            states = _budgeted(budget, lambda: [aligned_states(byte_model, bpe_model, tokenizer, text, anchor,
                                                                 allow_forced_patch=allow_forced_patch, token_bytes=token_bytes)
                                                for text, anchor in zip((original, edited), offsets)])
        except ExtractionBudgetExceeded:
            budget_exhausted = True
            break
        if any(state is None for state in states):
            skipped["not_aligned"] += 1
            continue
        first, second = states
        original_labels = _probe_labels(pair["original"], examples)
        edited_labels = _probe_labels(pair["edited"], examples)
        record: dict[str, Any] = {"kind": pair["kind"],
                                  "original_prefix_sha256": first["prefix_sha256"],
                                  "edited_prefix_sha256": second["prefix_sha256"],
                                  "forced_patch_closure": [first["forced_patch_closure"], second["forced_patch_closure"]]}
        for family, model, family_tokenizer in (("byte", byte_model, None), ("bpe", bpe_model, tokenizer)):
            device = next(model.parameters()).device
            try:
                original_nll, edited_nll = _budgeted(budget, lambda: (
                    continuation_nll(model, pair["context"], pair["original"], tokenizer=family_tokenizer, device=device),
                    continuation_nll(model, pair["context"], pair["edited"], tokenizer=family_tokenizer, device=device)))
            except ExtractionBudgetExceeded:
                budget_exhausted = True
                break
            evaluations = {}
            for attribute in ATTRIBUTES:
                old, new = original_labels.get(attribute), edited_labels.get(attribute)
                fit = fit_results.get(attribute, {})
                if old is None or new is None or (attribute, family) not in fitted or fit.get("status") != "run":
                    continue
                # Frozen probes are evaluated only on groups absent from their training set.
                if old[1] not in fit["test_groups"] or new[1] in fit["train_groups"]:
                    continue
                probe = fitted[(attribute, family)]
                prediction_old = str(probe.predict(first[family]["final"][None, :])[0])
                prediction_new = str(probe.predict(second[family]["final"][None, :])[0])
                evaluations[attribute] = {"original_label": old[0], "edited_label": new[0],
                                          "original_prediction": prediction_old, "edited_prediction": prediction_new}
            record[family] = {"state_cosine_drift": _cosine_drift(first[family]["final"], second[family]["final"]),
                              "original_nll": original_nll, "edited_nll": edited_nll,
                              "excess_nats_per_byte": edited_nll / len(pair["edited"].encode("utf-8"))
                              - original_nll / len(pair["original"].encode("utf-8")), "frozen_probes": evaluations}
        if budget_exhausted:
            break
        rows.append(record)
    groups = {}
    for kind in sorted({row["kind"] for row in rows}):
        cases = [row for row in rows if row["kind"] == kind]
        group: dict[str, Any] = {"count": len(cases)}
        for family in ("byte", "bpe"):
            drifts = [row[family]["state_cosine_drift"] for row in cases if row[family]["state_cosine_drift"] is not None]
            evaluation = {}
            for attribute in ATTRIBUTES:
                probes = [row[family]["frozen_probes"][attribute] for row in cases if attribute in row[family]["frozen_probes"]]
                changed = [p for p in probes if p["original_label"] != p["edited_label"]]
                evaluation[attribute] = {"count": len(probes),
                                         "original_accuracy": float(np.mean([p["original_prediction"] == p["original_label"] for p in probes])) if probes else None,
                                         "edited_accuracy": float(np.mean([p["edited_prediction"] == p["edited_label"] for p in probes])) if probes else None,
                                         "label_changed_count": len(changed),
                                         "change_detection_rate": float(np.mean([p["original_prediction"] != p["edited_prediction"] for p in changed])) if changed else None}
            group[family] = {"mean_excess_nats_per_byte": float(np.mean([row[family]["excess_nats_per_byte"] for row in cases])),
                             "mean_state_drift": float(np.mean(drifts)) if drifts else None, "frozen_probes": evaluation}
        groups[kind] = group
    return {"groups": groups, "pairs": rows, "skipped": skipped,
            "missing_edit_types": sorted({"extra_space", "newline", "casing", "digit"} - set(groups)),
            "alignment": "complete edited continuation; exact common prefix and BPE token end",
            "forced_closure_is_analysis_only": True, "partial": budget_exhausted}


def _gains(probes: dict) -> tuple[dict[str, float], list[str]]:
    gains, missing = {}, []
    for attribute in ATTRIBUTES:
        result = probes.get(attribute, {})
        value = result.get("models", {}).get("byte", {}).get("controlled_final_gain")
        if result.get("status") != "run" or not isinstance(value, (int, float)) or not np.isfinite(value):
            missing.append(attribute)
        else:
            gains[attribute] = float(value)
    return gains, missing


def probe_criteria(result: dict, replicate: dict | None = None) -> dict:
    """Evaluate registered claims, preserving 'not run' for absent evidence."""
    current_gains, missing = _gains(result.get("probes", {}))
    passed_attributes = {name for name, gain in current_gains.items() if gain >= .03 - 1e-12}
    accessibility = {"status": "not run", "reason": "a matching independently trained seed is required",
                     "controlled_gains": current_gains, "passing_attributes": sorted(passed_attributes),
                     "missing_attributes": missing}
    if len(passed_attributes) + len(missing) < 2:
        accessibility.update(status="fail", reason="fewer than two attributes can reach a 3-point controlled gain")
    elif replicate is not None:
        compatible = (replicate.get("type") == "representation_probes_v1"
                      and replicate.get("dataset_sha256") == result.get("dataset_sha256")
                      and not replicate.get("alignment", {}).get("partial", False)
                      and replicate.get("split_seed") == result.get("split_seed")
                      and replicate.get("test_fraction") == result.get("test_fraction")
                      and replicate.get("control_seed") == result.get("control_seed")
                      and replicate.get("model_names") == result.get("model_names")
                      and isinstance(replicate.get("training_seeds"), dict)
                      and isinstance(result.get("training_seeds"), dict)
                      and replicate["training_seeds"].get("byte") == replicate["training_seeds"].get("bpe")
                      and result["training_seeds"].get("byte") == result["training_seeds"].get("bpe")
                      and all(replicate["training_seeds"].get(k) is not None
                              and result["training_seeds"].get(k) is not None
                              and replicate["training_seeds"][k] != result["training_seeds"][k] for k in ("byte", "bpe"))
                      and all(replicate.get("checkpoints", {}).get(k, {}).get("sha256")
                              and result.get("checkpoints", {}).get(k, {}).get("sha256")
                              and replicate["checkpoints"][k]["sha256"] != result["checkpoints"][k]["sha256"]
                              for k in ("byte", "bpe")))
        if not compatible:
            accessibility["reason"] = "repeat must use the same aligned dataset and split with independent model seeds"
        else:
            other, other_missing = _gains(replicate.get("probes", {}))
            common = sorted(passed_attributes & {name for name, gain in other.items() if gain >= .03 - 1e-12})
            accessibility.update(repeat_controlled_gains=other, replicated_attributes=common,
                                 repeat_missing_attributes=other_missing)
            if len(common) >= 2:
                accessibility.update(status="pass", reason="at least two attributes have replicated ≥3-point controlled gains")
            elif not missing and not other_missing:
                accessibility.update(status="fail", reason="fewer than two attributes have replicated ≥3-point controlled gains")
            else:
                accessibility["reason"] = "some required attribute probes did not run"
    robustness = result.get("robustness", {})
    groups = robustness.get("groups", {})
    pairs = robustness.get("pairs", [])
    harmless = [row for row in pairs if row.get("kind") in {"extra_space", "newline", "casing"}]
    digit = [row for row in pairs if row.get("kind") == "digit"]
    robust = {"status": "not run", "reason": "all harmless edits, digit edits and frozen-probe sensitivity are required"}
    if all(kind in groups for kind in ("extra_space", "newline", "casing", "digit")) and harmless and digit:
        byte_excess = float(np.mean([row["byte"]["excess_nats_per_byte"] for row in harmless]))
        bpe_excess = float(np.mean([row["bpe"]["excess_nats_per_byte"] for row in harmless]))
        signals = {"byte": [], "bpe": []}
        for row in digit:
            for family in signals:
                for attribute in ("parity", "magnitude"):
                    probe = row[family]["frozen_probes"].get(attribute)
                    if probe and probe["original_label"] != probe["edited_label"]:
                        signals[family].append(probe["original_prediction"] != probe["edited_prediction"])
        byte_drift = groups["digit"]["byte"].get("mean_state_drift")
        robust.update(byte_harmless_excess_nats_per_byte=byte_excess, bpe_harmless_excess_nats_per_byte=bpe_excess,
                      byte_digit_drift=byte_drift, digit_label_changes={k: len(v) for k, v in signals.items()})
        if bpe_excess <= 0:
            robust["reason"] = "BPE harmless excess NLL is not positive, so a 20% reduction is not defined"
        elif not all(signals.values()) or byte_drift is None:
            robust["reason"] = "no held-out, label-changing digit edits were available to test frozen-probe sensitivity"
        else:
            byte_sensitivity, bpe_sensitivity = (float(np.mean(signals[key])) for key in ("byte", "bpe"))
            ratio = byte_excess / bpe_excess
            passed = (ratio <= .8 + 1e-12 and byte_drift > 0 and byte_sensitivity > 0
                      and byte_sensitivity >= bpe_sensitivity)
            robust.update(status="pass" if passed else "fail",
                          reason="requires ≥20% lower harmless excess NLL and nonzero digit sensitivity at least matching BPE",
                          excess_nll_ratio=ratio, byte_digit_sensitivity=byte_sensitivity,
                          bpe_digit_sensitivity=bpe_sensitivity)
    if result.get("alignment", {}).get("partial", False):
        accessibility.update(status="not run", reason="state extraction stopped at its time budget")
    if robustness.get("partial", False):
        robust.update(status="not run", reason="perturbation extraction stopped at its time budget")
    overall_status = ("fail" if "fail" in (accessibility["status"], robust["status"]) else
                      "pass" if accessibility["status"] == robust["status"] == "pass" else "not run")
    return {"accessibility": accessibility, "robustness": robust,
            "overall": {"status": overall_status, "reason": "both preregistered representation criteria must pass"}}
