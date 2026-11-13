from __future__ import annotations

import json
import hashlib
from pathlib import Path
import random
import re
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .config import BPEConfig, ByteConfig
from .data import tokenize_window, utf8_end
from .eval_tokens import TokenBytes
from .model import BPELM, ByteLatentLM
from .score import autocast_for


@torch.inference_mode()
def continuation_nlls(model: torch.nn.Module, context: str, continuations: list[str], *, tokenizer: Any = None,
                      device: str | torch.device = "cpu") -> list[float]:
    """Score a forced token boundary so every continuation byte is accounted for."""
    config = model.config
    device = torch.device(device)
    prefix, suffixes = context.encode("utf-8"), [value.encode("utf-8") for value in continuations]
    if not suffixes or any(not suffix for suffix in suffixes):
        raise ValueError("continuation must be nonempty")
    model.eval()
    if isinstance(config, ByteConfig):
        suffix_ids = [list(suffix) for suffix in suffixes]
        offset = len(prefix)
        sequences = [list(prefix) + ids for ids in suffix_ids]
        padding = 0
    elif isinstance(config, BPEConfig):
        if tokenizer is None:
            raise ValueError("BPE continuation scoring requires the tokenizer")
        prefix_ids = tokenize_window(tokenizer, prefix)
        suffix_ids = [tokenize_window(tokenizer, suffix) for suffix in suffixes]
        sequences = [[config.bos_id, *prefix_ids, *ids[:-1]] for ids in suffix_ids]
        offset = len(prefix_ids)
        padding = config.bos_id
    else:
        raise ValueError("unknown model family")
    max_length, max_suffix = max(map(len, sequences)), max(map(len, suffix_ids))
    inputs = torch.full((len(sequences), max_length), padding, dtype=torch.long, device=device)
    targets = torch.full((len(sequences), max_suffix), -100, dtype=torch.long, device=device)
    for row, (sequence, ids) in enumerate(zip(sequences, suffix_ids)):
        inputs[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        targets[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    with autocast_for(device):
        if isinstance(model, BPELM):
            hidden, _ = model.trunk(model.embedding(inputs))
            logits = F.linear(hidden[:, offset:offset + max_suffix], model.embedding.weight)
        elif isinstance(model, ByteLatentLM):
            hidden = model.hidden(inputs)
            logits = F.linear(hidden[:, offset:offset + max_suffix], model.output_weight)
        else:
            logits = model(inputs)[:, offset:offset + max_suffix]
        # Flatten rather than pass (batch, classes, positions): on the CPU, cross-entropy over the strided class
        # dimension summed 50K exponentials less accurately, about 1e-4 nats low per BPE token (Sol round 16 check).
        logits = logits.float()
        losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100,
                                 reduction="none")
        return losses.view(targets.shape).sum(dim=1).cpu().tolist()


def continuation_nll(model: torch.nn.Module, context: str, continuation: str, *, tokenizer: Any = None,
                     device: str | torch.device = "cpu") -> float:
    return continuation_nlls(model, context, [continuation], tokenizer=tokenizer, device=device)[0]


def hellaswag_preprocess(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    return re.sub(r"\[.*?\]", "", text).replace("  ", " ")


def format_candidates(task: str, record: dict[str, Any]) -> tuple[str, list[str], int]:
    if task == "piqa":
        context = f"Question: {record['goal']}\nAnswer:"
        candidates = [" " + str(record["sol1"]), " " + str(record["sol2"])]
    elif task == "hellaswag":
        context = hellaswag_preprocess(record["activity_label"] + ": " + record["ctx_a"] + " " + record["ctx_b"].capitalize())
        candidates = [" " + hellaswag_preprocess(choice) for choice in record["endings"]]
    else:
        raise ValueError("task must be piqa or hellaswag")
    label = int(record["label"])
    if not 0 <= label < len(candidates):
        raise ValueError("invalid answer label")
    return context, candidates, label


def load_local_examples(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_dir() and ((path / "dataset_dict.json").exists() or (path / "state.json").exists()):
        from datasets import DatasetDict, load_from_disk
        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            if "validation" not in dataset:
                raise ValueError("saved dataset has no validation split")
            dataset = dataset["validation"]
        return [dict(row) for row in dataset]
    files = [path] if path.is_file() else sorted([*path.rglob("*.parquet"), *path.rglob("*.jsonl")])
    if not files:
        raise ValueError("no local parquet or JSONL files found")
    if path.is_dir():
        validation = [file for file in files if "validation" in str(file).lower() or file.stem.lower().startswith("val")]
        if validation:
            files = validation
        elif any("train" in file.name.lower() or "test" in file.name.lower() for file in files):
            raise ValueError("ambiguous dataset directory; supply its validation files")
    records = []
    for file in files:
        if file.suffix == ".jsonl":
            with open(file, encoding="utf-8") as handle:
                records.extend(json.loads(line) for line in handle if line.strip())
        elif file.suffix == ".parquet":
            import pyarrow.parquet as pq
            records.extend(pq.read_table(file).to_pylist())
        else:
            raise ValueError(f"unsupported local dataset file: {file}")
    return records


def evaluate_candidates(model: torch.nn.Module, records: list[dict[str, Any]], task: str, *, tokenizer: Any = None,
                        device: str | torch.device = "cpu", limit: int = 1000, seed: int = 0) -> dict[str, Any]:
    if limit <= 0 or not records:
        raise ValueError("a positive limit and nonempty dataset are required")
    selected = sorted(random.Random(seed).sample(range(len(records)), min(limit, len(records))))
    examples = []
    for index in selected:
        context, candidates, label = format_candidates(task, records[index])
        nll = continuation_nlls(model, context, candidates, tokenizer=tokenizer, device=device)
        normalised = [value / len(candidate.encode("utf-8")) for value, candidate in zip(nll, candidates)]
        examples.append({"index": index, "label": label, "context": context, "candidates": candidates,
                         "nll": nll, "nll_per_byte": normalised,
                         "prediction": min(range(len(nll)), key=nll.__getitem__),
                         "prediction_byte_normalised": min(range(len(nll)), key=normalised.__getitem__)})
    return {"task": task, "seed": seed, "count": len(examples),
            "accuracy": sum(row["prediction"] == row["label"] for row in examples) / len(examples),
            "accuracy_byte_normalised": sum(row["prediction_byte_normalised"] == row["label"] for row in examples) / len(examples),
            "examples": examples, "bpe_boundary": "context and continuation encoded separately; no token crosses the boundary"}


def retrieval_cases(filler: bytes, context_bytes: int, depth: float, trials: int, seed: int) -> list[dict[str, Any]]:
    filler.decode("utf-8")
    if not filler or context_bytes < 128 or not 0 <= depth <= 1 or trials <= 0:
        raise ValueError("invalid retrieval case settings")
    source = filler * (context_bytes // len(filler) + 2)
    rng = random.Random(seed)
    cases = []
    for _ in range(trials):
        key = f"{rng.randrange(100000):05d}"
        marker = f" The pass key is {key}. ".encode("ascii")
        insertion = min(int(context_bytes * depth), context_bytes - len(marker))
        while insertion and source[insertion] & 0xC0 == 0x80:
            insertion -= 1
        remaining = context_bytes - insertion - len(marker)
        if remaining:
            try:
                stop = utf8_end(source, insertion, remaining)
            except ValueError:
                stop = insertion
        else:
            stop = insertion
        raw = source[:insertion] + marker + source[insertion:stop]
        raw += b" " * (context_bytes - len(raw))
        distractors = set()
        while len(distractors) < 9:
            candidate = f"{rng.randrange(100000):05d}"
            if candidate != key:
                distractors.add(candidate)
        cases.append({"context": raw.decode("utf-8"), "query": "\nWhat is the pass key? The pass key is",
                      "key": key, "distractors": sorted(distractors), "insertion_byte": insertion + 1,
                      "context_bytes": context_bytes, "depth": depth})
    return cases


@torch.inference_mode()
def greedy_bytes(model: torch.nn.Module, context: str, count: int, *, tokenizer: Any = None,
                 device: str | torch.device = "cpu", vocabulary: TokenBytes | None = None) -> bytes:
    config, device = model.config, torch.device(device)
    raw = context.encode("utf-8")
    model.eval()
    with autocast_for(device):
        if isinstance(config, ByteConfig):
            state = model.init_state(1, len(raw) + count, device, torch.bfloat16 if device.type == "cuda" else torch.float32)
            logits = model.prefill(state, torch.tensor(list(raw), dtype=torch.long, device=device)[None, :])
            vocabulary = None
        elif isinstance(config, BPEConfig):
            ids = [config.bos_id, *tokenize_window(tokenizer, raw)]
            state = model.init_state(1, len(ids) + count, device, torch.bfloat16 if device.type == "cuda" else torch.float32)
            logits = model.prefill(state, torch.tensor(ids, device=device)[None, :])
            vocabulary = vocabulary or TokenBytes(tokenizer, config.vocab_size)
        else:
            raise ValueError("unknown model family")
        output = bytearray()
        while len(output) < count:
            token = int(logits[0, -1].argmax().item())
            if vocabulary is None:
                if token >= 256:
                    break
                piece = bytes([token])
            else:
                piece = vocabulary.values[token]
                if piece is None:
                    break
            output.extend(piece)
            if len(output) < count:
                logits = model.step(state, torch.tensor([token], device=device))
        return bytes(output)


def evaluate_retrieval(model: torch.nn.Module, cases: Iterable[dict[str, Any]], *, tokenizer: Any = None,
                       device: str | torch.device = "cpu") -> dict[str, Any]:
    results = []
    vocabulary = TokenBytes(tokenizer, model.config.vocab_size) if isinstance(model.config, BPEConfig) else None
    for case in cases:
        prompt = case["context"] + case["query"] + " "
        candidates = [case["key"], *case["distractors"]]
        nll = continuation_nlls(model, prompt, candidates, tokenizer=tokenizer, device=device)
        rank = 1 + sum(value < nll[0] for value in nll[1:]) + 0.5 * sum(value == nll[0] for value in nll[1:])
        generated = greedy_bytes(model, prompt, 5, tokenizer=tokenizer, device=device, vocabulary=vocabulary)
        results.append({"key": case["key"], "distractors": case["distractors"],
                        "context_sha256": hashlib.sha256(case["context"].encode("utf-8")).hexdigest(),
                        "insertion_byte": case["insertion_byte"], "depth": case["depth"],
                        "context_bytes": case["context_bytes"], "greedy_first_five": generated[:5].decode("ascii", errors="replace"),
                        "greedy_exact": generated[:5] == case["key"].encode("ascii"), "generated_hex": generated.hex(),
                        "overshoot_bytes": max(0, len(generated) - 5),
                        "true_key_rank": rank, "candidate_nll": nll})
    if not results:
        raise ValueError("no retrieval cases")
    return {"count": len(results), "greedy_accuracy": sum(row["greedy_exact"] for row in results) / len(results),
            "mean_true_key_rank": sum(row["true_key_rank"] for row in results) / len(results), "trials": results}


def robustness_pairs(text: str, limit: int = 1000) -> list[dict[str, str]]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    pairs = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        middle = sentence.find(" ", len(sentence) // 3)
        if middle <= 0 or middle + 2 >= len(sentence):
            continue
        context, continuation = sentence[:middle], sentence[middle:]
        edits = [("extra_space", continuation.replace(" ", "  ", 1)),
                 ("newline", continuation.replace(" ", "\n", 1)),
                 ("casing", re.sub(r"[A-Za-z]+", lambda match: match.group().swapcase(), continuation, count=1))]
        digit = re.search(r"[0-9]", continuation)
        if digit:
            changed = str((int(digit.group()) + 1) % 10)
            edits.append(("digit", continuation[:digit.start()] + changed + continuation[digit.end():]))
        for kind, edited in edits:
            if edited != continuation:
                pairs.append({"context": context, "original": continuation, "edited": edited, "kind": kind})
                if len(pairs) >= limit:
                    return pairs
    return pairs


@torch.inference_mode()
def states_at_aligned_bytes(model: torch.nn.Module, texts: list[str], offsets: list[int], *, tokenizer: Any = None,
                            device: str | torch.device = "cpu") -> list[dict[str, torch.Tensor]]:
    if len(texts) != len(offsets):
        raise ValueError("each text needs a byte offset")
    device = torch.device(device)
    model.eval()
    result = []
    for text, offset in zip(texts, offsets):
        raw = text.encode("utf-8")
        if not 0 <= offset <= len(raw):
            raise ValueError("byte offset outside text")
        prefix = raw[:offset]
        prefix.decode("utf-8")
        with autocast_for(device):
            if isinstance(model, ByteLatentLM):
                inputs = torch.tensor(list(prefix) + [0], device=device)[None, :]
                final, info = model._full_hidden(inputs)
                patch_id = int(info["patches"].patch_ids[0, offset])
                global_state = info["global_states"][0, patch_id]
                final_state = final[0, offset]
            elif isinstance(model, BPELM):
                ids = [model.config.bos_id, *tokenize_window(tokenizer, prefix)]
                tokens = torch.tensor(ids, device=device)[None, :]
                hidden, _ = model.trunk(model.embedding(tokens))
                global_state = final_state = hidden[0, -1]
            else:
                raise ValueError("state hook requires a Track B model")
        result.append({"global": global_state.float().cpu(), "final": final_state.float().cpu()})
    return result


def evaluate_robustness(model: torch.nn.Module, pairs: Iterable[dict[str, str]], *, tokenizer: Any = None,
                        device: str | torch.device = "cpu") -> dict[str, Any]:
    rows = []
    for pair in pairs:
        original = continuation_nll(model, pair["context"], pair["original"], tokenizer=tokenizer, device=device)
        edited = continuation_nll(model, pair["context"], pair["edited"], tokenizer=tokenizer, device=device)
        texts = [pair["context"] + pair["original"], pair["context"] + pair["edited"]]
        states = states_at_aligned_bytes(model, texts, [len(text.encode("utf-8")) for text in texts],
                                         tokenizer=tokenizer, device=device)
        drift = {name: float(1 - F.cosine_similarity(states[0][name][None], states[1][name][None]).item())
                 for name in ("global", "final")}
        rows.append({"kind": pair["kind"], "context": pair["context"], "original": pair["original"], "edited": pair["edited"],
                     "original_nll": original, "edited_nll": edited, "excess_nats": edited - original,
                     "excess_nats_per_byte": edited / len(pair["edited"].encode("utf-8")) - original / len(pair["original"].encode("utf-8")),
                     "state_cosine_drift": drift})
    groups = {}
    if not rows:
        raise ValueError("no usable robustness pairs were found")
    for kind in sorted({row["kind"] for row in rows}):
        values = [row for row in rows if row["kind"] == kind]
        groups[kind] = {"count": len(values), "mean_excess_nats": float(np.mean([row["excess_nats"] for row in values])),
                        "mean_excess_nats_per_byte": float(np.mean([row["excess_nats_per_byte"] for row in values])),
                        "mean_global_drift": float(np.mean([row["state_cosine_drift"]["global"] for row in values])),
                        "mean_final_drift": float(np.mean([row["state_cosine_drift"]["final"] for row in values]))}
    return {"groups": groups, "pairs": rows, "missing_edit_types": sorted({"extra_space", "newline", "casing", "digit"} - set(groups)),
            "alignment": "end of continuation; BPE is re-tokenized at the anchor"}
