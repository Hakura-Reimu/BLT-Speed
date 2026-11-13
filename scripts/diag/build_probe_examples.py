"""Build the annotated examples for the representation probes (plan section 4, "internal representations").

Source: the validation split only (the test split stays closed). No model is loaded. Annotation is rule-based and
recorded in the manifest:
  numeric  an integer of 1-6 digits (no leading zero), preceded by a space, "(", "$" or "#", and followed by one of
           , . ; : ! ? ) and then whitespace or the end of the document. Decimals and digit groups ("1,000") are
           excluded by these two conditions. Labels (parity, magnitude) are derived by blt.probe.prepare_examples.
  suffix   an all-lowercase word of at least 4 letters, preceded by a space and followed by the same punctuation
           rule, ending in ing, ed, ly or s (longest first). The lemma is the first candidate stem (standard English
           spelling rules: drop the suffix, restore a final e, undouble a final consonant, y for i) that occurs at
           least 3 times as a word in the validation split; words with no such stem, and a short stoplist of known
           false derivations, are left out. Lemmas only define the disjoint train/test groups.
Every kept example is aligned without forcing: the anchor (just after the punctuation) is a natural patch end of
the byte models' causal patcher (min_len, max_len, block_size of M-P6) and an independent BPE token end of the 16K
tokenizer (blt.probe._token_prefix). The text starts at a whitespace boundary at most --context bytes before the
anchor (or at the document start) and keeps up to 64 bytes after it, which are never fed to a model.
Quotas per class, caps per value or form, lemma and document, and a fixed seed keep the classes balanced and the
groups many. Validation-split documents that supply no example give the robustness text.

Usage: PYTHONPATH=src python scripts/diag/build_probe_examples.py --supply ~/blt/data/supply-v1 \
  --byte-config ~/blt/runs/m-p6-seed0/config.json --tokenizer ~/blt/data/tokenizer16k-v1/tokenizer.json \
  --output ~/blt/data/probe-v1
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from blt.eval_tokens import TokenBytes
from blt.patcher import patch_batch
from blt.probe import _token_prefix, magnitude, prepare_examples

AFTER = rb"(?=[,.;:!?)](?:\s|$))"
NUMBER = re.compile(rb"(?<=[ (\$#])(0|[1-9][0-9]{0,5})" + AFTER)
WORD = re.compile(rb"(?<= )([a-z]{4,})" + AFTER)
VOCAB_WORD = re.compile(rb"\b[A-Za-z]+\b")
SUFFIX_ORDER = ("ing", "ed", "ly", "s")
STOPLIST = {
    "evening", "morning", "during", "ceiling", "nothing", "something", "anything", "everything", "thing", "things",
    "string", "spring", "bring", "king", "ring", "wing", "sing", "early", "only", "family", "reply", "apply", "supply",
    "july", "italy", "holy", "ugly", "fly", "feed", "need", "seed", "speed", "indeed", "hundred", "news", "series",
    "species", "means", "lens", "always", "perhaps", "towards", "afterwards", "whereas", "sometimes", "nevertheless",
    "bless", "less", "unless", "this", "thus", "plus", "bonus", "status", "focus", "campus", "virus", "census",
    "analysis", "basis", "crisis", "thesis", "emphasis", "diagnosis", "physics", "mathematics", "economics",
    "politics", "ethics", "statistics", "genetics", "athletics", "electronics", "united", "naked", "wicked", "sacred",
    "beloved", "rugged", "jagged", "ragged", "kindred", "rally", "belly", "jelly", "bully", "ally", "tally", "billy",
    "lily", "silly", "hilly", "jolly", "folly", "assembly", "anomaly", "italy", "monopoly",
}
MAGNITUDES = ("0-9", "10-99", "100-999", "1000+")


def lemma_for(word: str, suffix: str, vocab: collections.Counter) -> str | None:
    w = word
    if suffix == "ing":
        stem = w[:-3]
        candidates = [stem, stem + "e"]
        if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] not in "aeiouy":
            candidates.append(stem[:-1])
    elif suffix == "ed":
        candidates = [w[:-2], w[:-1]]
        if len(w) >= 5 and w[-3] == w[-4] and w[-3] not in "aeiouy":
            candidates.append(w[:-3])
        if w.endswith("ied"):
            candidates.append(w[:-3] + "y")
    elif suffix == "ly":
        candidates = [w[:-2], w[:-1] + "e"]
        if w.endswith("ily"):
            candidates.append(w[:-3] + "y")
        if w.endswith("ally"):
            candidates.append(w[:-4])
    else:
        if w.endswith(("ss", "us", "is")):
            return None
        candidates = [w[:-1]]
        if w.endswith("es"):
            candidates.append(w[:-2])
        if w.endswith("ies"):
            candidates.append(w[:-3] + "y")
    for candidate in candidates:
        if len(candidate) >= 3 and candidate != w and vocab[candidate] >= 3:
            return candidate
    return None


def suffix_of(word: str) -> str | None:
    return next((s for s in SUFFIX_ORDER if word.endswith(s) and len(word) - len(s) >= 3), None)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--supply", type=Path, required=True)
    parser.add_argument("--byte-config", type=Path, required=True, help="config.json of the byte run (patcher)")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-class", type=int, default=400)
    parser.add_argument("--per-value", type=int, default=6)
    parser.add_argument("--per-form", type=int, default=2)
    parser.add_argument("--per-lemma", type=int, default=4)
    parser.add_argument("--per-document", type=int, default=3)
    parser.add_argument("--context", type=int, default=448, help="maximum prefix bytes (probe default limit is 512)")
    parser.add_argument("--robustness-bytes", type=int, default=400_000)
    args = parser.parse_args()

    model = json.loads(args.byte_config.read_text())["training"]["model"]
    if model.get("family") != "byte":
        parser.error("--byte-config must be a byte run")
    min_len, max_len, block = model["min_len"], model["max_len"], model["block_size"]
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    token_bytes = TokenBytes(tokenizer, tokenizer.get_vocab_size())

    data = (args.supply / "val.bin").read_bytes()
    documents = [json.loads(line) for line in open(args.supply / "val.idx", encoding="utf-8")]
    vocab = collections.Counter(m.group().decode().lower() for m in VOCAB_WORD.finditer(data))

    order = list(range(len(documents)))
    random.Random(args.seed).shuffle(order)
    quota = {**{("numeric", m): args.per_class for m in MAGNITUDES},
             **{("suffix", s): args.per_class for s in SUFFIX_ORDER}}
    per_group, per_form, per_doc = collections.Counter(), collections.Counter(), collections.Counter()
    records, used_docs = [], set()
    rejected = collections.Counter()

    def aligned(doc: dict, start_abs: int, end_abs: int) -> tuple[str, int, int] | None:
        doc_start, doc_end = doc["start"], doc["end"]
        anchor_abs = end_abs + 1  # just after the punctuation byte
        lo = max(doc_start, anchor_abs - args.context)
        if lo > doc_start:
            space = data.find(b" ", lo, start_abs)
            if space < 0:
                return None
            lo = space + 1
        hi = min(doc_end, anchor_abs + 64)
        while anchor_abs < hi < doc_end and data[hi] & 0xC0 == 0x80:
            hi -= 1
        raw = data[lo:hi]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            rejected["utf8"] += 1
            return None
        anchor = anchor_abs - lo
        prefix = raw[:anchor]
        patches = patch_batch(np.frombuffer(prefix, dtype=np.uint8)[None, :], min_len, max_len, block)
        if not bool(patches.boundaries[0, -1]):
            rejected["no_natural_patch_end"] += 1
            return None
        if _token_prefix(tokenizer, token_bytes, text, anchor) is None:
            rejected["no_bpe_token_end"] += 1
            return None
        return text, anchor, end_abs - lo

    for doc_index in order:
        if all(v <= 0 for v in quota.values()):
            break
        doc = documents[doc_index]
        body = data[doc["start"]:doc["end"]]
        candidates = []
        for match in NUMBER.finditer(body):
            value = int(match.group())
            candidates.append(("numeric", magnitude(value), str(value), None, match))
        for match in WORD.finditer(body):
            word = match.group().decode()
            suffix = suffix_of(word)
            if suffix is None or word in STOPLIST:
                continue
            lemma = lemma_for(word, suffix, vocab)
            if lemma is None:
                rejected["no_lemma"] += 1
                continue
            candidates.append(("suffix", suffix, lemma, word, match))
        random.Random(f"{args.seed}:{doc['sha256']}").shuffle(candidates)
        for kind, label, group, word, match in candidates:
            if per_doc[doc_index] >= args.per_document or quota[(kind, label)] <= 0:
                continue
            if kind == "numeric" and per_group[("numeric", group)] >= args.per_value:
                continue
            if kind == "suffix" and (per_group[("suffix", group)] >= args.per_lemma or per_form[word] >= args.per_form):
                continue
            result = aligned(doc, doc["start"] + match.start(), doc["start"] + match.end())
            if result is None:
                continue
            text, anchor, target_end = result
            record = {"text": text, "anchor_byte": anchor, "target_end_byte": target_end,
                      "attribute": kind, "source": {"split": "val", "document_sha256": doc["sha256"],
                                                    "target_offset_in_document": match.start()}}
            if kind == "numeric":
                record["value"] = int(group)
            else:
                record.update({"suffix": label, "lemma": group})
                per_form[word] += 1
            per_group[(kind, group)] += 1
            per_doc[doc_index] += 1
            quota[(kind, label)] -= 1
            records.append(record)
            used_docs.add(doc_index)

    examples = prepare_examples(records)  # the probe tool's own validation
    args.output.mkdir(parents=True, exist_ok=True)
    examples_path = args.output / "examples.jsonl"
    with open(examples_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    robust, size = [], 0
    for doc_index in order[::-1]:
        if size >= args.robustness_bytes:
            break
        if doc_index in used_docs:
            continue
        doc = documents[doc_index]
        text = data[doc["start"]:doc["end"]].decode("utf-8")
        robust.append(text)
        size += len(text.encode("utf-8"))
    robustness_path = args.output / "robustness.txt"
    robustness_path.write_text("\n".join(robust), encoding="utf-8")

    classes = collections.Counter((e["attribute"], e["label"]) for e in examples)
    groups = collections.defaultdict(set)
    for e in examples:
        groups[e["attribute"]].add(e["group"])
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        "type": "probe_examples_v1", "source_split": "val",
        "supply_manifest_sha256": sha256(args.supply / "manifest.json"), "val_bin_sha256": sha256(args.supply / "val.bin"),
        "tokenizer_sha256": sha256(args.tokenizer), "byte_config": str(args.byte_config),
        "patcher": {"min_len": min_len, "max_len": max_len, "block_size": block}, "forced_patch_closures": 0,
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "records": len(records), "examples": len(examples),
        "classes": {f"{a}:{l}": n for (a, l), n in sorted(classes.items())},
        "distinct_groups": {a: len(g) for a, g in groups.items()},
        "documents_used": len(used_docs), "unfilled_quota": {f"{k[0]}:{k[1]}": v for k, v in quota.items() if v > 0},
        "rejected": dict(rejected),
        "robustness_text": {"documents": len(robust), "bytes": size, "disjoint_from_example_documents": True},
        "files": {"examples.jsonl": sha256(examples_path), "robustness.txt": sha256(robustness_path)},
        "builder_commit": commit,
        "annotation": "rule-based (see the builder docstring); lemmas define groups only",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ("records", "examples", "classes", "distinct_groups", "unfilled_quota",
                                               "rejected", "documents_used")}, indent=1))


if __name__ == "__main__":
    main()
