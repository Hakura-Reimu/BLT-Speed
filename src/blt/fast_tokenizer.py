from __future__ import annotations

"""An exact, faster GPT-2 encoder for timed BPE preprocessing (GPT-6.1 Sol round 9, item 4a).

tiktoken is built from the local GPT-2 merges and vocabulary, so nothing is downloaded. With every special token
allowed it reproduced HF `tokenizers` token IDs on all 4,237 validation windows and 33 adversarial cases
(scripts/tokenizer_check.py; HF maps a literal "<|endoftext|>" in text to the special token, as in training).
"""

from pathlib import Path
from typing import Callable

import numpy as np

# The GPT-2 pre-tokenization pattern of tiktoken's `gpt2` encoding (tiktoken_ext/openai_public.py).
GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def gpt2_tiktoken(tokenizer_dir: str | Path):
    import tiktoken
    from tiktoken.load import data_gym_to_mergeable_bpe_ranks
    directory = Path(tokenizer_dir)
    ranks = data_gym_to_mergeable_bpe_ranks(vocab_bpe_file=str(directory / "merges.txt"),
                                            encoder_json_file=str(directory / "vocab.json"))
    return tiktoken.Encoding(name="gpt2-local", pat_str=GPT2_PATTERN, mergeable_ranks=ranks,
                             special_tokens={"<|endoftext|>": 50256}, explicit_n_vocab=50257)


def gpt2_numpy_encoder(tokenizer_dir: str | Path) -> Callable[[str], np.ndarray]:
    encoding = gpt2_tiktoken(tokenizer_dir)
    return lambda text: encoding.encode_to_numpy(text, allowed_special="all")


def _byte_level_decoder() -> dict[str, int]:
    """Inverse of the GPT-2 / HF ByteLevel byte-to-unicode table."""
    printable = [b for b in range(256) if chr(b).isprintable() and chr(b) != " "]
    table, extra = {chr(b): b for b in printable}, 0
    for b in range(256):
        if b not in printable:
            table[chr(256 + extra)] = b
            extra += 1
    return table


def hf_byte_level_tiktoken(tokenizer_json: str | Path):
    """tiktoken for an HF byte-level BPE trained with the GPT-2 regex (the 16K tokenizer).

    tiktoken merges by the rank of the concatenated bytes, which here is the HF token ID; this is exact only if IDs
    follow merge order, which is asserted. Exactness on text is checked separately by scripts/tokenizer_check.py.
    """
    import json
    import tiktoken
    spec = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))
    model, pre = spec["model"], spec["pre_tokenizer"]
    if (model["type"] != "BPE" or spec.get("normalizer") is not None or pre.get("type") != "ByteLevel"
            or not pre.get("use_regex") or pre.get("add_prefix_space")):
        raise ValueError("expected a byte-level BPE with the GPT-2 regex and no normalizer or prefix space")
    special = {token["content"]: token["id"] for token in spec["added_tokens"] if token["special"]}
    decode = _byte_level_decoder()
    ranks = {bytes(decode[ch] for ch in token): index for token, index in model["vocab"].items() if token not in special}
    merged = [model["vocab"]["".join(pair) if isinstance(pair, list) else pair.replace(" ", "")]
              for pair in model["merges"]]
    if merged != sorted(merged) or len(set(merged)) != len(merged) or len({k for k in ranks if len(k) == 1}) != 256:
        raise ValueError("token IDs do not follow merge order or the byte alphabet is incomplete")
    return tiktoken.Encoding(name=Path(tokenizer_json).parent.name, pat_str=GPT2_PATTERN, mergeable_ranks=ranks,
                             special_tokens=special, explicit_n_vocab=len(ranks) + len(special))


def hf_byte_level_numpy_encoder(tokenizer_json: str | Path) -> Callable[[str], np.ndarray]:
    encoding = hf_byte_level_tiktoken(tokenizer_json)
    return lambda text: encoding.encode_to_numpy(text, allowed_special="all")
