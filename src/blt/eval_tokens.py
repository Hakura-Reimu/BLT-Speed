from __future__ import annotations

from typing import Any

import torch


def _byte_decoder() -> dict[str, int]:
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = values.copy()
    extra = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            chars.append(256 + extra)
            extra += 1
    return {chr(char): value for value, char in zip(values, chars)}


class TokenBytes:
    """Recover the raw bytes of GPT-2-style ByteLevel tokens, including partial UTF-8."""
    def __init__(self, tokenizer: Any, vocab_size: int) -> None:
        decoder = _byte_decoder()
        added = tokenizer.get_added_tokens_decoder() if hasattr(tokenizer, "get_added_tokens_decoder") else {}
        special = {int(index) for index, token in added.items() if getattr(token, "special", False)}
        self.values: list[bytes | None] = []
        for index in range(vocab_size):
            text = tokenizer.id_to_token(index)
            if index in special or text is None or not text or any(char not in decoder for char in text):
                self.values.append(None)
            else:
                self.values.append(bytes(decoder[char] for char in text))
        if not any(self.values):
            raise ValueError("tokenizer has no GPT-2 ByteLevel output tokens")
        self._tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = str(device)
        if key not in self._tensors:
            first = torch.tensor([value[0] if value else 256 for value in self.values], device=device, dtype=torch.long)
            allowed = torch.tensor([value is not None for value in self.values], device=device, dtype=torch.bool)
            self._tensors[key] = (first, allowed)
        return self._tensors[key]


def first_byte_probabilities(logits: torch.Tensor, first_bytes: torch.Tensor) -> torch.Tensor:
    probabilities = logits.float().softmax(dim=-1)
    result = probabilities.new_zeros((*probabilities.shape[:-1], 257))
    indices = first_bytes.expand(*probabilities.shape[:-1], -1)
    return result.scatter_add(-1, indices, probabilities)
