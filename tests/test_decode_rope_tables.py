from contextlib import nullcontext

import pytest
import torch

import blt.decode as decode
from blt.config import TransformerConfig
from blt.model import Transformer, rope


def transformer():
    return Transformer(TransformerConfig(2, 8, 2, 1, 4, 16, max_positions=8)).eval()


def cache_for(module, dtype):
    cache = decode.StaticCache.from_reference(module.cache(2, 8, "cpu", dtype))
    for tensor in cache.tensors():
        tensor.zero_()
    return cache


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("chunk", [False, True])
def test_static_decode_uses_one_table_lookup_and_matches_formula(monkeypatch, bf16, chunk):
    torch.manual_seed(20)
    module = transformer()
    cache = cache_for(module, torch.bfloat16 if bf16 else torch.float32)
    starts = torch.tensor([0, 5] if chunk else [0, 7])
    length = 3 if chunk else 1
    positions = starts[:, None] + torch.arange(length)[None, :]
    values = torch.randn(2, length, 8)
    table_dtypes, rotations = [], []
    tables, rotate = module.rotary.tables, decode._rotate

    def record_tables(dtype):
        table_dtypes.append(dtype)
        return tables(dtype)

    def check_rotation(x, cos, sin):
        actual = rotate(x, cos, sin)
        torch.testing.assert_close(actual, rope(x, positions, module.config.rope_theta), rtol=0, atol=0)
        rotations.append(x.dtype)
        return actual

    monkeypatch.setattr(module.rotary, "tables", record_tables)
    monkeypatch.setattr(decode, "_rotate", check_rotation)
    context = torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else nullcontext()
    with torch.no_grad(), context:
        if chunk:
            decode._transformer_chunk(module, values, cache, starts)
        else:
            decode._transformer_step(module, values, cache, starts)
    expected_dtype = torch.bfloat16 if bf16 else torch.float32
    assert table_dtypes == [expected_dtype]
    assert rotations == [expected_dtype] * (2 * len(module.blocks))


def test_full_cache_masked_rows_and_ragged_suffixes_stay_in_bounds():
    torch.manual_seed(21)
    module = transformer()
    cache = cache_for(module, torch.float32)
    old = [tensor[0].clone() for tensor in cache.tensors()]
    with torch.no_grad():
        step = decode._transformer_step(module, torch.randn(2, 1, 8), cache,
                                        torch.tensor([8, 3]), torch.tensor([False, True]))
    assert torch.isfinite(step).all()
    assert all(torch.equal(before, after[0]) for before, after in zip(old, cache.tensors()))

    cache = cache_for(module, torch.float32)
    old = [tensor[0].clone() for tensor in cache.tensors()]
    with torch.no_grad():
        chunk = decode._transformer_chunk(module, torch.randn(2, 3, 8), cache,
                                         torch.tensor([8, 7]), valid_counts=[0, 1])
    assert torch.isfinite(chunk).all()
    assert all(torch.equal(before, after[0]) for before, after in zip(old, cache.tensors()))
