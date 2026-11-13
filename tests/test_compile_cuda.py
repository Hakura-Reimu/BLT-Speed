import copy

import pytest
import torch
import torch.nn.functional as F

from blt.config import BPEConfig, ByteConfig, TransformerConfig
from blt.model import BPELM, ByteLatentLM
from blt.patcher import max_patch_count, patch_batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 compilation requires a CUDA device")
@pytest.mark.parametrize("family", ["bpe", "byte"])
def test_compiled_training_step_matches_eager_bf16_and_does_not_recompile(family):
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    torch.manual_seed(37)
    trunk = TransformerConfig(1, 16, 2, 1, 8, 32, max_positions=128)
    if family == "bpe":
        model = BPELM(BPEConfig("tiny", trunk, 64, 0))
        inputs = {"tokens": torch.randint(0, 64, (2, 64), device="cuda")}
        targets = torch.randint(0, 64, (2, 64), device="cuda")
    else:
        config = ByteConfig("tiny", trunk, 16, 2, 32, 3, 6, max_len=7, hash_rows=32)
        model = ByteLatentLM(config)
        raw = torch.randint(0, 256, (2, 64), dtype=torch.uint8)
        patches = patch_batch(raw.numpy(), 3, 7, 64, max_patches=max_patch_count(64, 3))
        inputs = {"bytes": raw.cuda(), "patches": {"ends": torch.from_numpy(patches.ends).cuda(),
                  "patch_ids": torch.from_numpy(patches.patch_ids).cuda(), "counts": torch.from_numpy(patches.counts).cuda(),
                  "boundaries": torch.from_numpy(patches.boundaries).cuda()}}
        targets = raw.long().cuda()
    targets[:, -3:] = -100
    eager = model.cuda().train()
    compiled_base = copy.deepcopy(eager)
    compiled = torch.compile(compiled_base, fullgraph=True, dynamic=False)
    opt_eager = torch.optim.AdamW(eager.parameters(), lr=1e-3, fused=True)
    opt_compiled = torch.optim.AdamW(compiled_base.parameters(), lr=1e-3, fused=True)
    def step(net, optimizer):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(inputs)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        loss.backward()
        optimizer.step()
        return loss.detach().item()
    eager_loss = step(eager, opt_eager)
    compiled_loss = step(compiled, opt_compiled)
    assert compiled_loss == pytest.approx(eager_loss, rel=0.01, abs=0.01)
    graphs = counters["stats"].get("unique_graphs", 0)
    if family == "bpe":
        inputs = {"tokens": torch.randint(0, 64, (2, 64), device="cuda")}
    else:
        raw = torch.full((2, 64), ord(" "), dtype=torch.uint8)
        patches = patch_batch(raw.numpy(), 3, 7, 64, max_patches=max_patch_count(64, 3))
        inputs = {"bytes": raw.cuda(), "patches": {"ends": torch.from_numpy(patches.ends).cuda(),
                  "patch_ids": torch.from_numpy(patches.patch_ids).cuda(), "counts": torch.from_numpy(patches.counts).cuda(),
                  "boundaries": torch.from_numpy(patches.boundaries).cuda()}}
    step(compiled, opt_compiled)
    assert counters["stats"].get("unique_graphs", 0) == graphs
