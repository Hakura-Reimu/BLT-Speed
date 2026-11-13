import torch

from blt.config import BPEConfig, TransformerConfig
from blt.model import BPELM


def test_ragged_bpe_prefill_clears_garbage_cache_slots():
    """A right-padded BPE prefill must not leave uninitialised (here NaN) slots that masked attention reads."""
    torch.manual_seed(3)
    model = BPELM(BPEConfig("tiny", TransformerConfig(2, 16, 2, 1, 8, 32, max_positions=128), 32, 0)).eval()
    prompt = torch.tensor([[0, 1, 2, 0, 0, 0], [0, 3, 4, 5, 6, 7]])
    lengths = [3, 6]
    with torch.inference_mode():
        state = model.init_state(2, 40, "cpu", torch.float32)
        for key, value in zip(state.cache.keys, state.cache.values):
            key.fill_(float("nan"))
            value.fill_(float("nan"))
        logits = model.prefill(state, prompt, lengths=lengths)
        assert torch.isfinite(logits).all()
        # The ragged prefill gives each row the logits of its own unpadded prefill.
        for row, size in enumerate(lengths):
            alone = model.prefill(model.init_state(1, 40, "cpu", torch.float32), prompt[row:row + 1, :size])
            assert torch.allclose(alone[0, -1], logits[row, -1], atol=1e-5)
        # Stepping past the padded slots stays finite and matches single-row decoding.
        singles = [model.init_state(1, 40, "cpu", torch.float32) for _ in lengths]
        single_logits = [model.prefill(s, prompt[row:row + 1, :size]) for row, (s, size) in enumerate(zip(singles, lengths))]
        token = logits[:, -1].argmax(-1)
        for _ in range(12):
            logits = model.step(state, token)
            assert torch.isfinite(logits).all()
            for row, single in enumerate(singles):
                single_logits[row] = model.step(single, token[row:row + 1])
                assert torch.allclose(single_logits[row][0, -1], logits[row, -1], atol=1e-4)
            token = logits[:, -1].argmax(-1)
