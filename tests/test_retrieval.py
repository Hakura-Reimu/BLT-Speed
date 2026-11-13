import torch

from blt.config import ByteConfig, TransformerConfig
from blt.retrieval import RETRIEVAL_ANSWER_BYTES, evaluate_retrieval, retrieval_prompt
from blt.tasks import retrieval_cases
from blt.train import make_model


def test_v2_prompt_ends_before_the_answer_space():
    filler = ("Plain filler text with words. " * 40).encode("ascii")
    case = retrieval_cases(filler, 512, 0.5, 1, 3)[0]
    prompt = retrieval_prompt(case)
    assert prompt.endswith("The pass key is") and not prompt.endswith(" ")
    assert RETRIEVAL_ANSWER_BYTES == len(" " + case["key"])


def test_v2_evaluation_scores_answers_with_the_leading_space():
    torch.manual_seed(0)
    trunk = TransformerConfig(1, 8, 2, 1, 4, 16, max_positions=256)
    config = ByteConfig("tiny", trunk, 8, 2, 16, 2, 6, max_len=8, block_size=16, hash_rows=8)
    model = make_model(config).eval()
    filler = ("Plain filler text with words. " * 40).encode("ascii")
    result = evaluate_retrieval(model, retrieval_cases(filler, 256, 0.5, 2, 5))
    assert result["count"] == 2 and result["prompt_format"].startswith("v2")
    for trial in result["trials"]:
        assert len(trial["candidate_nll"]) == 10 and 1 <= trial["true_key_rank"] <= 10
        assert trial["greedy_exact"] == (bytes.fromhex(trial["generated_hex"])[:6] == (" " + trial["key"]).encode())
