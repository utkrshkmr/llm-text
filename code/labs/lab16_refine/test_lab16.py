import math

import torch

from _labutil import load_impl
from llmsys.qwen3 import QWEN3_CONFIG_TINY, Qwen3Model

lab = load_impl(__file__)


def test_heuristic():
    a = lab.heuristic_score(r"so \boxed{4}")
    b = lab.heuristic_score("the answer is 4")
    c = lab.heuristic_score("no idea")
    assert a > b > c
    short, long = lab.heuristic_score(r"\boxed{4}"), lab.heuristic_score("x" * 2000 + r"\boxed{4}")
    assert short > long > 2.0
    assert abs(lab.heuristic_score("") - 1.0) < 1e-9


def test_token_logprobs_match_incremental():
    torch.manual_seed(0)
    m = Qwen3Model(QWEN3_CONFIG_TINY).eval()
    prompt, ans = [5, 9, 2, 7], [11, 3, 8]
    lp = lab.answer_token_logprobs(m, prompt, ans)
    assert lp.shape == (3,)
    with torch.no_grad():
        seq = list(prompt)
        for t, tok in enumerate(ans):
            logits = m(torch.tensor([seq]))[0, -1]
            ref = torch.log_softmax(logits.float(), -1)[tok]
            assert torch.allclose(lp[t], ref, atol=1e-5)
            seq.append(tok)


def test_confidence():
    s = lab.confidence_scores(torch.tensor([-0.1, -0.5, -2.0, 0.0]))
    assert abs(s["sum"] + 2.6) < 1e-6 and abs(s["mean"] + 0.65) < 1e-6
    assert abs(s["ppl"] - math.exp(0.65)) < 1e-6 and abs(s["min"] + 2.0) < 1e-6


def _fake_llm(script):
    it = iter(script)
    return lambda prompt: next(it)


def test_refine_accepts_and_rejects():
    cp = lambda q, d: f"CRIT[{d}]"
    rp = lambda q, d, c: f"REF[{d}|{c}]"
    script = ["draft 1", "crit A", r"\boxed{2}", "crit B", "rambling worse answer with no box"]
    out = lab.self_refine("q", _fake_llm(script), cp, rp, score_fn=lab.heuristic_score, iterations=2)
    assert out["final"] == r"\boxed{2}" and out["calls"] == 5
    assert [h["accepted"] for h in out["history"]] == [True, False]
    out2 = lab.self_refine("q", _fake_llm(script), cp, rp, score_fn=None, iterations=2)
    assert out2["final"] == "rambling worse answer with no box"


def test_cost():
    c = lab.refine_cost_tokens(q=100, a=300, c=80, iterations=2)
    assert c == {"prefill": 100 + 2 * (400 + 480), "decode": 300 + 2 * 380, "sequential_calls": 5}
