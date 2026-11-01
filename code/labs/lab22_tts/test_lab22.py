import math

import torch

from _labutil import load_impl
from llmsys.qwen3 import QWEN3_CONFIG_TINY, Qwen3Model
from llmsys.tts import token_logprobs

lab = load_impl(__file__)


def entropy(p):
    p = p.clamp_min(1e-12)
    return float(-(p * p.log()).sum())


def test_temperature():
    logits = torch.tensor([4.0, 2.0, 1.0, 0.5])
    p1 = lab.softmax_with_temperature(logits, 1.0)
    assert torch.allclose(p1, torch.softmax(logits, -1))
    hs = [entropy(lab.softmax_with_temperature(logits, t)) for t in (0.25, 0.5, 1.0, 2.0, 8.0)]
    assert all(a < b for a, b in zip(hs, hs[1:]))           # entropy grows with T
    assert hs[-1] < math.log(4) + 1e-6
    g = lab.softmax_with_temperature(logits, 0.0)
    assert g.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_top_p():
    p = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    out = lab.top_p_filter_probs(p, 0.7)
    assert torch.allclose(out, torch.tensor([[0.625, 0.375, 0.0, 0.0]]))
    out = lab.top_p_filter_probs(p, 0.1)                    # top token always survives
    assert torch.allclose(out, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.equal(lab.top_p_filter_probs(p, 1.0), p)


def test_sampling_frequencies():
    g = torch.Generator().manual_seed(0)
    probs = torch.tensor([[0.1, 0.6, 0.3]]).expand(20000, -1)
    s = lab.sample_from_probs(probs, g).squeeze(1)
    freq = torch.bincount(s, minlength=3).float() / s.numel()
    assert torch.allclose(freq, torch.tensor([0.1, 0.6, 0.3]), atol=0.015)
    zero = torch.tensor([[0.0, 1.0, 0.0]])
    assert int(lab.sample_from_probs(zero, g)) == 1


def test_majority_vote():
    assert lab.majority_vote(["3", "4", "4", "3"]) == "3"                # tie -> first seen
    assert lab.majority_vote([None, "", "2", "5", "5"]) == "5"
    assert lab.majority_vote(["1/2", "0.5", "2"], key=lambda a: str(float(eval(a)))) == "1/2"
    assert lab.majority_vote([None, ""]) is None


def test_self_consistency_early_stop():
    outs = ["7", "7", "7", "7", "2", "7", "7", "7"]
    r = lab.self_consistency(lambda i: outs[i], lambda t: t, n=8, early_stop=True)
    assert r["answer"] == "7" and r["num_samples"] == 6     # after 6: lead 5-1=4 > 2 remaining
    r = lab.self_consistency(lambda i: outs[i], lambda t: t, n=8, early_stop=False)
    assert r["num_samples"] == 8


def test_shared_prefix_logprobs_and_greedy():
    torch.manual_seed(0)
    m = Qwen3Model(QWEN3_CONFIG_TINY).eval()
    prompt = [3, 14, 15, 92, 65]
    g = torch.Generator().manual_seed(7)
    r = lab.sample_n_shared_prefix(m, prompt, n=3, max_new_tokens=6, temperature=2.0, generator=g)
    assert len(r["tokens"]) == 3 and all(len(t) == 6 for t in r["tokens"])
    for t, lp in zip(r["tokens"], r["logprobs"]):
        assert torch.allclose(token_logprobs(m, prompt, t), torch.tensor(lp), atol=1e-4)
    gr = lab.sample_n_shared_prefix(m, prompt, n=2, max_new_tokens=5, temperature=0.0)
    assert gr["tokens"][0] == gr["tokens"][1]


def test_counts():
    c = lab.forward_token_counts(1000, 200, 8)
    assert c == {"naive": 9600, "shared": 2600}
