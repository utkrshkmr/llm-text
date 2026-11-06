"""Tests for Lab 60 (policy-gradient foundations). CPU, < 30 s.

    LAB_IMPL=starter  python -m pytest -q labs/lab60_pg
    LAB_IMPL=solution python -m pytest -q labs/lab60_pg
"""
import math

import pytest
import torch

from _labutil import load_impl
from llmsys import pg as ref

lab = load_impl(__file__)
torch.set_num_threads(1)

LOGITS = torch.tensor([0.5, 0.0, -0.5])
REWARDS = torch.tensor([1.0, 0.2, 0.0])


def test_bandit_estimates_unbiased_for_every_baseline():
    true_grad, _ = ref.bandit_grad_moments(LOGITS, REWARDS, 0.0)
    g = torch.Generator().manual_seed(0)
    for kind, b in [("none", 0.0), ("const", 0.7), ("loo", 0.0)]:
        est = lab.bandit_estimates(LOGITS, REWARDS, n=20000, G=4, baseline=kind, b=b, generator=g)
        assert est.shape == (20000, 3)
        assert torch.allclose(est.mean(0).double(), true_grad, atol=0.01), kind


def test_optimal_baseline_minimises_variance():
    b_star = lab.optimal_baseline(LOGITS, REWARDS)
    v_star = ref.bandit_grad_moments(LOGITS, REWARDS, b_star)[1]
    for b in torch.linspace(-0.5, 1.5, 21):
        assert v_star <= ref.bandit_grad_moments(LOGITS, REWARDS, float(b))[1] + 1e-12
    assert abs(b_star - ref.optimal_baseline(LOGITS, REWARDS)) < 1e-9


def test_build_rl_batch_layout():
    out = lab.build_rl_batch([[1, 2, 3], [4]], [[5, 6], [7, 8, 9]], pad_id=0)
    assert out["input_ids"].tolist() == [[1, 2, 3, 5], [4, 7, 8, 9]]
    assert out["labels"].tolist() == [[2, 3, 5, 6], [7, 8, 9, 0]]
    assert out["response_mask"].tolist() == [[False, False, True, True], [True, True, True, False]]


def test_logprobs_and_entropy_match_reference_and_chunking():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(3, 5, 11, generator=g).to(torch.bfloat16)
    labels = torch.randint(0, 11, (3, 5), generator=g)
    lp, ent = lab.token_logprobs_entropy(logits, labels)
    lp2, ent2 = lab.token_logprobs_entropy(logits, labels, chunk_rows=4)
    ls = torch.log_softmax(logits.float(), -1)
    assert lp.dtype == torch.float32
    assert torch.allclose(lp, ls.gather(-1, labels[..., None]).squeeze(-1), atol=1e-5)
    assert torch.allclose(ent, -(ls.exp() * ls).sum(-1), atol=1e-5)
    assert torch.allclose(lp, lp2) and torch.allclose(ent, ent2)


def test_group_advantages():
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    a = lab.group_advantages(r, "mean", "none")
    assert torch.allclose(a, torch.tensor([[0.5, -0.5, -0.5, 0.5], [0.0, 0.0, 0.0, 0.0]]))
    loo = lab.group_advantages(r, "loo", "none")
    assert torch.allclose(loo[0], torch.tensor([2 / 3, -2 / 3, -2 / 3, 2 / 3]))
    z = lab.group_advantages(r, "mean", "std", eps=0.0)
    assert torch.allclose(z[0], a[0] / r[0].std())
    assert torch.allclose(lab.group_advantages(r, "mean", "std")[1], torch.zeros(4))
    assert torch.equal(lab.group_advantages(r, "none", "none"), r)


def test_aggregate_modes():
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]])
    m = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    assert torch.isclose(lab.aggregate_loss(x, m, "seq_mean"), torch.tensor((2.0 + 4.0) / 2))
    assert torch.isclose(lab.aggregate_loss(x, m, "token_mean"), torch.tensor(10.0 / 4))
    assert torch.isclose(lab.aggregate_loss(x, m, "constant", Z=8.0), torch.tensor(10.0 / 8))
    assert torch.isclose(lab.aggregate_loss(x, m, "seq_mean", n_seq_total=4), torch.tensor(6.0 / 4))


def test_pg_losses_values_and_gradients():
    logp = torch.log(torch.tensor([[0.5, 0.25], [0.2, 0.4]])).requires_grad_()
    old = torch.log(torch.tensor([[0.4, 0.25], [0.4, 0.4]]))
    adv = torch.tensor([1.0, -1.0])
    mask = torch.ones(2, 2, dtype=torch.bool)
    loss, _ = lab.pg_token_loss(logp, adv, mask, method="reinforce")
    assert torch.allclose(loss, -adv[:, None] * logp)
    l_is, _ = lab.pg_token_loss(logp, adv, mask, old, method="is_token")
    assert torch.allclose(l_is, -adv[:, None] * torch.exp(logp - old))
    l_seq, _ = lab.pg_token_loss(logp, adv, mask, old, method="is_seq")
    seqw = torch.exp((logp - old).sum(1, keepdim=True))
    assert torch.allclose(l_seq, -adv[:, None] * seqw.expand(-1, 2))
    l_clip, st = lab.pg_token_loss(logp, adv, mask, old, method="ppo_clip", eps=0.2)
    # ratios: 1.25 (A>0 -> clipped), 1.0, 0.5 (A<0 -> clipped), 1.0
    assert abs(st["clip_frac"] - 0.5) < 1e-6
    l_clip.sum().backward()
    gr = logp.grad
    assert gr[0, 0] == 0 and gr[1, 0] == 0          # clipped tokens get no gradient
    assert gr[0, 1] != 0 and gr[1, 1] != 0


def test_rft_loss_keeps_only_successes():
    logp = torch.tensor([[-1.0, -2.0], [-0.5, -0.5]])
    out = lab.rft_token_loss(logp, torch.tensor([0.0, 1.0]))
    assert torch.allclose(out, torch.tensor([[0.0, 0.0], [0.5, 0.5]]))


@pytest.mark.parametrize("mode", ["seq_mean", "token_mean", "constant"])
def test_accumulation_matches_full_batch(mode):
    task = ref.DigitSumTask()
    model = ref.make_toy_policy(task, seed=0)
    g = torch.Generator().manual_seed(3)
    prompts = task.sample_prompts(8, g)
    resp = torch.randint(0, 10, (8, 5), generator=g)
    resp[1, 3:] = -1                    # variable lengths via padding
    resp[5, 2:] = -1
    adv = torch.randn(8, generator=g)
    full = lab.accumulated_gradient(model, prompts, resp, adv, 1, mode, Z=40.0)
    micro = lab.accumulated_gradient(model, prompts, resp, adv, 4, mode, Z=40.0)
    assert torch.allclose(full, micro, atol=1e-6), mode


def test_sequence_ratio_variance_grows_exponentially():
    p0 = torch.tensor([0.5, 0.3, 0.2])
    p1 = torch.tensor([0.45, 0.33, 0.22])
    v1, v100 = lab.seq_is_variance(p0, p1, 1), lab.seq_is_variance(p0, p1, 100)
    assert abs(v1 - ref.chi_square(p0, p1)) < 1e-12
    assert v100 > 50 * v1
    # Monte Carlo check at L = 10
    g = torch.Generator().manual_seed(0)
    y = torch.multinomial(p0.expand(200000, -1), 10, replacement=True, generator=g)
    w = (p1[y] / p0[y]).prod(1)
    assert abs(float(w.var()) - lab.seq_is_variance(p0, p1, 10)) < 0.01


def test_toy_policy_learns_with_group_baseline():
    hist = ref.train_toy(ref.PGConfig(method="grpo", steps=25, prompts_per_step=8, group_size=8,
                                      lr=5e-3, seed=0))
    assert sum(hist.reward[-5:]) / 5 > sum(hist.reward[:5]) / 5 + 0.1
