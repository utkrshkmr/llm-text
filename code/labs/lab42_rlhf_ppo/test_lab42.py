"""Tests for Lab 42 (PPO-RLHF). Run with LAB_IMPL=starter (default) or LAB_IMPL=solution."""
import itertools
import math

import pytest
import torch
import torch.nn.functional as F

from _labutil import load_impl

lab = load_impl(__file__)
torch.manual_seed(0)


def _mask(lengths, T):
    return (torch.arange(T)[None] < torch.tensor(lengths)[:, None]).float()


# ----------------------------------------------------------------------------- Task 1
def test_masked_mean_and_whiten():
    x = torch.tensor([[1.0, 2.0, 100.0], [3.0, 100.0, 100.0]])
    m = _mask([2, 1], 3)
    assert torch.isclose(lab.masked_mean(x, m), torch.tensor(2.0))
    w = lab.whiten(x, m)
    real = w[m > 0]
    assert abs(float(real.mean())) < 1e-5 and abs(float(real.std(unbiased=False)) - 1) < 1e-3
    assert torch.all(w[m == 0] == 0), "padding must stay zero after whitening"


# ----------------------------------------------------------------------------- Task 2
def test_bt_loss_and_accuracy():
    rc, rr = torch.tensor([2.0, 0.0, -1.0]), torch.tensor([1.0, 0.5, -3.0])
    expected = -torch.log(torch.sigmoid(rc - rr)).mean()
    assert torch.isclose(lab.bt_loss(rc, rr), expected, atol=1e-6)
    m = torch.tensor([0.5, 0.5, 0.5])
    assert torch.isclose(lab.bt_loss(rc, rr, m), -torch.log(torch.sigmoid(rc - rr - m)).mean(), atol=1e-6)
    assert abs(lab.rm_accuracy(rc, rr) - 2 / 3) < 1e-6
    # gradient points the right way
    a = torch.zeros(4, requires_grad=True)
    lab.bt_loss(a, torch.zeros(4)).backward()
    assert torch.all(a.grad < 0)


# ----------------------------------------------------------------------------- Task 3
def test_ranking_loss():
    s = torch.randn(5, 2)
    ranks = torch.tensor([[0, 1], [1, 0], [0, 1], [0, 1], [1, 0]])
    chosen = torch.where(ranks[:, 0] == 0, s[:, 0], s[:, 1])
    rejected = torch.where(ranks[:, 0] == 0, s[:, 1], s[:, 0])
    assert torch.isclose(lab.ranking_loss(s, ranks), lab.bt_loss(chosen, rejected), atol=1e-6)
    s4 = torch.randn(3, 4)
    r4 = torch.stack([torch.randperm(4) for _ in range(3)])
    terms = []
    for p in range(3):
        for i, j in itertools.combinations(range(4), 2):
            w, l = (i, j) if r4[p, i] < r4[p, j] else (j, i)
            terms.append(-F.logsigmoid(s4[p, w] - s4[p, l]))
    assert torch.isclose(lab.ranking_loss(s4, r4), torch.stack(terms).mean(), atol=1e-6)


# ----------------------------------------------------------------------------- Task 4
def test_kl_shaped_rewards():
    lp = torch.tensor([[-1.0, -0.5, -2.0, -0.3], [-0.2, -0.1, 0.0, 0.0]])
    lr = torch.tensor([[-1.2, -0.5, -1.5, -0.4], [-0.4, -0.1, 0.0, 0.0]])
    m = _mask([4, 2], 4)
    r, kl = lab.kl_shaped_rewards(torch.tensor([2.0, -1.0]), lp, lr, m, beta=0.1)
    assert torch.allclose(kl, (lp - lr) * m)
    assert torch.allclose(r[0], torch.tensor([-0.02, 0.0, 0.05, 1.99]), atol=1e-6)
    assert torch.allclose(r[1], torch.tensor([-0.02, -1.0, 0.0, 0.0]), atol=1e-6), \
        "score goes on each row's own last real token"
    _, k3 = lab.kl_shaped_rewards(torch.zeros(2), lp, lr, m, beta=0.1, kind="k3")
    assert torch.all(k3 >= -1e-7), "k3 is non-negative"


# ----------------------------------------------------------------------------- Task 5
def _gae_bruteforce(r, v, lengths, gamma, lam):
    B, T = r.shape
    adv = torch.zeros(B, T)
    for b in range(B):
        n = lengths[b]
        for t in range(n):
            total = 0.0
            for l in range(t, n):
                v_next = v[b, l + 1] if l + 1 < n else 0.0
                delta = r[b, l] + gamma * v_next - v[b, l]
                total += (gamma * lam) ** (l - t) * delta
            adv[b, t] = total
    return adv


@pytest.mark.parametrize("gamma,lam", [(1.0, 0.95), (0.9, 0.5), (1.0, 1.0), (1.0, 0.0)])
def test_gae_matches_bruteforce(gamma, lam):
    lengths = [5, 2, 3]
    m = _mask(lengths, 5)
    r = torch.randn(3, 5) * m
    v = torch.randn(3, 5) * m
    adv, ret = lab.gae(r, v, m, gamma, lam)
    ref = _gae_bruteforce(r, v, lengths, gamma, lam)
    assert torch.allclose(adv, ref, atol=1e-5)
    assert torch.allclose(ret, (ref + v) * m, atol=1e-5)
    assert torch.all(adv[m == 0] == 0)


def test_gae_worked_example():
    r = torch.tensor([[-0.02, 0.0, 0.05, 1.99]])
    v = torch.tensor([[0.5, 0.8, 1.2, 1.6]])
    adv, _ = lab.gae(r, v, torch.ones(1, 4), 1.0, 0.95)
    assert torch.allclose(adv, torch.tensor([[1.4005, 1.1795, 0.8205, 0.39]]), atol=1e-3)


# ----------------------------------------------------------------------------- Task 6
def test_ppo_policy_loss():
    m = _mask([3, 2], 3)
    adv = torch.tensor([[1.0, -1.0, 2.0], [0.5, -0.5, 0.0]])
    lp = torch.randn(2, 3)
    loss, st = lab.ppo_policy_loss(lp, lp.clone(), adv, m, clip=0.2)
    assert torch.isclose(loss, -(adv * m).sum() / m.sum(), atol=1e-6)
    assert st["clipfrac"] == 0.0 and abs(st["approx_kl"]) < 1e-12
    # ratio 1.5 with positive advantage is clipped to 1.2; with negative advantage it is not
    old = torch.zeros(1, 2)
    new = torch.log(torch.tensor([[1.5, 1.5]]))
    loss, st = lab.ppo_policy_loss(new, old, torch.tensor([[1.0, -1.0]]), torch.ones(1, 2), clip=0.2)
    assert torch.isclose(loss, torch.tensor(-(1.2 * 1.0 + 1.5 * -1.0) / 2), atol=1e-6)
    assert st["clipfrac"] == 1.0
    # no gradient flows through the clipped positive-advantage token
    new = new.clone().requires_grad_(True)
    loss, _ = lab.ppo_policy_loss(new, old, torch.tensor([[1.0, -1.0]]), torch.ones(1, 2), clip=0.2)
    loss.backward()
    assert new.grad[0, 0] == 0 and new.grad[0, 1] != 0


# ----------------------------------------------------------------------------- Task 7
def test_value_loss():
    m = torch.ones(1, 3)
    R = torch.tensor([[1.0, 1.0, 1.0]])
    old = torch.zeros(1, 3)
    v = torch.tensor([[0.1, 0.5, 2.0]])
    loss, frac = lab.value_loss(v, old, R, m, clip=None)
    assert torch.isclose(loss, 0.5 * ((v - R) ** 2).mean())
    loss, frac = lab.value_loss(v, old, R, m, clip=0.2)
    vc = torch.tensor([[0.1, 0.2, 0.2]])
    expected = 0.5 * torch.maximum((v - R) ** 2, (vc - R) ** 2).mean()
    assert torch.isclose(loss, expected) and abs(frac - 2 / 3) < 1e-6


# ----------------------------------------------------------------------------- Task 8
def test_explained_variance_and_kl_controller():
    m = torch.ones(2, 4)
    R = torch.randn(2, 4)
    assert abs(lab.explained_variance(R.clone(), R, m) - 1.0) < 1e-6
    assert abs(lab.explained_variance(torch.full_like(R, float(R.mean())), R, m)) < 1e-5
    ctl = lab.AdaptiveKLController(0.1, target=6.0, horizon=100)
    b1 = ctl.update(12.0, 10)          # overspending -> beta rises (error clipped at +0.2)
    assert math.isclose(b1, 0.1 * (1 + 0.2 * 0.1))
    b2 = ctl.update(3.0, 10)           # underspending -> beta falls
    assert b2 < b1


# ----------------------------------------------------------------------------- Task 9: the system
def test_pipeline_smoke():
    torch.set_num_threads(1)
    from llmsys.rlhf.ppo import PPOConfig
    out = lab.train_pipeline(seed=0, sft_steps=60, n_pairs=384, rm_epochs=3, ppo_iters=6,
                             ppo_cfg=PPOConfig(batch_size=48, ppo_epochs=4, minibatch_size=16,
                                               init_beta=0.05, seed=0))
    assert out["rm_hist"]["val_acc"][-1] > 0.6, "the reward model should beat chance clearly"
    h = out["history"]
    assert h[0]["kl"] == pytest.approx(0.0, abs=1e-6), "first rollout: policy == reference"
    start = h[0]["rm_score"]
    end = sum(x["rm_score"] for x in h[-2:]) / 2
    assert end > start + 0.5, f"PPO should raise the learned reward ({start:.2f} -> {end:.2f})"
