import math

import pytest
import torch

from _labutil import load_impl
from llmsys.generate import sequence_logprob
from llmsys.gpt import GPT_CONFIG_TINY, GPTModel
from llmsys.qwen3 import QWEN3_CONFIG_TINY, Qwen3Model
from llmsys.utils import set_seed

lab = load_impl(__file__)


# ---------------------------------------------------------------- Task 1: bin packing
def test_ffd_pack_valid():
    lens = [700, 50, 300, 300, 1000, 20, 999, 1, 450, 600]
    bins = lab.ffd_pack(lens, 1024)
    flat = sorted(i for b in bins for i in b)
    assert flat == list(range(len(lens)))
    assert all(sum(lens[i] for i in b) <= 1024 for b in bins)
    lower = math.ceil(sum(lens) / 1024)
    assert lower <= len(bins) <= lower + 2
    with pytest.raises(ValueError):
        lab.ffd_pack([2000], 1024)


def test_efficiency_report():
    lens = [100, 20, 30, 500, 60, 45, 1000, 80]
    rep = lab.efficiency_report(lens, batch_size=4, max_tokens=1024)
    assert rep["padded"] < rep["sorted_padded"] < rep["packed"] <= 1.0
    assert rep["real_tokens"] == sum(lens)


# ---------------------------------------------------------------- Task 2: varlen / packed log-probs
def test_block_causal_mask():
    m = lab.block_causal_mask(torch.tensor([0, 2, 5]))
    allowed = (~m).int().tolist()
    assert allowed == [[1, 0, 0, 0, 0],
                       [1, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 1, 1, 0],
                       [0, 0, 1, 1, 1]]


@pytest.mark.parametrize("Model,cfg", [(Qwen3Model, QWEN3_CONFIG_TINY), (GPTModel, GPT_CONFIG_TINY)])
def test_packed_logprobs_match_unpacked(Model, cfg):
    set_seed(0)
    m = Model(cfg).eval()
    lens, pls = [5, 9, 3, 12, 7, 4], [2, 3, 1, 5, 4, 2]
    seqs = [torch.randint(0, 500, (n,)).tolist() for n in lens]
    got = lab.packed_completion_logprobs(m, seqs, pls, max_tokens=20)
    ref = [float(sequence_logprob(m, torch.tensor(s[:p]), torch.tensor(s[p:])))
           for s, p in zip(seqs, pls)]
    for g, r in zip(got, ref):
        assert abs(g - r) < 1e-3


# ---------------------------------------------------------------- Task 3: mismatch
def test_is_weights():
    lt = torch.log(torch.tensor([0.5, 0.5, 0.1, 0.9]))
    lr = torch.log(torch.tensor([0.5, 0.1, 0.5, 0.3]))
    w = lab.is_weights(lt, lr, cap=2.0, mode="truncate")
    assert torch.allclose(w, torch.tensor([1.0, 2.0, 0.2, 2.0]), atol=1e-5)
    m = lab.is_weights(lt, lr, cap=2.0, mode="mask")
    assert m.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert not lab.is_weights(lt.requires_grad_(), lr).requires_grad


def test_mismatch_metrics_bf16():
    set_seed(0)
    m = Qwen3Model(QWEN3_CONFIG_TINY).eval()
    ids = torch.randint(0, 500, (2, 16))
    a = lab.token_logprobs(m, ids)
    b = lab.token_logprobs(m, ids, dtype=torch.bfloat16)
    met = lab.mismatch_metrics(a, b)
    assert met["kl_k3"] >= 0 and met["mean_abs_logp_diff"] > 0
    zero = lab.mismatch_metrics(a, a)
    assert zero["kl_k3"] == pytest.approx(0.0, abs=1e-7) and zero["max_ratio"] == pytest.approx(1.0)


def test_clipped_pg_loss():
    set_seed(0)
    B, T = 3, 5
    old = torch.randn(B, T) - 2
    new = (old + 0.3 * torch.randn(B, T)).requires_grad_()
    adv = torch.tensor([1.0, -1.0, 0.5])
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, :2] = False
    base = lab.clipped_pg_loss(new, old, adv, mask)
    same = lab.clipped_pg_loss(new, old, adv, mask, logp_rollout=old, tis_cap=2.0)
    assert torch.allclose(base, same)
    # at new == old the gradient is -A / n_tokens on every valid token
    new0 = old.clone().requires_grad_()
    lab.clipped_pg_loss(new0, old, adv, mask).backward()
    exp = -(adv[:, None].expand(B, T) * mask) / mask.sum()
    assert torch.allclose(new0.grad, exp, atol=1e-6)


# ---------------------------------------------------------------- Task 4: pipeline simulation
def test_run_generation_continuous_batching():
    rs = [lab.Rollout(i, L, float(L)) for i, L in enumerate([10, 20, 30, 40])]
    t, fin, unfin, busy = lab.run_generation(rs, n_slots=2, tps=10.0, stop_after=None, version=0)
    # slot A: 10 then 30 -> 4 s ; slot B: 20 then 40 -> 6 s
    assert t == pytest.approx(6.0) and len(fin) == 4 and not unfin
    assert busy == pytest.approx(10.0)
    rs = [lab.Rollout(i, L, float(L)) for i, L in enumerate([10, 100])]
    t, fin, unfin, _ = lab.run_generation(rs, n_slots=2, tps=10.0, stop_after=1, version=3)
    assert t == pytest.approx(1.0) and len(unfin) == 1 and unfin[0].remaining == pytest.approx(90.0)
    assert unfin[0].versions == [3]


def test_policies():
    cfg = lab.SimConfig(n_steps=12, seed=1)
    res = {p: lab.simulate(p, cfg) for p in ["colocated", "sync", "async", "partial"]}
    assert res["sync"].mean_staleness == 0 and res["colocated"].mean_staleness == 0
    assert res["async"].samples_per_sec > res["sync"].samples_per_sec
    assert max(res["async"].staleness) <= cfg.max_staleness
    assert res["partial"].samples_per_sec > res["sync"].samples_per_sec
    assert res["partial"].mixed_policy_fraction > 0
    assert all(r.samples == cfg.n_steps * cfg.batch_size for r in res.values())


def test_weight_sync():
    t = lab.weight_sync_seconds(8e9, 2, 50.0)
    assert t == pytest.approx(0.32)
    assert lab.weight_sync_seconds(8e9, 2, 50.0, 4, pipelined_broadcast=False) == pytest.approx(1.28)


# ---------------------------------------------------------------- Task 5: sandboxed reward
def test_sandbox():
    ok = lab.run_python_reward("def f(x):\n    return x + 1", "assert f(1) == 2")
    bad = lab.run_python_reward("def f(x):\n    return x", "assert f(1) == 2")
    loop = lab.run_python_reward("while True:\n    pass", "", timeout=1.0)
    crash = lab.run_python_reward("import nonexistent_module_xyz", "")
    assert (ok.status, ok.reward) == ("pass", 1.0)
    assert (bad.status, bad.reward) == ("fail", 0.0)
    assert loop.status == "timeout" and crash.status == "error"
    batch = lab.score_batch([("x=1", "assert x==1"), ("x=1", "assert x==2")], max_workers=2)
    assert [r.reward for r in batch] == [1.0, 0.0]
