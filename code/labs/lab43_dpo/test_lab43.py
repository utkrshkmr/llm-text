import copy
import math

import pytest
import torch
import torch.nn.functional as F

from _labutil import load_impl
from llmsys import prefopt as ref_impl
from llmsys.lora import apply_lora
from llmsys.qwen3 import Qwen3Model

lab = load_impl(__file__)
torch.set_num_threads(1)
P = ref_impl.PreferencePair


@pytest.fixture(scope="module")
def toy():
    torch.manual_seed(0)
    task = ref_impl.ToyPrefTask(seed=0)
    model = Qwen3Model(task.model_config(emb_dim=32, n_layers=1)).eval()
    return task, model


def test_build_pair_batch_layout():
    pairs = [P([1, 2], [3, 4, 9], [5, 9]), P([7], [8, 9], [6, 6, 6, 9])]
    b = lab.build_pair_batch(pairs, pad_id=0)
    ids, m = b["input_ids"], b["loss_mask"]
    assert ids.shape == (4, 5) and m.dtype == torch.bool
    assert ids[0].tolist() == [1, 2, 3, 4, 9] and ids[1].tolist() == [7, 8, 9, 0, 0]
    assert ids[2].tolist() == [1, 2, 5, 9, 0] and ids[3].tolist() == [7, 6, 6, 6, 9]
    assert m[0].tolist() == [False, False, True, True, True]
    assert m[1].tolist() == [False, True, True, False, False]
    assert m[3].tolist() == [False, True, True, True, True]


def test_sequence_logps_matches_manual(toy):
    task, model = toy
    pairs = task.pairs(3)
    b = lab.build_pair_batch(pairs, task.pad_id)
    s, n = lab.sequence_logps(model, b["input_ids"], b["loss_mask"])
    for row in range(6):
        pr = pairs[row % 3]
        resp = pr.chosen if row < 3 else pr.rejected
        ids = torch.tensor([pr.prompt + resp])
        lp = torch.log_softmax(model(ids)[0].float(), -1)
        want = sum(lp[len(pr.prompt) - 1 + t, resp[t]] for t in range(len(resp)))
        assert torch.allclose(s[row], want, atol=1e-4)
        assert int(n[row]) == len(resp)


def test_dpo_loss_properties():
    pc, pr = torch.tensor([-3.0, -1.0]), torch.tensor([-3.0, -5.0])
    rc, rr = torch.tensor([-3.0, -2.0]), torch.tensor([-3.0, -2.0])
    loss, a, c = lab.dpo_loss(pc, pr, rc, rr, beta=0.5)
    assert torch.allclose(loss[0], torch.tensor(math.log(2.0)))      # policy == reference
    h = 0.5 * ((-1 + 2) - (-5 + 2))
    assert torch.allclose(loss[1], -F.logsigmoid(torch.tensor(h)))
    assert torch.allclose(a, 0.5 * (pc - rc)) and torch.allclose(c, 0.5 * (pr - rr))
    # gradient pushes chosen up and rejected down, weighted by sigma(-h)
    x = pc.clone().requires_grad_(); y = pr.clone().requires_grad_()
    lab.dpo_loss(x, y, rc, rr, beta=0.5)[0].sum().backward()
    assert (x.grad < 0).all() and (y.grad > 0).all()
    assert torch.allclose(-x.grad[1], 0.5 * torch.sigmoid(torch.tensor(-h)))
    # conservative DPO at eps = 0.5 is symmetric in h
    l1, _, _ = lab.dpo_loss(pc, pr, rc, rr, 0.5, label_smoothing=0.5)
    l2, _, _ = lab.dpo_loss(pr, pc, rr, rc, 0.5, label_smoothing=0.5)
    assert torch.allclose(l1, l2)


def test_ipo_kto_simpo_orpo_values():
    pc, pr, rc, rr = (torch.tensor([v]) for v in (-1.0, -4.0, -2.0, -2.0))
    tau = 0.25                                       # target margin 1/(2 tau) = 2
    l, _, _ = lab.ipo_loss(pc, pr, rc, rr, tau)       # h = 1 - (-2) = 3 -> (3-2)^2
    assert torch.allclose(l, torch.tensor([1.0]))
    # KTO: desirable with r = 1, z0 = 0.5, beta = 2 -> 1 - sigma(1)
    l, r = lab.kto_loss(torch.tensor([-1.0, -5.0]), torch.tensor([-2.0, -4.0]),
                        torch.tensor([True, False]), torch.tensor(0.5), beta=2.0,
                        lambda_d=1.0, lambda_u=1.5)
    assert torch.allclose(l[0], 1 - torch.sigmoid(torch.tensor(1.0)))
    assert torch.allclose(l[1], 1.5 - 1.5 * torch.sigmoid(torch.tensor(2.0 * (0.5 + 1.0))))
    # SimPO uses per-token averages
    l, a, c = lab.simpo_loss(torch.tensor([-4.0]), torch.tensor([-9.0]), torch.tensor([2.0]),
                             torch.tensor([3.0]), beta=2.0, gamma=0.5)
    assert torch.allclose(l, -F.logsigmoid(torch.tensor([2 * (-2.0) - 2 * (-3.0) - 0.5])))
    # ORPO against the library reference
    args = (torch.tensor([-2.0, -6.0]), torch.tensor([-6.0, -3.0]),
            torch.tensor([4.0, 3.0]), torch.tensor([3.0, 3.0]))
    got, _, _ = lab.orpo_loss(*args, lam=0.2)
    want, _, _ = ref_impl.orpo_loss(*args, lam=0.2)
    assert torch.allclose(got, want, atol=1e-6)
    p = math.exp(-0.5); q = math.exp(-2.0)
    lo = math.log(p / (1 - p)) - math.log(q / (1 - q))
    assert abs(float(got[0]) - (0.5 - 0.2 * math.log(1 / (1 + math.exp(-lo))))) < 1e-5


def test_reference_cache(toy):
    task, model = toy
    pairs = task.pairs(10)
    cache = lab.precompute_reference_logps(model, pairs, task.pad_id, batch_size=4)
    assert cache.shape == (10, 2) and cache.dtype == torch.float32
    b = lab.build_pair_batch(pairs, task.pad_id)
    s, _ = lab.sequence_logps(model, b["input_ids"], b["loss_mask"])
    assert torch.allclose(cache[:, 0], s[:10], atol=1e-4)
    assert torch.allclose(cache[:, 1], s[10:], atol=1e-4)


def test_metrics():
    m = lab.preference_metrics(torch.tensor([1.0, 0.0, 2.0, 0.5]), torch.tensor([0.0, 1.0, 2.0, 0.0]))
    assert abs(m["reward_acc"] - 0.625) < 1e-6
    assert abs(m["margin"] - 0.125) < 1e-6


def test_dpo_shifts_policy(toy):
    """End to end: a few DPO steps on style pairs move P(first token == 'sure') up."""
    task, model = toy
    torch.manual_seed(1)
    pol = copy.deepcopy(model)
    ref = copy.deepcopy(model).eval()
    pairs = task.pairs(128, frac_style=1.0)
    cache = lab.precompute_reference_logps(ref, pairs, task.pad_id)
    before = task.p_polite(pol, n=50)
    hist = lab.dpo_train_loop(pol, pairs, cache, task.pad_id, steps=25, batch_size=16,
                              lr=3e-3, beta=0.5, grad_accum=2)
    after = task.p_polite(pol, n=50)
    assert abs(hist[0]["loss"] - math.log(2)) < 0.05        # starts at log 2: policy == ref
    assert hist[-1]["reward_acc"] > 0.9
    assert after > before + 0.3, (before, after)


def test_lora_reference_is_base(toy):
    """With LoRA, disabling the adapters reproduces the frozen base policy exactly."""
    task, model = toy
    m = apply_lora(copy.deepcopy(model), rank=2, alpha=4.0)
    for mod in m.modules():
        if hasattr(mod, "lora_B"):
            torch.nn.init.normal_(mod.lora_B, std=0.5)
    pairs = task.pairs(4)
    with lab.disable_lora(m):
        a = lab.precompute_reference_logps(m, pairs, task.pad_id)
    b = lab.precompute_reference_logps(model, pairs, task.pad_id)
    c = lab.precompute_reference_logps(m, pairs, task.pad_id)
    assert torch.allclose(a, b, atol=1e-5) and not torch.allclose(a, c, atol=1e-3)
