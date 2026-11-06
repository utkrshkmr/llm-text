import math

import pytest
import torch

from _labutil import load_impl
from llmsys.qwen3 import Qwen3Model
from llmsys.rl.toy import ToyTokenizer, toy_model_config

lab = load_impl(__file__)
torch.set_num_threads(1)


def test_entropy_uniform_and_peaked():
    V = 7
    assert abs(lab.token_entropy(torch.zeros(V)).item() - math.log(V)) < 1e-5
    peaked = torch.tensor([20.0] + [0.0] * (V - 1))
    assert lab.token_entropy(peaked).item() < 1e-3


def test_sequence_logprob_and_entropy():
    torch.manual_seed(0)
    tok = ToyTokenizer()
    m = Qwen3Model(toy_model_config(tok)).eval()
    ids = torch.tensor(tok.encode("Q:1+1=\nA:\\boxed{2}<|eos|>"))
    pl = len(tok.encode("Q:1+1=\nA:"))
    lp, ent = lab.sequence_logprob_and_entropy(m, ids, pl)
    assert lp.requires_grad and not ent.requires_grad
    logits = m(ids[None])[0]
    manual_ent = torch.stack([lab.token_entropy(logits[t]) for t in range(pl - 1, len(ids) - 1)]).mean()
    assert torch.allclose(ent, manual_ent, atol=1e-5)
    assert 0 < ent.item() <= math.log(tok.vocab_size) + 1e-4


def test_chunked_matches_full():
    torch.manual_seed(0)
    logits = torch.randn(37, 11)
    tg = torch.randint(0, 11, (37,))
    lp, ent = lab.chunked_logprob_entropy(logits, tg, chunk=5)
    full = torch.log_softmax(logits, -1)
    assert torch.allclose(lp, full.gather(-1, tg[:, None]).squeeze(-1), atol=1e-6)
    assert torch.allclose(ent, lab.token_entropy(logits), atol=1e-6)


def test_advantage_stats():
    adv, m, s = lab.advantage_stats([1.0, 1.0, 0.0, 0.0])
    assert abs(m) < 1e-6 and abs(s - 1.0) < 1e-3
    adv, m, s = lab.advantage_stats([0.0, 0.0, 0.0])
    assert s == 0.0 and torch.all(adv == 0)


def test_policy_ratios_and_clip():
    new = torch.tensor([-2.0, -5.0, -4.0])
    old = torch.tensor([-4.0, -5.2, -3.0])
    r, rc = lab.policy_ratios(new, old, clip_eps=0.2)
    assert torch.allclose(r, torch.exp(new - old))
    assert torch.allclose(rc, torch.tensor([1.2, 1.2, 0.8]), atol=1e-6)


def test_clipped_loss_values_and_gradients():
    new = torch.tensor([0.5, -0.5, 0.05], requires_grad=True)   # log ratios vs old=0
    old = torch.zeros(3)
    adv = torch.tensor([1.0, -1.0, 1.0])
    loss, mean_ratio, frac = lab.clipped_policy_loss(new, old, adv, 0.2, 0.2)
    # rollout 0: ratio e^0.5 > 1.2 with A>0 -> clipped, no gradient
    # rollout 1: ratio e^-0.5 < 0.8 with A<0 -> clipped, no gradient
    # rollout 2: inside the trust region -> gradient -A*ratio/3
    expected = -(1.2 * 1.0 + 0.8 * -1.0 + math.exp(0.05) * 1.0) / 3
    assert abs(loss.item() - expected) < 1e-6
    loss.backward()
    assert new.grad[0] == 0 and new.grad[1] == 0
    assert abs(new.grad[2].item() + math.exp(0.05) / 3) < 1e-6
    assert abs(frac - 2 / 3) < 1e-6


def test_clip_does_not_block_corrective_direction():
    # A<0 and ratio already *above* 1+eps: the unclipped term is smaller -> still gradient
    new = torch.tensor([0.5], requires_grad=True)
    loss, _, _ = lab.clipped_policy_loss(new, torch.zeros(1), torch.tensor([-1.0]), 0.2, 0.2)
    loss.backward()
    assert new.grad[0] > 0  # descent lowers the log-prob of this bad rollout


def test_kl_estimators():
    lp_new = torch.log(torch.tensor([0.5, 0.2]))
    lp_ref = torch.log(torch.tensor([0.25, 0.4]))
    r = torch.exp(lp_ref - lp_new)
    assert torch.allclose(lab.kl_k1(lp_new, lp_ref), -torch.log(r))
    assert torch.allclose(lab.kl_k2(lp_new, lp_ref), 0.5 * torch.log(r) ** 2)
    assert torch.allclose(lab.kl_k3(lp_new, lp_ref), r - 1 - torch.log(r))
    assert torch.all(lab.kl_k3(lp_new, lp_ref) >= 0)
    # unbiasedness of k3 for a 3-way categorical: E_p[k3] = KL(p||q)
    p = torch.tensor([0.6, 0.3, 0.1]); q = torch.tensor([0.2, 0.5, 0.3])
    k3 = lab.kl_k3(torch.log(p), torch.log(q))
    assert abs((p * k3).sum().item() - (p * torch.log(p / q)).sum().item()) < 1e-6


def test_kl_surrogates():
    new = torch.tensor([-3.0, -5.0], requires_grad=True)
    ref = torch.tensor([-4.0, -4.0])
    old = torch.tensor([-3.0, -5.0])
    assert abs(lab.kl_sequence_surrogate(new, ref, 0.1).item() - 0.1 * 0.0) < 1e-6
    w = lab.kl_importance_weighted(new, old, ref, 0.1)
    assert abs(w.item() - 0.1 * ((1 * (1.0)) + (1 * (-1.0))) / 2) < 1e-6
    new2 = torch.tensor([-2.0, -5.0])
    w2 = lab.kl_importance_weighted(new2, old, ref, 1.0)
    assert abs(w2.item() - (math.e * 2.0 + 1.0 * -1.0) / 2) < 1e-5


def test_reward_format_and_total():
    tok = ToyTokenizer()
    p = tok.encode("Q:1+1=\nA:")
    ok = torch.tensor(p + tok.encode("<think>1+1=2</think>\\boxed{2}<|eos|>"))
    rev = torch.tensor(p + tok.encode("</think>1+1=2<think>\\boxed{2}<|eos|>"))
    one = torch.tensor(p + tok.encode("<think>1+1=2\\boxed{2}<|eos|>"))
    args = (len(p), tok.think_start_id, tok.think_end_id)
    assert lab.reward_format(ok, *args) == 1.0
    assert lab.reward_format(rev, *args) == 0.0
    assert lab.reward_format(one, *args) == 0.0
    assert lab.total_reward(1.0, 1.0, 0.1) == pytest.approx(1.1)
    assert lab.total_reward(0.0, 1.0, 1.0, conditional=True) == 0.0
    assert lab.total_reward(0.0, 1.0, 1.0, conditional=False) == 1.0


def test_dr_grpo_and_aggregation():
    adv = lab.dr_grpo_advantages(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(adv, torch.tensor([[0.75, -0.25, -0.25, -0.25]]))
    per = torch.tensor([[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    assert lab.aggregate_token_loss(per, mask, "seq_mean_token_mean").item() == pytest.approx(1.5)
    assert lab.aggregate_token_loss(per, mask, "token_mean").item() == pytest.approx(10 / 6)
    assert lab.aggregate_token_loss(per, mask, "seq_mean_token_sum_norm", 8).item() == pytest.approx(10 / 16)


def test_zero_variance():
    r = torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    assert lab.zero_variance_keep(r).tolist() == [False, True, False]
    assert lab.expected_wasted_fraction(0.5, 8) == pytest.approx(2 / 256)


# ---------------------------------------------------------------- library checks (llmsys.rl)
def test_library_token_loss_matches_lab_for_one_token_sequences():
    from llmsys.rl.losses import token_clipped_loss
    new = torch.tensor([[0.5], [-0.5], [0.05]], requires_grad=True)
    old = torch.zeros(3, 1)
    adv = torch.tensor([1.0, -1.0, 1.0])
    lib, _ = token_clipped_loss(new, old, adv, torch.ones(3, 1), 0.2, 0.2)
    ref, _, _ = lab.clipped_policy_loss(new.detach().squeeze(1), old.squeeze(1), adv)
    assert abs(lib.item() - ref.item()) < 1e-6


@pytest.mark.parametrize("variant", ["baseline", "clip_reuse4", "kl_k3", "kl_seq_surrogate",
                                     "format_cond", "format_gdpo", "dapo", "dr_grpo", "gspo", "cispo"])
def test_trainer_variants_smoke(variant):
    import importlib.util, os
    from llmsys.rl.toy import make_examples, render_prompt
    from llmsys.rl.trainer import GRPOConfig, GRPOTrainer
    spec = importlib.util.spec_from_file_location(
        "lab30_run", os.path.join(os.path.dirname(__file__), "run.py"))
    run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
    torch.manual_seed(0)
    tok = ToyTokenizer()
    model = Qwen3Model(toy_model_config(tok)).eval()
    data, _ = make_examples(0, holdout=0)
    kw = dict(prompts_per_step=2, num_rollouts=3, max_new_tokens=5)
    kw.update(run.VARIANTS[variant])
    tr = GRPOTrainer(model, tok, render_prompt, GRPOConfig(**kw))
    row = tr.step(data[:2])
    for k in ("loss", "reward_avg", "adv_std", "entropy_avg", "zero_var_frac", "t_rollout"):
        assert k in row and row[k] == row[k]  # present and not NaN
