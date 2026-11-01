import math

import pytest
import torch

from _labutil import load_impl
from llmsys.qwen3 import Qwen3Model
from llmsys.rl.toy import ToyTokenizer, make_examples, render_prompt, toy_model_config

lab = load_impl(__file__)
torch.set_num_threads(1)


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    tok = ToyTokenizer()
    model = Qwen3Model(toy_model_config(tok)).eval()
    train, test = make_examples(0)
    return tok, model, train


def test_reward_rlvr():
    assert lab.reward_rlvr(r"so \boxed{12}", "12") == 1.0
    assert lab.reward_rlvr(r"answer: 12", "12") == 0.0          # not boxed
    assert lab.reward_rlvr(r"\boxed{21}", "12") == 0.0          # wrong
    assert lab.reward_rlvr(r"\boxed{3} then \boxed{12.0}", "12") == 1.0  # last box, 12.0 == 12
    assert lab.reward_rlvr(r"\boxed{\frac{3}{2}}", "3/2") == 1.0


def test_reward_partial():
    assert lab.reward_partial(r"\boxed{7}", "7") == 1.0
    assert lab.reward_partial("the answer is 7", "7") == 0.5
    assert lab.reward_partial("the answer is 8", "7") == 0.0


def test_advantages():
    adv = lab.compute_advantages(torch.tensor([1.0, 1.0, 0.0, 0.0]))
    s = math.sqrt(1 / 3)  # unbiased std of [1,1,0,0]
    assert torch.allclose(adv, torch.tensor([0.5, 0.5, -0.5, -0.5]) / (s + 1e-4), atol=1e-4)
    assert torch.allclose(lab.compute_advantages(torch.ones(4)), torch.zeros(4))
    assert abs(adv.mean().item()) < 1e-6


def test_sequence_logprob_matches_manual(setup):
    tok, model, _ = setup
    ids = torch.tensor(tok.encode("Q:1+2=\nA:\\boxed{3}<|eos|>"))
    plen = len(tok.encode("Q:1+2=\nA:"))
    lp = lab.sequence_logprob(model, ids, plen)
    assert lp.requires_grad
    logits = model(ids[None])[0]
    manual = sum(torch.log_softmax(logits[t - 1], -1)[ids[t]] for t in range(plen, len(ids)))
    assert torch.allclose(lp, manual, atol=1e-5)


def test_pg_loss_zero_when_all_equal():
    logps = torch.tensor([-3.0, -5.0, -7.0], requires_grad=True)
    loss = lab.policy_gradient_loss(torch.zeros(3), logps)
    loss.backward()
    assert loss.item() == 0.0 and torch.all(logps.grad == 0)


def test_pg_loss_value_and_direction():
    adv = torch.tensor([1.0, -1.0])
    logps = torch.tensor([-2.0, -4.0], requires_grad=True)
    loss = lab.policy_gradient_loss(adv, logps)
    assert abs(loss.item() - (-(1 * -2.0 + -1 * -4.0) / 2)) < 1e-6
    loss.backward()
    # minimising the loss must raise logp of the positive-advantage rollout
    assert logps.grad[0] < 0 < logps.grad[1]


def test_sample_response(setup):
    tok, model, train = setup
    g = torch.Generator().manual_seed(1)
    ids, plen, text = lab.sample_response(model, tok, render_prompt(train[0]), 8, 1.0, 1.0, g)
    assert plen == len(tok.encode(render_prompt(train[0])))
    assert 1 <= ids.numel() - plen <= 8
    assert tok.decode(ids[plen:].tolist()) == text
    g2 = torch.Generator().manual_seed(1)
    ids2, _, _ = lab.sample_response(model, tok, render_prompt(train[0]), 8, 1.0, 1.0, g2)
    assert torch.equal(ids, ids2)


def test_compute_grpo_loss_and_update_direction(setup):
    tok, model, train = setup
    m = Qwen3Model(toy_model_config(tok))
    m.load_state_dict(model.state_dict())
    out = lab.compute_grpo_loss(m, tok, train[0], render_prompt, num_rollouts=3,
                                max_new_tokens=6, generator=torch.Generator().manual_seed(0))
    for k in ("loss", "loss_tensor", "rewards", "advantages", "samples"):
        assert k in out
    assert len(out["samples"]) == 3 and out["loss_tensor"].requires_grad


def test_one_step_raises_logprob_of_rewarded_rollout(setup):
    """A gradient step on a fixed group must favour the positive-advantage completion."""
    tok, model, _ = setup
    m = Qwen3Model(toy_model_config(tok))
    m.load_state_dict(model.state_dict())
    prompt = "Q:4+4=\nA:"
    plen = len(tok.encode(prompt))
    good = torch.tensor(tok.encode(prompt + "\\boxed{8}<|eos|>"))
    bad = torch.tensor(tok.encode(prompt + "9<|eos|>"))
    adv = lab.compute_advantages(torch.tensor([1.0, 0.0]))
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    before = lab.sequence_logprob(m, good, plen).item()
    loss = lab.policy_gradient_loss(adv, torch.stack([lab.sequence_logprob(m, good, plen),
                                                       lab.sequence_logprob(m, bad, plen)]))
    loss.backward()
    opt.step()
    assert lab.sequence_logprob(m, good, plen).item() > before


def test_train_loop_smoke(setup, tmp_path):
    tok, model, train = setup
    m = Qwen3Model(toy_model_config(tok))
    m.load_state_dict(model.state_dict())
    hist = lab.train_rlvr_grpo(m, tok, train, render_prompt, steps=2, num_rollouts=2,
                               max_new_tokens=6, checkpoint_every=2,
                               checkpoint_dir=str(tmp_path), csv_log_path=str(tmp_path / "m.csv"),
                               log_every=0)
    assert len(hist) == 2 and {"loss", "reward_avg", "avg_response_len"} <= set(hist[0])
    assert (tmp_path / "grpo-step00002.pt").exists() and (tmp_path / "m.csv").exists()


def test_time_breakdown(setup):
    tok, model, train = setup
    t = lab.time_breakdown(model, tok, train[0], render_prompt, num_rollouts=2, max_new_tokens=4)
    assert set(t) == {"sample", "logprob", "backward"} and all(v >= 0 for v in t.values())
