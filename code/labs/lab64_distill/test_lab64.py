import math

import pytest
import torch

from _labutil import load_impl
from llmsys.generate import sequence_logprob
from llmsys.qwen3 import QWEN3_CONFIG_TINY, Qwen3Model
from llmsys.utils import set_seed

lab = load_impl(__file__)


@pytest.fixture(scope="module")
def tok():
    return lab.CharTokenizer()


def _examples(tok, n=32, use_think=True, seed=0):
    recs = lab.make_records(n, digits=2, seed=seed)
    ex, skipped = lab.build_examples(recs, tok.encode_chat_prompt, tok.encode, tok.eos_id,
                                     use_think=use_think, render=lab.render_short_prompt)
    return recs, ex, skipped


def _tiny(vocab, **kw):
    cfg = dict(QWEN3_CONFIG_TINY, vocab_size=vocab, **kw)
    return Qwen3Model(cfg)


# ---------------------------------------------------------------- Task 1: data formatting
def test_format_and_build(tok):
    recs, ex, skipped = _examples(tok, 8)
    assert skipped == 0 and len(ex) == 8
    e, r = ex[0], recs[0]
    prompt = tok.encode_chat_prompt(lab.render_short_prompt(r["problem"]))
    assert e["prompt_len"] == len(prompt)
    assert e["token_ids"][: e["prompt_len"]] == prompt
    assert e["token_ids"][-1] == tok.eos_id
    text = tok.decode(e["token_ids"][e["prompt_len"]:-1])
    assert text.startswith("<think>") and "</think>" in text and r["gtruth"] in text
    _, ex2, _ = _examples(tok, 8, use_think=False)
    assert "<think>" not in tok.decode(ex2[0]["token_ids"])


def test_bad_records_are_skipped(tok):
    recs = lab.make_records(3)
    recs[1]["answer"] = ""
    del recs[2]["problem"]
    ex, skipped = lab.build_examples(recs, tok.encode_chat_prompt, tok.encode, tok.eos_id,
                                     render=lab.render_short_prompt)
    assert len(ex) == 1 and skipped == 2


# ---------------------------------------------------------------- Task 2: filter / split
def test_stats_filter_split(tok):
    _, ex, _ = _examples(tok, 40)
    st = lab.length_stats(ex)
    assert st["min"] <= st["mean"] <= st["max"] and st["n"] == 40
    assert len(ex[st["argmax"]]["token_ids"]) == st["max"]
    short = lab.filter_by_max_len(ex, st["p50"])
    assert all(len(e["token_ids"]) <= st["p50"] for e in short) and len(short) < 40
    tr, va = lab.split_train_val(ex, 5, seed=1)
    tr2, va2 = lab.split_train_val(ex, 5, seed=1)
    assert len(va) == 5 and len(tr) == 35 and va == va2


# ---------------------------------------------------------------- Task 3: losses
def test_answer_only_loss_equals_mean_neg_logprob(tok):
    set_seed(0)
    m = _tiny(tok.vocab_size).eval()
    _, ex, _ = _examples(tok, 2)
    e = ex[0]
    ids = torch.tensor(e["token_ids"])
    p = e["prompt_len"]
    lp = sequence_logprob(m, ids[:p], ids[p:])
    n = len(ids) - p
    with torch.no_grad():
        loss = lab.answer_only_loss(m, e)
    assert torch.allclose(loss, -lp / n, atol=1e-5)


def test_batched_losses(tok):
    set_seed(0)
    m = _tiny(tok.vocab_size).eval()
    _, ex, _ = _examples(tok, 6)
    with torch.no_grad():
        per = [lab.answer_only_loss(m, e) for e in ex]
        seq = lab.batch_answer_only_loss(m, ex, tok.pad_id, reduction="sequence")
        tokl = lab.batch_answer_only_loss(m, ex, tok.pad_id, reduction="token")
    assert torch.allclose(seq, torch.stack(per).mean(), atol=1e-5)
    n = torch.tensor([len(e["token_ids"]) - e["prompt_len"] for e in ex], dtype=torch.float)
    assert torch.allclose(tokl, (torch.stack(per) * n).sum() / n.sum(), atol=1e-5)
    inp, tgt, n_ans = lab.collate_answer_only(ex, tok.pad_id)
    assert n_ans == int(n.sum()) and (tgt == lab.IGNORE).any()


def test_chunked_ce():
    set_seed(0)
    head = torch.nn.Linear(16, 50, bias=False)
    h = torch.randn(37, 16)
    t = torch.randint(0, 50, (37,))
    t[:5] = lab.IGNORE
    full = torch.nn.functional.cross_entropy(head(h), t, ignore_index=lab.IGNORE)
    assert torch.allclose(lab.chunked_cross_entropy(h, head, t, chunk=8), full, atol=1e-5)


def test_kd_losses():
    set_seed(0)
    s, t = torch.randn(3, 7, 20), torch.randn(3, 7, 20)
    assert lab.kd_loss(t, t).abs() < 1e-6
    assert lab.kd_loss(s, t) > 0 and lab.kd_loss(s, t, direction="reverse") > 0
    assert not torch.allclose(lab.kd_loss(s, t), lab.kd_loss(s, t, direction="reverse"))
    v, i = lab.topk_teacher(t, k=20)              # k = V -> exact (up to fp16 storage)
    assert torch.allclose(lab.sparse_kd_loss(s, v, i), lab.kd_loss(s, t), atol=2e-3)
    v4, i4 = lab.topk_teacher(t, k=4)
    assert v4.shape == (3, 7, 4) and v4.dtype == torch.float16


# ---------------------------------------------------------------- Task 4: training
def test_hard_distillation_reduces_loss(tok):
    set_seed(0)
    _, ex, _ = _examples(tok, 48, seed=3)
    tr, va = lab.split_train_val(ex, 8)
    m = _tiny(tok.vocab_size)
    before = sum(lab.answer_only_loss(m, e).item() for e in va) / len(va)
    rows = lab.train_distillation(m, tr, va, epochs=4, lr=3e-3, batch_size=8, pad_id=tok.pad_id,
                                  log_every=5, log=lambda *_: None)
    after = rows[-1]["val_loss"]
    assert after < 0.6 * before, (before, after)


def test_soft_distillation_reduces_kl():
    set_seed(0)
    V = 64
    teacher = _tiny(V, emb_dim=96, n_heads=4, head_dim=24, hidden_dim=192).eval()
    student = _tiny(V)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-3)
    x = torch.randint(0, V, (16, 12))
    with torch.no_grad():
        tl = teacher(x)
        tl = 3.0 * (tl - tl.mean(-1, keepdim=True)) / tl.std()   # a peaked "trained" teacher
    first = None
    for _ in range(40):
        loss = lab.kd_loss(student(x), tl, temperature=1.0)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.6 * first


# ---------------------------------------------------------------- Task 5: systems helpers
def test_systems_helpers(tok):
    recs = lab.make_records(10, digits=1, seed=1) + lab.make_records(10, digits=6, seed=2)
    ex, _ = lab.build_examples(recs, tok.encode_chat_prompt, tok.encode, tok.eos_id,
                               render=lab.render_short_prompt)
    longest = max(len(e["token_ids"]) for e in ex)
    rep = lab.padding_waste(ex, batch_size=4, max_tokens=2 * longest)
    assert 0 < rep["padded"] <= rep["sorted_padded"] + 1e-9 <= 1.0 + 1e-9
    assert rep["packed"] > rep["padded"]
    full = lab.logit_storage_bytes(3000, 151_936)
    topk = lab.logit_storage_bytes(3000, 151_936, k=32)
    assert full == 3000 * 151_936 * 2 and topk < full / 100
