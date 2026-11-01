import math
import warnings

import pytest
import torch

from _labutil import load_impl
from llmsys import alignment as A
from llmsys import vlm as V

warnings.filterwarnings("ignore")
lab = load_impl(__file__)


def enc(s):
    return [ord(c) % 200 for c in s]


DOCS = [{"prompt": "hi", "response": "yo"}, {"prompt": "abc", "response": "de"},
        {"prompt": "q?", "response": "answer!"}] * 4
TEMPLATE = "Q:{prompt} A:{response}"


def test_mixture_weights():
    sizes = {"chat": 900, "code": 90, "safety": 10}
    p1 = lab.mixture_weights(sizes, alpha=1.0)
    assert abs(p1["chat"] - 0.9) < 1e-9 and abs(sum(p1.values()) - 1) < 1e-9
    p0 = lab.mixture_weights(sizes, alpha=0.0)
    assert all(abs(v - 1 / 3) < 1e-9 for v in p0.values())
    ph = lab.mixture_weights(sizes, alpha=0.5)
    ref = A.mixture_weights(sizes, alpha=0.5)
    assert all(abs(ph[k] - ref[k]) < 1e-9 for k in sizes)
    assert ph["safety"] > p1["safety"]          # small sources are up-weighted


def test_pack_stream_matches_library_and_masks_prompts():
    rows = list(lab.pack_stream(DOCS, enc, 6, 255, TEMPLATE, mask_prompt=True))
    ref = list(A.PackedSFTStream(DOCS, enc, 6, 255, TEMPLATE, mask_prompt=True))
    assert len(rows) == len(ref) > 0
    for r, q in zip(rows, ref):
        for k in ("input_ids", "labels", "doc_ids"):
            assert torch.equal(r[k], q[k]), k
    # every non-masked label is a response token or EOT
    stream = []
    for d in DOCS:
        stream += enc("Q:" + d["prompt"] + " A:") + enc(d["response"]) + [255]
    total = sum(len(r["input_ids"]) for r in rows)
    assert total == (len(stream) - 1) // 6 * 6
    # rows overlap by exactly one token: inputs concatenate back into the stream prefix
    flat = torch.cat([r["input_ids"] for r in rows]).tolist()
    assert flat == stream[: len(flat)]


def test_pack_stream_unmasked_labels_are_shifted_inputs():
    rows = list(lab.pack_stream(DOCS, enc, 5, 255, TEMPLATE, mask_prompt=False))
    for a, b in zip(rows, rows[1:]):
        assert torch.equal(a["labels"][:-1], a["input_ids"][1:])
        assert a["labels"][-1] == b["input_ids"][0]


def test_stream_sharding_and_resume():
    full = list(A.PackedSFTStream(DOCS, enc, 4, 255, TEMPLATE))
    r0 = list(A.PackedSFTStream(DOCS, enc, 4, 255, TEMPLATE, rank=0, world=2))
    r1 = list(A.PackedSFTStream(DOCS, enc, 4, 255, TEMPLATE, rank=1, world=2))
    assert len(r0) + len(r1) == len(full)
    assert torch.equal(r1[0]["input_ids"], full[1]["input_ids"])
    s = A.PackedSFTStream(DOCS, enc, 4, 255, TEMPLATE)
    it = iter(s)
    for _ in range(3):
        next(it)
    resumed = list(A.PackedSFTStream(DOCS, enc, 4, 255, TEMPLATE, **A.stream_state(s)))
    assert torch.equal(resumed[0]["input_ids"], full[3]["input_ids"])


def test_debiased_preference_cancels_position_bias():
    # a judge that always adds +0.2 to whichever answer is shown first
    true_p = 0.6
    def judge(instr, first, second):
        base = true_p if first == "M" else 1 - true_p
        return min(1.0, base + 0.2)
    assert abs(judge("x", "M", "R") - 0.8) < 1e-9             # biased raw read-out
    assert abs(lab.debiased_preference(judge, "x", "M", "R") - true_p) < 1e-9


def test_safety_report_two_sided():
    recs = ([{"kind": "harmful", "unsafe": i < 3, "refused": i >= 3} for i in range(20)]
            + [{"kind": "benign", "unsafe": False, "refused": i < 5} for i in range(25)])
    r = lab.safety_report(recs)
    assert abs(r["unsafe_rate"] - 0.15) < 1e-9 and abs(r["over_refusal_rate"] - 0.2) < 1e-9
    lo, hi = r["unsafe_ci95"]
    assert lo < 0.15 < hi
    ref = A.safety_report(recs)
    assert abs(ref["over_refusal_rate"] - r["over_refusal_rate"]) < 1e-12


def test_length_controlled_win_rate_removes_length_effect():
    torch.manual_seed(0)
    n = 400
    dlen = torch.randn(n) * 200
    # the judge prefers longer answers; quality alone would give 50%
    p = torch.sigmoid(0.0 + 2.0 * torch.tanh(dlen / 200))
    prefs = torch.bernoulli(p).tolist()
    lm = (500 + dlen).round().int().tolist()
    lr = [500] * n
    out = A.length_controlled_win_rate(prefs, lm, lr, scale=200.0)
    assert out["theta_len"] > 1.0
    assert abs(out["lc_win_rate"] - 0.5) < 0.08


def test_splice_and_errors():
    ids = torch.tensor([[1, 3, 3, 5], [3, 3, 6, 7]])
    emb = torch.zeros(2, 4, 2)
    feats = torch.arange(8.0).view(2, 2, 2)
    out = lab.splice_image_embeddings(emb, ids, feats, 3)
    assert torch.equal(out, V.splice_image_embeddings(emb, ids, feats, 3))
    assert torch.equal(out[0, 1], torch.tensor([0.0, 1.0]))
    assert torch.equal(out[1, 1], torch.tensor([6.0, 7.0]))
    with pytest.raises(ValueError):
        lab.splice_image_embeddings(emb, ids, feats[:1], 3)


def test_contrastive_losses():
    torch.manual_seed(0)
    i, t = torch.randn(8, 16), torch.randn(8, 16)
    s, b = torch.tensor(math.log(10.0)), torch.tensor(-5.0)
    assert torch.allclose(lab.clip_loss(i, t, s), V.clip_loss(i, t, s), atol=1e-6)
    assert torch.allclose(lab.siglip_loss(i, t, s, b), V.siglip_loss(i, t, s, b), atol=1e-6)
    assert torch.allclose(V.siglip_loss_chunked(i, t, s, b, 4), V.siglip_loss(i, t, s, b),
                          atol=1e-5)
    # perfectly aligned embeddings give a smaller loss than random ones
    assert lab.clip_loss(i, i, s) < lab.clip_loss(i, t, s)


def test_anyres_tokens():
    assert lab.anyres_tokens(336, 336) == 2 * 576           # one tile + overview
    assert lab.anyres_tokens(672, 1008) == 7 * 576          # 2x3 grid + overview
    assert lab.anyres_tokens(2000, 2000) == V.anyres_tokens(2000, 2000)["tokens"]
    assert lab.anyres_tokens(2000, 2000) <= 7 * 576         # capped


def test_multimodal_block_hashes():
    IMG = 9
    toks = [1, 2, IMG, IMG, IMG, IMG, 4, 5]
    a = lab.multimodal_block_hashes(toks, 4, IMG, ["catA"])
    b = lab.multimodal_block_hashes(toks, 4, IMG, ["dogB"])
    assert a[0] != b[0]                         # same ids, different image: no false hit
    assert a == V.multimodal_block_hashes(toks, 4, IMG, ["catA"])
    c = lab.multimodal_block_hashes([1, 2, 3, 4] + toks[:4], 4, IMG, ["catA"])
    d = lab.multimodal_block_hashes([1, 2, 3, 4] + toks[:4], 4, IMG, ["catA"])
    assert c == d and len(c) == 2


def test_image_embedding_cache():
    calls = []
    def enc_fn(x):
        calls.append(1)
        return torch.ones(x.shape[0], 4, 8)     # 4 tokens x 8 dims x 4 bytes = 128 B
    cache = lab.ImageEmbeddingCache(capacity_bytes=256)
    imgs = [torch.full((3, 2, 2), float(k)) for k in range(3)]
    cache.get_or_compute(imgs[0], enc_fn)
    cache.get_or_compute(imgs[0], enc_fn)
    assert cache.hits == 1 and len(calls) == 1
    cache.get_or_compute(imgs[1], enc_fn)
    cache.get_or_compute(imgs[2], enc_fn)       # evicts imgs[0] (LRU)
    assert cache.used <= 256
    cache.get_or_compute(imgs[0], enc_fn)
    assert len(calls) == 4


def test_tiny_vlm_learns_and_stage1_freezes_lm():
    torch.set_num_threads(1)
    vlm = lab.build_tiny_vlm(seed=0)
    lm_before = [p.clone() for p in vlm.lm.parameters()]
    vis_before = [p.clone() for p in vlm.vision.parameters()]
    lab.train_vlm(vlm, steps=3, stage=1)
    assert all(torch.equal(a, b) for a, b in zip(lm_before, vlm.lm.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(vis_before, vlm.vision.parameters()))
    vlm2 = lab.build_tiny_vlm(seed=0)
    losses = lab.train_vlm(vlm2, steps=40, stage=2)
    first, last = sum(losses[:3]) / 3, sum(losses[-5:]) / 5
    assert last < 0.6 * first
