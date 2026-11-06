"""Lab 14 tests: objectives, encoder, cross-attention cache, embeddings, ViT, cost model."""
import math

import pytest
import torch
import torch.nn.functional as F

from _labutil import load_impl
from llmsys import encdec as ref_ed
from llmsys import encoder as ref
from llmsys import vit as ref_vit

lab = load_impl(__file__)
IGN = ref.IGNORE_INDEX
CFG = dict(ref.ENCODER_CONFIG_TINY)
ED_CFG = dict(ref_ed.ENCDEC_CONFIG_TINY)


def tiny_encoder_pair():
    """A library encoder and a lab encoder holding the same weights."""
    torch.manual_seed(0)
    r = ref.EncoderModel(CFG).eval()
    m = lab.EncoderModel(CFG).eval()
    m.load_state_dict(r.state_dict())
    return r, m


# ---- Task 1
def test_mlm_matches_reference():
    ids = torch.randint(5, 500, (8, 40), generator=torch.Generator().manual_seed(1))
    a = lab.mlm_corrupt(ids, 3, 512, (1, 2), generator=torch.Generator().manual_seed(7))
    b = ref.mlm_corrupt(ids, 3, 512, (1, 2), generator=torch.Generator().manual_seed(7))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_mlm_80_10_10():
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(5, 500, (200, 500), generator=g)
    inputs, labels = lab.mlm_corrupt(ids, 3, 512, mask_prob=0.15, generator=g)
    selected = labels != IGN
    assert abs(selected.float().mean() - 0.15) < 0.01
    assert torch.equal(labels[selected], ids[selected])          # the label is always the original
    masked = (inputs == 3) & selected
    same = (inputs == ids) & selected
    assert abs(masked.sum() / selected.sum() - 0.8) < 0.02
    assert abs(same.sum() / selected.sum() - 0.1) < 0.02         # plus a tiny random-collision rate
    assert torch.equal(inputs[~selected], ids[~selected])        # nothing else is touched


def test_mlm_never_selects_specials():
    ids = torch.randint(5, 500, (4, 64), generator=torch.Generator().manual_seed(2))
    ids[:, 0], ids[:, -1] = 1, 2
    _, labels = lab.mlm_corrupt(ids, 3, 512, (1, 2), 0.9, torch.Generator().manual_seed(3))
    assert (labels[:, 0] == IGN).all() and (labels[:, -1] == IGN).all()


# ---- Task 2
def test_span_corrupt_worked_example():
    ids = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17])
    noise = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1], dtype=torch.bool)
    enc, tgt = lab.span_corrupt_from_mask(ids, noise, 100, eos_id=1)
    assert enc.tolist() == [10, 100, 13, 14, 101, 16, 102]
    assert tgt.tolist() == [100, 11, 12, 101, 15, 102, 17, 103, 1]


def test_span_corrupt_matches_reference_and_inverts():
    ids = torch.arange(200, 260)
    a = lab.span_corrupt(ids, 500, 0.3, 3.0, generator=torch.Generator().manual_seed(5))
    b = ref.span_corrupt(ids, 500, 0.3, 3.0, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    enc, tgt = a
    assert enc[0] < 500                                          # never starts with a sentinel
    spans, cur = {}, None                                        # undo the corruption
    for t in tgt.tolist():
        if t >= 500:
            cur = t
            spans[cur] = []
        else:
            spans[cur].append(t)
    rebuilt = []
    for t in enc.tolist():
        rebuilt.extend(spans[t] if t >= 500 else [t])
    assert rebuilt == ids.tolist()


# ---- Task 3
def test_prefix_mask_matches_reference():
    assert torch.equal(lab.prefix_lm_mask(3, 7), ref.prefix_lm_mask(3, 7))
    m = lab.prefix_lm_mask(3, 6)[0, 0]
    assert not m[0, 2]            # a prefix token sees later prefix tokens
    assert m[1, 3]                # ...but no suffix token
    assert not m[4, 2] and not m[4, 4] and m[4, 5]


def test_prefix_targets_shift():
    ids = torch.arange(10, 16)[None]
    _, labels, _ = lab.prefix_lm_targets(ids, 2)
    assert labels.tolist() == [[IGN, 12, 13, 14, 15, IGN]]


def test_prefix_lm_does_not_leak_the_target():
    r, m = tiny_encoder_pair()
    ids = torch.randint(5, 500, (1, 12), generator=torch.Generator().manual_seed(4))
    ids, labels, mask = lab.prefix_lm_targets(ids, 5)
    for i in torch.nonzero(labels[0] != IGN).flatten().tolist():  # no position sees its own label
        changed = ids.clone()
        changed[0, i + 1] = (changed[0, i + 1] + 1) % 512
        with torch.no_grad():
            a, b = m(ids, attn_mask=mask)[0, i], m(changed, attn_mask=mask)[0, i]
        assert torch.allclose(a, b, atol=1e-6)


# ---- Task 4
def test_encoder_matches_library():
    r, m = tiny_encoder_pair()
    ids = torch.randint(5, 500, (3, 20), generator=torch.Generator().manual_seed(6))
    pad = torch.ones_like(ids, dtype=torch.bool)
    pad[1, 15:] = False
    with torch.no_grad():
        assert torch.allclose(m(ids, pad)[pad], r(ids, pad)[pad], atol=1e-6)


def test_encoder_is_bidirectional():
    _, m = tiny_encoder_pair()
    ids = torch.randint(5, 500, (1, 10), generator=torch.Generator().manual_seed(8))
    changed = ids.clone()
    changed[0, -1] = (changed[0, -1] + 1) % 512
    with torch.no_grad():
        assert not torch.allclose(m(ids)[0, 0], m(changed)[0, 0], atol=1e-6)  # future reaches past


def test_padding_is_invisible():
    _, m = tiny_encoder_pair()
    ids = torch.randint(5, 500, (1, 9), generator=torch.Generator().manual_seed(9))
    padded = torch.cat([ids, torch.zeros(1, 7, dtype=torch.long)], 1)
    pad = torch.tensor([[True] * 9 + [False] * 7])
    with torch.no_grad():
        assert torch.allclose(m(padded, pad)[0, :9], m(ids)[0], atol=1e-5)
        assert torch.isfinite(m(padded, pad)).all()


# ---- Task 5
def ed_pair():
    torch.manual_seed(0)
    model = ref_ed.EncoderDecoder(ED_CFG).eval()
    src = torch.randint(5, 500, (2, 14), generator=torch.Generator().manual_seed(1))
    tgt = torch.randint(5, 500, (2, 9), generator=torch.Generator().manual_seed(2))
    pad = torch.ones_like(src, dtype=torch.bool)
    pad[1, 10:] = False
    return model, src, tgt, pad


def test_cross_attention_matches_library():
    torch.manual_seed(0)
    r = ref_ed.CrossAttention(64, 4).eval()
    m = lab.CrossAttention(64, 4).eval()
    m.load_state_dict(r.state_dict())
    x, mem = torch.randn(2, 5, 64), torch.randn(2, 8, 64)
    pad = torch.ones(2, 8, dtype=torch.bool)
    pad[0, 6:] = False
    with torch.no_grad():
        assert torch.allclose(m(x, mem, pad), r(x, mem, pad), atol=1e-6)
        m.layer_idx = 0
        cache = ref_ed.CrossKVCache(1)
        full = m(x, mem, pad, cache)
        again = m(x, None, pad, cache)                  # second call must not need memory
    assert torch.allclose(full, again, atol=1e-6)


def test_incremental_equals_full_recompute():
    model, src, tgt, pad = ed_pair()
    with torch.no_grad():
        full = model(src, tgt, pad)
        inc = lab.incremental_decode(model, src, tgt, pad)
    assert inc.shape == full.shape and torch.allclose(inc, full, atol=1e-4)


def test_cross_kv_projected_once_and_encoder_runs_once():
    model, src, tgt, pad = ed_pair()
    calls = {"key": 0, "enc": 0}
    model.encoder.register_forward_hook(lambda *a: calls.__setitem__("enc", calls["enc"] + 1))
    for blk in model.dec_blocks:
        blk.cross_att.W_key.register_forward_hook(
            lambda *a: calls.__setitem__("key", calls["key"] + 1))
    with torch.no_grad():
        lab.incremental_decode(model, src, tgt, pad)
    assert calls["enc"] == 1
    assert calls["key"] == model.n_dec_layers       # one per layer, not one per layer per step


def test_cross_cache_does_not_grow_while_self_cache_does():
    from llmsys.kvcache import KVCache
    model, src, tgt, pad = ed_pair()
    mem = model.encode(src, pad)
    sc, cc = KVCache(model.n_dec_layers), ref_ed.CrossKVCache(model.n_dec_layers)
    sizes = []
    with torch.no_grad():
        for t in range(tgt.shape[1]):
            model.decode(tgt[:, t:t + 1], mem if t == 0 else None, pad, sc, cc)
            sizes.append((sc.nbytes(), cc.nbytes()))
    assert len({s[1] for s in sizes}) == 1
    assert sizes[-1][0] == tgt.shape[1] * sizes[0][0]
    d_h = ED_CFG["emb_dim"] // ED_CFG["n_heads"]
    per_tok = 2 * model.n_dec_layers * ED_CFG["n_heads"] * d_h * 4     # fp32 here
    assert sizes[0][1] == per_tok * src.shape[0] * src.shape[1]


# ---- Task 6
def test_pool_modes():
    h = torch.arange(24.0).reshape(2, 4, 3)
    pad = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    for how in ("cls", "mean", "last"):
        assert torch.allclose(lab.pool(h, pad, how), ref.pool(h, pad, how))
    assert torch.allclose(lab.pool(h, pad, "mean")[1], h[1, :2].mean(0))
    assert torch.equal(lab.pool(h, pad, "last")[0], h[0, 2])
    assert torch.equal(lab.pool(h, pad, "last")[1], h[1, 1])


def test_info_nce_formula():
    torch.manual_seed(0)
    q, d = F.normalize(torch.randn(6, 8), dim=-1), F.normalize(torch.randn(6, 8), dim=-1)
    loss, acc = lab.info_nce_loss(q, d, 0.1)
    logits = q @ d.T / 0.1
    manual = -(logits.diag() - torch.logsumexp(logits, 1)).mean()
    assert torch.allclose(loss, manual, atol=1e-5)
    assert 0 <= acc <= 1
    perfect_loss, perfect_acc = lab.info_nce_loss(q, q, 0.01)
    assert perfect_acc == 1.0 and perfect_loss < 0.1


def test_info_nce_hard_negatives_raise_the_loss():
    torch.manual_seed(1)
    q, d = F.normalize(torch.randn(6, 8), dim=-1), F.normalize(torch.randn(6, 8), dim=-1)
    base, _ = lab.info_nce_loss(q, d, 0.1)
    hard, _ = lab.info_nce_loss(q, d, 0.1, hard_negatives=q[:3])  # each very close to its query
    assert hard > base
    assert torch.allclose(hard, ref.info_nce_loss(q, d, 0.1, q[:3])[0], atol=1e-6)


# ---- Task 7
@pytest.mark.parametrize("shape,P", [((2, 3, 16, 16), 4), ((1, 1, 12, 20), 4), ((3, 3, 6, 9), 3)])
def test_patchify_equals_unfold(shape, P):
    torch.manual_seed(0)
    img = torch.randn(shape)
    out = lab.patchify(img, P)
    assert out.shape[1] == ref_vit.n_tokens(shape[2], shape[3], P)
    assert torch.equal(out, F.unfold(img, P, stride=P).transpose(1, 2))
    assert torch.equal(ref_vit.unpatchify(out, P, shape[1], shape[2], shape[3]), img)


def test_vit_forward_matches_library():
    torch.manual_seed(0)
    r = ref_vit.ViTEncoder(ref_vit.VIT_CONFIG_TINY).eval()
    m = lab.ViTEncoder(ref_vit.VIT_CONFIG_TINY).eval()
    m.load_state_dict(r.state_dict())
    img = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        out = m(img)
        assert out.shape == (2, 17, 64)
        assert torch.allclose(out, r(img), atol=1e-6)
        assert m.embed(img).shape == (2, 64)


# ---- Task 8
def test_cost_model():
    assert lab.encoder_flops_per_sequence(1e6, 10, 2, 64) == pytest.approx(
        ref.encoder_flops_per_sequence(1e6, 10, 2, 64))
    assert lab.encdec_step_bytes(1e6, 2, 4, 16, 30, 5, 3) == pytest.approx(
        ref_ed.encdec_step_bytes(1e6, 2, 4, 16, 30, 5, 3))
    args = (170e6, 2.2e10, 312e12, 1.555e12)
    assert lab.batch_for_compute_bound(*args) == ref.batch_for_compute_bound(*args) == 2
    assert lab.batch_for_compute_bound(*args, 66e6) == ref.batch_for_compute_bound(*args, 66e6) == 4
    with pytest.raises(ValueError):
        lab.batch_for_compute_bound(170e6, 2.2e10, 312e12, 1.555e12, 2e9)


# ---- library behaviour the chapter relies on (these exercise llmsys directly)
def test_library_span_mask_density_and_extremes():
    g = torch.Generator().manual_seed(0)
    assert ref.random_spans_noise_mask(100, 0.15, 3.0, g).sum() == 15
    assert ref.random_spans_noise_mask(10, 0.9, 1.0, g).sum() == 9   # more noise than clean tokens
    assert not ref.random_spans_noise_mask(2, 0.5, 3.0, g)[0]
    with pytest.raises(ValueError):
        ref.random_spans_noise_mask(1, 0.5, 3.0, g)


def test_library_mlm_head_scores_only_selected_positions():
    torch.manual_seed(0)
    enc = ref.EncoderModel(CFG)
    head = ref.MLMHead(CFG["emb_dim"], enc.tok_emb)
    assert head.decoder_weight is enc.tok_emb.weight
    ids = torch.randint(5, 500, (2, 16))
    labels = torch.full_like(ids, IGN)
    labels[0, 3], labels[1, 7] = 42, 43
    logits, loss = head(enc(ids), labels)
    assert logits.shape == (2, 512)
    # a small init must start the loss near log(V), not orders of magnitude above it
    assert abs(loss.item() - math.log(512)) < 1.0
    _, empty = head(enc(ids), torch.full_like(ids, IGN))
    assert empty.item() == 0.0


def test_library_packed_equals_padded():
    torch.manual_seed(0)
    enc = ref.EncoderModel(CFG).eval()
    seqs = [torch.randint(5, 500, (n,)) for n in (5, 12, 1, 8)]
    ids, pos, mask, spans = ref.pack_sequences(seqs)
    assert ids.shape == (1, 26) and spans[1] == (5, 17)
    tmax = max(len(s) for s in seqs)
    padded = torch.stack([F.pad(s, (0, tmax - len(s))) for s in seqs])
    pad_mask = torch.stack([torch.arange(tmax) < len(s) for s in seqs])
    with torch.no_grad():
        packed_out = enc(ids, attn_mask=mask, position_ids=pos)[0]
        padded_out = enc(padded, pad_mask)
    for n, (a, b) in enumerate(spans):
        assert torch.allclose(packed_out[a:b], padded_out[n, : b - a], atol=1e-5)


def test_library_length_sorted_batches():
    lengths = [5, 100, 7, 6, 90, 8, 95, 5]
    batches = ref.length_sorted_batches(lengths, max_tokens=200)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    assert all(len(b) * max(lengths[i] for i in b) <= 200 for b in batches)
    naive = [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert ref.padding_waste(lengths, batches) < ref.padding_waste(lengths, naive)
    assert ref.length_sorted_batches([500, 3], 100) == [[1], [0]]    # oversized goes alone
    assert all(len(b) <= 2 for b in ref.length_sorted_batches(lengths, 10_000, max_batch=2))


def test_library_greedy_decode_equals_uncached():
    model, src, _, pad = ed_pair()
    out = ref_ed.greedy_decode(model, src, 1, 2, 6, pad)
    cur = torch.full((2, 1), 1)
    with torch.no_grad():
        for _ in range(out.shape[1]):
            cur = torch.cat([cur, model(src, cur, pad)[:, -1].argmax(-1, keepdim=True)], 1)
    assert torch.equal(out, cur[:, 1:])


def test_library_relative_position_buckets():
    d = torch.tensor([0, -1, -7, -8, -9, -200, 1, 7, 8, 200])
    b = ref_ed.relative_position_bucket(d, bidirectional=True, num_buckets=32, max_distance=128)
    assert b[:4].tolist() == [0, 1, 7, 8]                # exact, then logarithmic
    assert b[5] == 15                                    # far past shares the last bucket
    assert b[6] == 17 and b[7] == 23 and b[9] == 31      # future uses the upper half


def test_library_rerank_and_late_interaction():
    torch.manual_seed(0)
    enc = ref.EncoderModel(CFG).eval()
    ids, pad, seg = ref.build_pair_input([5, 6, 7], list(range(10, 90)), 1, 2, 0, 16)
    assert ids.shape == pad.shape == seg.shape == (16,)
    assert ids[:5].tolist() == [1, 5, 6, 7, 2] and ids[-1] == 2 and seg[4] == 0 and seg[5] == 1
    score = ref.CrossEncoderReranker(enc)(ids[None], pad[None], seg[None])
    assert score.shape == (1,)
    q = F.normalize(torch.randn(2, 3, 8), dim=-1)
    d = F.normalize(torch.randn(4, 5, 8), dim=-1)
    s = ref.late_interaction_score(q, d, torch.ones(2, 3, dtype=torch.bool),
                                   torch.ones(4, 5, dtype=torch.bool))
    manual = (q[1] @ d[2].T).max(-1).values.sum()
    assert s.shape == (2, 4) and torch.allclose(s[1, 2], manual, atol=1e-6)


def test_library_vit_resolution_and_cost():
    torch.manual_seed(0)
    vit = ref_vit.ViTEncoder(ref_vit.VIT_CONFIG_TINY).eval()
    with torch.no_grad():
        assert vit(torch.randn(1, 3, 24, 32)).shape == (1, 1 + 6 * 8, 64)
    cfg = dict(img_size=224, patch=16, in_ch=3, emb_dim=768, n_heads=12, n_layers=12)
    assert ref_vit.n_tokens(224, 224, 16) == 196
    assert ref_vit.vit_params_nonembed(cfg) == pytest.approx(85e6, rel=0.02)
    # halving the patch size quadruples the tokens and the GEMM FLOPs
    ratio = ref_vit.vit_flops(dict(cfg, patch=8)) / ref_vit.vit_flops(cfg)
    assert 3.9 < ratio < 5
