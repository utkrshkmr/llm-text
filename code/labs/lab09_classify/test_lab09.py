import torch

from _labutil import load_impl
from llmsys.gpt import GPT_CONFIG_TINY, GPTModel
from llmsys.utils import set_seed

lab = load_impl(__file__)


def test_balance_and_split():
    labels = [0] * 80 + [1] * 20
    keep = lab.balance_by_undersampling(labels, seed=1)
    kept = [labels[i] for i in keep]
    assert kept.count(0) == kept.count(1) == 20
    tr, va, te = lab.split_indices(40, 0.7, 0.1, seed=1)
    assert (len(tr), len(va), len(te)) == (28, 4, 8)
    assert sorted(tr + va + te) == list(range(40))


def test_pad_batch_sides():
    ids, lengths = lab.pad_batch([[1, 2, 3], [4]], pad_id=0)
    assert ids.tolist() == [[1, 2, 3], [4, 0, 0]] and lengths.tolist() == [3, 1]
    ids, _ = lab.pad_batch([[1, 2, 3], [4]], pad_id=0, side="left")
    assert ids.tolist() == [[1, 2, 3], [0, 0, 4]]
    ids, lengths = lab.pad_batch([[1, 2, 3], [4]], pad_id=0, max_length=2)
    assert ids.tolist() == [[1, 2], [4, 0]] and lengths.tolist() == [2, 1]


def test_head_and_freezing():
    set_seed(0)
    m = lab.add_classification_head(GPTModel(GPT_CONFIG_TINY), num_classes=2, trainable_blocks=1)
    assert m.out_head.out_features == 2 and m.out_head.weight.requires_grad
    assert not m.tok_emb.weight.requires_grad
    assert not any(p.requires_grad for p in m.trf_blocks[0].parameters())
    assert all(p.requires_grad for p in m.trf_blocks[-1].parameters())
    assert m.final_norm.scale.requires_grad


def test_select_last_real_token():
    logits = torch.arange(2 * 4 * 2, dtype=torch.float).view(2, 4, 2)
    out = lab.select_class_logits(logits, torch.tensor([4, 2]))
    assert out.tolist() == [logits[0, 3].tolist(), logits[1, 1].tolist()]
    assert torch.equal(lab.select_class_logits(logits, None), logits[:, -1])


def test_right_padding_invariance():
    """With last-real-token pooling the prediction must not depend on how much padding follows."""
    set_seed(0)
    m = lab.add_classification_head(GPTModel(GPT_CONFIG_TINY), 2).eval()
    seq = [5, 6, 7, 8]
    a, la = lab.pad_batch([seq], 0)
    b, lb = lab.pad_batch([seq, seq + [9] * 10], 0)
    ya = lab.select_class_logits(m(a), la)[0]
    yb = lab.select_class_logits(m(b), lb)[0]
    assert torch.allclose(ya, yb, atol=1e-5)


def test_bucketing_reduces_waste():
    seqs, labels = lab.make_synthetic_spam(400, seed=3)
    lengths = [len(s) for s in seqs]
    rand_batches = [list(range(i, i + 16)) for i in range(0, 400, 16)]
    buck = lab.length_bucketed_batches(lengths, 16, seed=0)
    assert sorted(i for b in buck for i in b) == list(range(400))
    w_rand = lab.padding_stats(lengths, rand_batches)["waste"]
    w_buck = lab.padding_stats(lengths, buck)["waste"]
    assert 0 <= w_buck < w_rand / 3


def test_training_learns_spam():
    set_seed(0)
    seqs, labels = lab.make_synthetic_spam(500, spam_frac=0.4, seed=0)
    train, val, test = lab.make_loaders(seqs, labels, batch_size=16, seed=0, bucketed=True)
    model = lab.add_classification_head(GPTModel(GPT_CONFIG_TINY), 2, trainable_blocks=2)
    for p in model.parameters():
        p.requires_grad = True
    lab.train_classifier(model, train, val, epochs=4, lr=2e-3, log=None)
    assert lab.accuracy_loader(model, test) > 0.8
