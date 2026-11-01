import os
import tempfile

import torch

from _labutil import load_impl

lab = load_impl(__file__)

CORPUS = (
    "Kernels that move fewer bytes run faster. A fused kernel keeps data on chip, "
    "so the kernel avoids a round trip to memory. Memory bandwidth, not arithmetic, "
    "often bounds the kernel. " * 8
)


def test_split_text_keeps_punctuation():
    toks = lab.split_text('Hello, world -- is this "it"?')
    assert toks == ["Hello", ",", "world", "--", "is", "this", '"', "it", '"', "?"]


def test_word_tokenizer_roundtrip_and_unk():
    vocab = lab.build_vocab(lab.split_text("the cat sat on the mat."))
    assert vocab["<|endoftext|>"] == len(vocab) - 2 and vocab["<|unk|>"] == len(vocab) - 1
    tok = lab.WordTokenizer(vocab)
    assert tok.decode(tok.encode("the cat sat.")) == "the cat sat."
    ids = tok.encode("the dog sat.")
    assert vocab["<|unk|>"] in ids


def test_pair_counts_and_merge():
    words = {(1, 2, 1, 2): 3, (2, 1): 1}
    c = lab.get_pair_counts(words)
    assert c[(1, 2)] == 6 and c[(2, 1)] == 4
    assert lab.merge_pair([1, 2, 1, 2, 3], (1, 2), 9) == [9, 9, 3]
    assert lab.merge_pair([1, 1, 1], (1, 1), 7) == [7, 1]


def test_bpe_roundtrip_and_compression():
    merges = lab.train_bpe(CORPUS, 320)
    assert len(merges) == 64 and min(merges.values()) == 256
    s = "Fused kernels avoid memory round trips — héllo!"
    ids = lab.bpe_encode(s, merges)
    assert lab.bpe_decode(ids, merges) == s
    corpus_ids = lab.bpe_encode(CORPUS, merges)
    assert lab.compression_ratio(CORPUS, corpus_ids) > 2.0


def test_fast_training_matches_naive():
    assert lab.train_bpe_fast(CORPUS, 330) == lab.train_bpe(CORPUS, 330)


def test_sliding_windows():
    x, y = lab.sliding_windows(list(range(20)), max_length=4, stride=2)
    assert x.shape == y.shape and x.shape[1] == 4
    assert x[1].tolist() == [2, 3, 4, 5] and y[1].tolist() == [3, 4, 5, 6]
    assert int(y.max()) <= 19
    assert torch.equal(x[:, 1:], y[:, :-1])  # the shift invariant


def test_pack_and_shard_roundtrip():
    stream = lab.pack_documents([[1, 2, 3], [4, 5]], eot_id=0)
    assert stream == [1, 2, 3, 0, 4, 5, 0]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "shard.bin")
        ids = list(range(1, 42))
        nbytes = lab.write_shard(ids, path, vocab_size=50257)
        assert nbytes == 2 * len(ids)
        ds = lab.ShardDataset(path, block_size=8, vocab_size=50257)
        assert len(ds) == 5
        x, y = ds[2]
        assert x.tolist() == list(range(17, 25)) and y.tolist() == list(range(18, 26))
        assert x.dtype == torch.long


def test_embeddings():
    torch.manual_seed(0)
    emb = lab.InputEmbedding(vocab_size=10, context_length=6, emb_dim=3)
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = emb(ids)
    assert out.shape == (2, 3, 3)
    assert torch.allclose(out[0, 1], emb.tok.weight[2] + emb.pos.weight[1])
    assert torch.allclose(lab.embedding_as_matmul(ids, emb.tok.weight), emb.tok(ids))
