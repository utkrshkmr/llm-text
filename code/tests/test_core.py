import math

import torch

from llmsys.data import SlidingWindowDataset
from llmsys.flops import ModelShape, params
from llmsys.generate import generate, generate_batch, generate_simple, sequence_logprob, top_p_filter
from llmsys.gpt import GPT_CONFIG_TINY, GPTModel, gpt2_config
from llmsys.kvcache import KVCache, StaticKVCache
from llmsys.qwen3 import QWEN3_CONFIG_TINY, QWEN3_CONFIGS, Qwen3Model
from llmsys.tokenizer import ByteBPETokenizer, SimpleTokenizer
from llmsys.train import lr_at
from llmsys.utils import count_parameters, set_seed

TEXT = "the quick brown fox jumps over the lazy dog. the dog sleeps. " * 20


def test_bpe_roundtrip():
    tok = ByteBPETokenizer.train(TEXT, vocab_size=300)
    s = "the lazy fox<|endoftext|> héllo"
    ids = tok.encode(s)
    assert tok.decode(ids) == s
    assert len(tok.encode("the quick brown fox")) < len("the quick brown fox".encode())


def test_simple_tokenizer_unk():
    tok = SimpleTokenizer.from_text("Hello, world. Is this-- a test?")
    ids = tok.encode("Hello, unseen world.")
    assert tok.str_to_int["<|unk|>"] in ids


def test_sliding_window():
    ds = SlidingWindowDataset(list(range(20)), max_length=4, stride=4)
    x, y = ds[1]
    assert x.tolist() == [4, 5, 6, 7] and y.tolist() == [5, 6, 7, 8]


def _check_cache_equivalence(model):
    set_seed(0)
    idx = torch.randint(0, model.cfg["vocab_size"], (2, 7))
    full = model(idx)
    cache = KVCache(model.n_layers)
    a = model(idx[:, :4], cache=cache)
    b = model(idx[:, 4:5], cache=cache)
    c = model(idx[:, 5:], cache=cache)
    inc = torch.cat([a, b, c], dim=1)
    assert torch.allclose(full, inc, atol=1e-5)
    st = StaticKVCache(model.n_layers, 2, model.trf_blocks[0].att.__dict__.get("num_kv_groups", None)
                       or model.cfg["n_heads"], 16,
                       getattr(model.trf_blocks[0].att, "head_dim"))
    s1 = model(idx[:, :4], cache=st)
    s2 = model(idx[:, 4:], cache=st)
    assert torch.allclose(full, torch.cat([s1, s2], 1), atol=1e-5)


def test_gpt_cache_equivalence():
    set_seed(0)
    _check_cache_equivalence(GPTModel(GPT_CONFIG_TINY).eval())


def test_qwen3_cache_equivalence():
    set_seed(0)
    _check_cache_equivalence(Qwen3Model(QWEN3_CONFIG_TINY).eval())


def test_sdpa_matches_explicit():
    set_seed(0)
    m1 = Qwen3Model(QWEN3_CONFIG_TINY).eval()
    m2 = Qwen3Model(dict(QWEN3_CONFIG_TINY, use_sdpa=True)).eval()
    m2.load_state_dict(m1.state_dict())
    idx = torch.randint(0, 512, (2, 9))
    assert torch.allclose(m1(idx), m2(idx), atol=1e-5)


def test_generate_cache_matches_nocache():
    set_seed(0)
    m = GPTModel(GPT_CONFIG_TINY).eval()
    idx = torch.randint(0, 512, (1, 5))
    a = generate(m, idx, 10, use_cache=True)
    b = generate(m, idx, 10, use_cache=False)
    c = generate_simple(m, idx, 10, context_size=64)
    assert torch.equal(a, b) and torch.equal(a, c)


def test_batch_generation_matches_single_rope():
    set_seed(0)
    m = Qwen3Model(QWEN3_CONFIG_TINY).eval()
    prompts = [[5, 6, 7, 8, 9], [10, 11], [3, 4, 5]]
    batched = generate_batch(m, prompts, 6, pad_id=0)
    for p, out in zip(prompts, batched):
        single = generate(m, torch.tensor([p]), 6)[0, len(p):].tolist()
        assert single == out


def test_top_p():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    kept = torch.isfinite(top_p_filter(logits, 0.7))
    assert kept.tolist() == [[True, True, False, False]]


def test_sequence_logprob():
    set_seed(0)
    m = GPTModel(GPT_CONFIG_TINY).eval()
    p, c = torch.tensor([1, 2, 3]), torch.tensor([4, 5])
    lp = sequence_logprob(m, p, c)
    logits = m(torch.tensor([[1, 2, 3, 4]]))[0]
    manual = torch.log_softmax(logits[2], -1)[4] + torch.log_softmax(logits[3], -1)[5]
    assert torch.allclose(lp, manual, atol=1e-5)


def test_param_counts():
    # GPT-2 small: ~124M with tied embeddings (incl. position embeddings & biases)
    m = GPTModel(gpt2_config("gpt2-small (124M)"))
    n = count_parameters(m, unique=True)
    assert 123e6 < n < 164e6
    p = params(ModelShape.from_qwen3(QWEN3_CONFIGS["0.6B"]))["total"]
    assert 0.55e9 < p < 0.65e9, p


def test_lr_schedule():
    assert lr_at(0, 1.0, 100, warmup_steps=10) == 0.1
    assert abs(lr_at(100, 1.0, 100, warmup_steps=10, min_lr=0.1) - 0.1) < 1e-9
