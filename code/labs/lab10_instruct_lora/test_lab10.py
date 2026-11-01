import torch
import torch.nn as nn
import torch.nn.functional as F

from _labutil import load_impl
from llmsys.gpt import GPT_CONFIG_TINY, GPTModel
from llmsys.utils import set_seed

lab = load_impl(__file__)
CFG = dict(GPT_CONFIG_TINY, context_length=128, vocab_size=260)


def test_format_alpaca():
    e = dict(instruction="Add the numbers.", input="2 and 3", output="5")
    p = lab.format_alpaca(e)
    assert "### Instruction:\nAdd the numbers." in p and "### Input:\n2 and 3" in p
    assert p.endswith("### Response:\n")
    assert lab.format_alpaca(e, with_response=True).endswith("### Response:\n5")
    assert "### Input" not in lab.format_alpaca(dict(instruction="Hi.", input="", output="x"))


def test_collate_masks_padding_but_keeps_first_eot():
    x, y = lab.instruction_collate([[0, 1, 2, 3, 4], [5, 6], [7, 8, 9]], pad_id=50256)
    assert x.tolist() == [[0, 1, 2, 3, 4], [5, 6, 50256, 50256, 50256], [7, 8, 9, 50256, 50256]]
    assert y.tolist() == [[1, 2, 3, 4, 50256], [6, 50256, -100, -100, -100],
                          [8, 9, 50256, -100, -100]]
    x2, y2 = lab.instruction_collate([[0, 1, 2, 3, 4], [5, 6]], pad_id=9, allowed_max_length=3)
    assert x2.shape == y2.shape == (2, 3)


def test_collate_prompt_masking():
    # (ids, prompt_len): prompt = first 3 tokens, response = last 2
    x, y = lab.instruction_collate([([10, 11, 12, 20, 21], 3)], pad_id=0, mask_prompt=True)
    # targets: predict 11,12 (prompt -> masked), 20, 21, EOT (kept)
    assert y.tolist() == [[-100, -100, 20, 21, 0]]


def test_packing_preserves_tokens_and_loss():
    set_seed(0)
    model = GPTModel(CFG).eval()
    seqs = [[1, 2, 3, 4, 5, 256], [7, 8, 256], [9, 10, 11, 12, 256], [13, 256], [14, 15, 16, 256]]
    packed = lab.pack_sequences(seqs, max_len=8, pad_id=256)
    n_targets = int((packed["targets"] != -100).sum())
    assert n_targets == sum(len(s) - 1 for s in seqs)
    assert packed["input_ids"].shape[0] < len(seqs)          # fewer rows than sequences
    # reference: each sequence alone, token-weighted mean loss
    tot, cnt = 0.0, 0
    for s in seqs:
        t = torch.tensor([s])
        logits = model(t[:, :-1])
        tot += F.cross_entropy(logits.flatten(0, 1), t[:, 1:].flatten(), reduction="sum").item()
        cnt += len(s) - 1
    assert abs(lab.packed_loss(model, packed).item() - tot / cnt) < 1e-4


def test_segment_mask_blocks_cross_document():
    seg = torch.tensor([[1, 1, 2, 2, 0]])
    m = lab.segment_mask(seg)[0, 0]
    assert not m[1, 0] and m[2, 1] and not m[3, 2] and m[0, 1]


def test_lora_linear_init_merge_unmerge():
    set_seed(0)
    base = nn.Linear(16, 12)
    ref = base.weight.detach().clone()
    lora = lab.LoRALinear(base, rank=4, alpha=8)
    x = torch.randn(3, 16)
    assert torch.allclose(lora(x), base(x))                    # B = 0 at init
    with torch.no_grad():
        lora.lora_B.normal_()
    y = lora(x)
    lora.merge()
    assert torch.allclose(lora(x), y, atol=1e-5)
    lora.unmerge()
    assert torch.allclose(base.weight, ref, atol=1e-5)
    assert torch.allclose(lora(x), y, atol=1e-5)


def test_apply_lora_counts():
    set_seed(0)
    model = lab.apply_lora(GPTModel(CFG), rank=4, alpha=8)
    trainable, total = lab.count_trainable(model)
    # every nn.Linear in the tiny GPT: 4 attention (64x64), 2 FFN (64x256, 256x64) per block, head
    d, h, V, L = 64, 256, 260, 2
    expect = L * (4 * 4 * (d + d) + 4 * (d + h) + 4 * (h + d)) + 4 * (d + V)
    assert trainable == expect and total > trainable


def test_multilora_matches_individual_adapters():
    set_seed(0)
    base = nn.Linear(8, 6)
    multi = lab.MultiLoRALinear(base, n_adapters=3, rank=2, alpha=4)
    with torch.no_grad():
        multi.A.normal_(); multi.B.normal_()
    x = torch.randn(4, 5, 8)
    ids = torch.tensor([2, 0, -1, 1])
    out = multi(x, ids)
    for row, a in enumerate(ids.tolist()):
        exp = base(x[row])
        if a >= 0:
            exp = exp + multi.scaling * x[row] @ multi.A[a].T @ multi.B[a].T
        assert torch.allclose(out[row], exp, atol=1e-5)


def test_nf4_roundtrip():
    torch.manual_seed(0)
    assert torch.allclose(lab.nf4_levels(), lab.NF4_LEVELS, atol=1e-6)
    w = torch.randn(32, 96)
    codes, absmax, shape = lab.quantize_nf4(w, block=64)
    assert codes.dtype == torch.uint8 and int(codes.max()) <= 15 and absmax.numel() == 48
    w2 = lab.dequantize_nf4(codes, absmax, shape)
    assert w2.shape == w.shape
    rel = (w2 - w).abs().mean() / w.abs().mean()
    assert rel < 0.12


def test_lora_finetune_reduces_loss():
    set_seed(0)
    tok = lab.ByteTokenizer()
    data = lab.make_toy_instructions(32, seed=0)
    seqs = [tok.encode(lab.format_phi3(e, with_response=True)) for e in data]
    batches = [lab.instruction_collate(seqs[i:i + 8], pad_id=tok.eot_id) for i in range(0, 32, 8)]
    model = lab.apply_lora(GPTModel(CFG), rank=8, alpha=16)
    losses = lab.finetune(model, batches, steps=40, lr=1e-2)
    assert losses[-1] < 0.7 * losses[0]


def test_extract_and_parse():
    text = "...### Response:\nThe answer is 5.<|endoftext|>"
    assert lab.extract_response(text) == "The answer is 5."
    assert lab.parse_score("Score: 87/100") == 87
    assert lab.parse_score("150") == 100 and lab.parse_score("n/a") is None
