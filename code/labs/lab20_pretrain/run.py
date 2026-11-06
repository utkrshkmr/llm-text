"""Lab 20 driver: pretrain a small GPT on a text file and benchmark the training step.

Examples
--------
# 1) pretrain GPT-2-small-shaped model (ctx 256) on a short story, e.g. Edith Wharton's
#    public-domain "The Verdict" (any UTF-8 .txt works):
python run.py train --text the-verdict.txt --epochs 10 --peak-lr 4e-4 --out ckpt.pt

# 2) benchmark one training step under different systems settings (GPU recommended):
python run.py bench --preset 124M --batch 8 --seq 1024 --amp bf16 --ckpt-act --compile

# 3) load OpenAI GPT-2 weights from a Hugging Face snapshot dir and sample:
python run.py sample --hf-dir ./gpt2 --prompt "Every effort moves you" --temperature 1.4 --top-k 25

Uses tiktoken's GPT-2 BPE if installed; otherwise trains a byte-level BPE on the text.
"""
import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

import solution as lab  # noqa: E402
from llmsys.data import create_dataloader  # noqa: E402
from llmsys.flops import HARDWARE  # noqa: E402
from llmsys.gpt import GPTModel, gpt2_config, load_gpt2_hf_state_dict  # noqa: E402
from llmsys.utils import count_parameters, get_device, set_seed  # noqa: E402


def get_tokenizer(text):
    try:
        import tiktoken
        return tiktoken.get_encoding("gpt2"), 50257
    except Exception:
        from llmsys.tokenizer import ByteBPETokenizer
        tok = ByteBPETokenizer.train(text, vocab_size=2048)
        return tok, tok.vocab_size


def cmd_train(a):
    set_seed(123)
    dev = get_device()
    text = open(a.text, encoding="utf-8").read()
    tok, vocab = get_tokenizer(text)
    tr_text, va_text = lab.train_val_split(text, 0.9)
    cfg = gpt2_config("gpt2-small (124M)", vocab_size=vocab, context_length=a.ctx,
                      qkv_bias=False, drop_rate=0.1)
    tr = create_dataloader(tr_text, tok, a.batch, a.ctx, a.ctx, shuffle=True, drop_last=True)
    va = create_dataloader(va_text, tok, a.batch, a.ctx, a.ctx, shuffle=False, drop_last=False)
    model = GPTModel(cfg).to(dev)
    print(f"params {count_parameters(model)/1e6:.1f}M | train batches {len(tr)} | device {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.peak_lr, weight_decay=0.1)
    tc = lab.TrainConfig(epochs=a.epochs, peak_lr=a.peak_lr, initial_lr=a.peak_lr / 100,
                         min_lr=a.peak_lr / 10, warmup_frac=0.1, grad_clip=1.0,
                         amp_dtype=torch.bfloat16 if a.amp == "bf16" else None,
                         eval_freq=5, eval_iter=5, device=str(dev))
    h = lab.train_model(model, tr, va, opt, tc)
    lab.save_checkpoint(a.out, model, opt, h.step[-1] if h.step else 0)
    ids = torch.tensor([tok.encode("Every effort moves you")], device=dev)
    out = lab.generate_text(model, ids, 30, a.ctx, temperature=1.0, top_k=25)
    print(tok.decode(out[0].tolist()))


def cmd_bench(a):
    dev = get_device()
    preset = {"124M": "gpt2-small (124M)", "355M": "gpt2-medium (355M)",
              "774M": "gpt2-large (774M)"}[a.preset]
    cfg = gpt2_config(preset, context_length=a.seq, drop_rate=0.0, qkv_bias=False,
                      use_sdpa=a.sdpa)
    model = GPTModel(cfg).to(dev)
    if a.ckpt_act:
        lab.enable_activation_checkpointing(model)
    step_model = torch.compile(model) if a.compile else model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=(dev.type == "cuda"))
    x = torch.randint(0, cfg["vocab_size"], (a.batch, a.seq), device=dev)
    amp = torch.bfloat16 if a.amp == "bf16" else None
    n = count_parameters(model)

    def step():
        with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
            logits = step_model(x)
        loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), x.flatten())
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(3):
        step()
    if dev.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(a.iters):
        step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / a.iters
    tps = a.batch * a.seq / dt
    print(f"{n/1e6:.0f}M params | {dt*1e3:.1f} ms/step | {tps:,.0f} tok/s")
    if dev.type == "cuda":
        print(f"peak memory {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
        print(f"MFU vs {a.hw}: {lab.mfu(tps, n, HARDWARE[a.hw]['flops']):.1%}")
    est = lab.training_memory_estimate(n, cfg["n_layers"], cfg["emb_dim"], cfg["n_heads"],
                                       a.seq, a.batch, a.ckpt_act, flash=a.sdpa)
    print({k: f"{v/2**30:.2f} GiB" for k, v in est.items()})


def cmd_sample(a):
    from safetensors.torch import load_file
    import tiktoken
    tok = tiktoken.get_encoding("gpt2")
    model = GPTModel(gpt2_config("gpt2-small (124M)"))
    load_gpt2_hf_state_dict(model, load_file(os.path.join(a.hf_dir, "model.safetensors")))
    ids = torch.tensor([tok.encode(a.prompt)])
    out = lab.generate_text(model.eval(), ids, a.max_new, 1024, a.temperature, a.top_k)
    print(tok.decode(out[0].tolist()))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--text", required=True); t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--peak-lr", type=float, default=4e-4); t.add_argument("--ctx", type=int, default=256)
    t.add_argument("--batch", type=int, default=2); t.add_argument("--amp", default="none")
    t.add_argument("--out", default="ckpt.pt")
    b = sub.add_parser("bench")
    b.add_argument("--preset", default="124M"); b.add_argument("--batch", type=int, default=8)
    b.add_argument("--seq", type=int, default=1024); b.add_argument("--amp", default="bf16")
    b.add_argument("--ckpt-act", action="store_true"); b.add_argument("--compile", action="store_true")
    b.add_argument("--sdpa", action="store_true"); b.add_argument("--iters", type=int, default=10)
    b.add_argument("--hw", default="H100-SXM")
    s = sub.add_parser("sample")
    s.add_argument("--hf-dir", required=True); s.add_argument("--prompt", default="Every effort moves you")
    s.add_argument("--temperature", type=float, default=1.0); s.add_argument("--top-k", type=int, default=25)
    s.add_argument("--max-new", type=int, default=40)
    a = p.parse_args()
    {"train": cmd_train, "bench": cmd_bench, "sample": cmd_sample}[a.cmd](a)


if __name__ == "__main__":
    main()
