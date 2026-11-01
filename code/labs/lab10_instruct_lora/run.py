"""Instruction-tune GPT-2 (full or LoRA), generate test responses, judge them with Ollama.

Run on your own machine (needs internet once for weights/data; GPU recommended):
  huggingface-cli download gpt2-medium --local-dir gpt2-medium
  # an Alpaca-style list of {"instruction","input","output"} records, e.g. the ~1.1k-example
  # instruction-data.json used by the LLMs-from-scratch repository, or any Alpaca subset.
  python run.py --data instruction-data.json --gpt2-dir gpt2-medium --epochs 2
  python run.py --data instruction-data.json --gpt2-dir gpt2-medium --lora-rank 16 --lr 3e-4
  ollama serve & ollama pull llama3
  python run.py --judge-only responses.json --judge-model llama3
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from functools import partial

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from llmsys.finetune import (InstructionDataset, build_judge_prompt, extract_response,  # noqa: E402
                             format_alpaca, instruction_collate, parse_score, query_ollama)
from llmsys.generate import generate  # noqa: E402
from llmsys.gpt import GPTModel, gpt2_config, load_gpt2_hf_state_dict  # noqa: E402
from llmsys.lora import apply_lora, count_trainable, lora_state_dict, merge_lora  # noqa: E402
from llmsys.train import loss_loader  # noqa: E402
from llmsys.utils import get_device, set_seed  # noqa: E402

EOT = 50256


def judge(path, model_name):
    entries = json.load(open(path))
    scores = []
    for e in entries:
        s = parse_score(query_ollama(build_judge_prompt(e, e["model_response"]), model_name))
        if s is not None:
            scores.append(s)
    print(f"judged {len(scores)}/{len(entries)}; mean score {sum(scores)/max(len(scores),1):.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--gpt2-dir")
    ap.add_argument("--size", default="gpt2-medium (355M)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--lora-rank", type=int, default=0)
    ap.add_argument("--lora-alpha", type=float, default=16.0)
    ap.add_argument("--mask-prompt", action="store_true")
    ap.add_argument("--judge-only")
    ap.add_argument("--judge-model", default="llama3")
    args = ap.parse_args()

    if args.judge_only:
        return judge(args.judge_only, args.judge_model)

    import tiktoken
    from safetensors.torch import load_file

    set_seed(123)
    device = get_device()
    tok = tiktoken.get_encoding("gpt2")
    data = json.load(open(args.data))
    random.Random(123).shuffle(data)
    n_tr, n_te = int(0.85 * len(data)), int(0.10 * len(data))
    train_d, test_d, val_d = data[:n_tr], data[n_tr:n_tr + n_te], data[n_tr + n_te:]

    collate = partial(instruction_collate, pad_id=EOT, allowed_max_length=args.max_length,
                      mask_prompt=args.mask_prompt, device=device)
    if not args.mask_prompt:  # plain token lists -> loss on prompt and response
        collate_fn = lambda b: collate([ids for ids, _ in b])  # noqa: E731
    else:
        collate_fn = collate
    mk = lambda d, sh: DataLoader(InstructionDataset(d, tok), batch_size=args.batch_size,  # noqa: E731
                                  shuffle=sh, drop_last=sh, collate_fn=collate_fn)
    train_loader, val_loader = mk(train_d, True), mk(val_d, False)

    model = GPTModel(gpt2_config(args.size, drop_rate=0.0))
    load_gpt2_hf_state_dict(model, load_file(os.path.join(args.gpt2_dir, "model.safetensors")))
    if args.lora_rank:
        apply_lora(model, args.lora_rank, args.lora_alpha)
    model.to(device)
    tr, tot = count_trainable(model)
    print(f"trainable {tr:,} / {tot:,} ({tr/tot:.2%})")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.1)
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
    t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        for step, (x, y) in enumerate(train_loader):
            loss = ce(model(x).flatten(0, 1).float(), y.flatten())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 50 == 0:
                print(f"ep {ep+1} step {step} loss {loss.item():.3f}")
        model.eval()
        print(f"epoch {ep+1} val loss {loss_loader(model, val_loader, device, 10):.3f}")
    print(f"train time {time.perf_counter()-t0:.1f}s; "
          f"peak mem {torch.cuda.max_memory_allocated()/1e9 if device.type=='cuda' else 0:.2f} GB")

    if args.lora_rank:
        torch.save(lora_state_dict(model), "lora_adapter.pth")
        merge_lora(model)

    out = []
    for e in test_d:
        prompt = format_alpaca(e)
        ids = torch.tensor([tok.encode(prompt)], device=device)
        gen = generate(model, ids, max_new_tokens=256, eos_id=EOT)
        text = tok.decode(gen[0].tolist())
        out.append(dict(e, model_response=extract_response(text)))
    json.dump(out, open("responses.json", "w"), indent=2)
    print("wrote responses.json; now run with --judge-only responses.json")


if __name__ == "__main__":
    main()
