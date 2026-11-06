"""Run the Chapter 8 ablation sweep with llmsys.ablate and plot learning curves.

CPU (minutes, synthetic corpus):
    python run.py --steps 400 --seeds 0 1 --out abl.json
GPU on a real text file (e.g. a TinyStories dump), bigger model:
    python run.py --text tinystories_valid.txt --device cuda --sdpa --amp --compile \
        --d-model 512 --layers 8 --heads 8 --ctx 256 --batch 64 --steps 5000 --lr 1e-3 --seeds 0
    python run.py --plot abl.json --png abl.png
"""
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llmsys.ablate import STANDARD_ABLATIONS, ArchConfig, make_corpus, train_lm  # noqa: E402


def load_ids(path):
    if path is None:
        return make_corpus(300_000)
    data = Path(path).read_bytes()
    return torch.tensor(list(data), dtype=torch.long)   # byte-level: vocab 256, no tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None, help="UTF-8 text file; default = synthetic corpus")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--variants", nargs="*", default=None, help="subset of variant names")
    ap.add_argument("--out", default="abl.json")
    ap.add_argument("--sdpa", action="store_true", help="fused attention kernel")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (GPU)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model")
    ap.add_argument("--plot", default=None, help="plot an existing json instead of training")
    ap.add_argument("--png", default="abl.png")
    a = ap.parse_args()

    if a.plot:
        import matplotlib.pyplot as plt
        res = json.load(open(a.plot))
        for r in res:
            if r["val_loss"]:
                plt.plot(r["steps"], r["val_loss"], label=f'{r["name"]} s{r["config"].get("seed", "")}')
        plt.xlabel("step"); plt.ylabel("val loss (nats/byte)"); plt.legend(fontsize=7)
        plt.savefig(a.png, dpi=150); print("wrote", a.png)
        return

    ids = load_ids(a.text)                    # stays on the CPU; batches are copied per step
    base = ArchConfig(d_model=a.d_model, n_layers=a.layers, n_heads=a.heads, context_length=a.ctx,
                      use_sdpa=a.sdpa)
    variants = {k: v for k, v in STANDARD_ABLATIONS.items() if not a.variants or k in a.variants}
    out = []
    for name, ov in variants.items():
        for s in a.seeds:
            r = train_lm(dataclasses.replace(base, **ov), ids, steps=a.steps, batch=a.batch,
                         lr=a.lr, seed=s, name=name, log=print,
                         device=a.device, amp=a.amp, compile=a.compile)
            out.append(dataclasses.asdict(r))
            json.dump(out, open(a.out, "w"))     # save after every run: sweeps get interrupted
    print("wrote", a.out)


if __name__ == "__main__":
    main()
