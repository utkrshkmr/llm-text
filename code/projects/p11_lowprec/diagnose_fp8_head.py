"""Project 11: the FP8 output-head diagnostic of the chapter (Chapter 5).

    CUDA_VISIBLE_DEVICES=1 python diagnose_fp8_head.py --out results/diag_fp8_head_a100_pcie.json

Trains the smoke transformer three times (single seed): with only the output head's gradient
operand rounded stochastically, with only that operand in BF16, and the plain `fp8_tensor` recipe
while capturing the head's and one inner layer's gradient (and the activation each multiplies) at
steps 300, 600 and 900. For each captured tensor it reports `diagnose.rounding_bias` under E5M2
round-to-nearest, E5M2 stochastic rounding (64 draws), E4M3 round-to-nearest and BF16.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))                    # code/ for llmsys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "smoke.json"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--impl", default=None)
    a = ap.parse_args(argv)
    import torch
    import impl
    p = impl.load(a.impl)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = p.study.load_config(a.config)["models"]["gpt"]
    out = {"versions": p.study.versions(device)}
    for rec in ("fp8_tensor_head_g_sr", "fp8_tensor_head_g_bf16"):
        r = p.train.train({**cfg, "recipe": rec, "seed": 0}, device)
        out[rec] = {"final_val": p.analysis.final_val(r), "floor": r["floor"]}
        print(rec, out[rec], flush=True)
    caps = {}

    def setup(prec):
        for layer in ("head", "b1.fc2"):
            for role in ("g", "xb"):
                caps[(layer, role)] = prec.policies[layer].q[role].capture = \
                    p.diagnose.EveryKth(300, offset=299)
    r = p.train.train({**cfg, "recipe": "fp8_tensor", "seed": 0}, device, setup=setup)
    out["fp8_tensor_final_val"] = p.analysis.final_val(r)
    gen = torch.Generator(device=device).manual_seed(0)
    specs = (("e5m2_rne", p.emulate.QSpec("e5m2", "tensor"), 1),
             ("e5m2_sr", p.emulate.QSpec("e5m2", "tensor", rounding="stochastic"), 64),
             ("e4m3_rne", p.emulate.QSpec("e4m3", "tensor"), 1),
             ("bf16", p.emulate.QSpec("bf16"), 1))
    for layer in ("head", "b1.fc2"):
        for i, (g, x) in enumerate(zip(caps[(layer, "g")], caps[(layer, "xb")])):
            for name, spec, draws in specs:
                res = p.diagnose.rounding_bias(g, x, spec, draws, gen)
                out[f"{layer}@{300 * (i + 1)}/{name}"] = res
                print(layer, 300 * (i + 1), name, {k: f"{v:.2e}" for k, v in res.items()},
                      flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
