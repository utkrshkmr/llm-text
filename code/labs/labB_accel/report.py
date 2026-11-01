"""Lab B report: utilisation curves, padding tables, roofline numbers.

    python labs/labB_accel/report.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from llmsys.qwen3 import QWEN3_CONFIGS  # noqa: E402
from llmsys.gpt import gpt2_config  # noqa: E402
from llmsys.systolic import (ACCELERATORS, decode_tokens_per_s_bound,  # noqa: E402
                             ridge_point, tiled_systolic_cycles)


def matrices_qwen3(cfg):
    d, h, g, hd, f, v = (cfg["emb_dim"], cfg["n_heads"], cfg["n_kv_groups"], cfg["head_dim"],
                         cfg["hidden_dim"], cfg["vocab_size"])
    return {"W_q": (d, h * hd), "W_k": (d, g * hd), "W_v": (d, g * hd), "W_o": (h * hd, d),
            "gate/up": (d, f), "down": (f, d), "lm_head": (d, v)}


def matrices_gpt2(cfg):
    d, v = cfg["emb_dim"], cfg["vocab_size"]
    return {"qkv (each)": (d, d), "out": (d, d), "fc1": (d, 4 * d), "fc2": (4 * d, d), "lm_head": (d, v)}


def main():
    print("== MXU utilisation vs rows streamed (K=N=4096)")
    for R in (128, 256):
        row = [f"{m}:{tiled_systolic_cycles(m, 4096, 4096, R, R)['utilization']:.2f}"
               for m in (1, 8, 64, 256, 1024, 4096)]
        print(f"  {R}x{R}: " + "  ".join(row))
    for name, mats in (("Qwen3-0.6B", matrices_qwen3(QWEN3_CONFIGS["0.6B"])),
                       ("GPT-2 small", matrices_gpt2(gpt2_config()))):
        print(f"== padding efficiency, {name}")
        for k, (K, N) in mats.items():
            e128 = tiled_systolic_cycles(1, K, N, 128, 128)["padding_efficiency"]
            e32 = tiled_systolic_cycles(1, K, N, 32, 32)["padding_efficiency"]
            print(f"  {k:12s} {K:6d} x {N:6d}   128-array {e128:.4f}   32-tiles {e32:.4f}")
    print("== rooflines (ridge, 8B@1B/param decode ceiling)")
    for n, a in ACCELERATORS.items():
        print(f"  {n:26s} ridge {ridge_point(n):7.0f} FLOP/B   "
              f"{decode_tokens_per_s_bound(8e9, 1.0, a['bw']):6.0f} tok/s")


if __name__ == "__main__":
    main()
