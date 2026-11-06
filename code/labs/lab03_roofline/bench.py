"""Lab 3: measure an empirical roofline and place the kernel catalogue on it.

    python labs/lab03_roofline/bench.py --device cuda --out-dir results
    python labs/lab03_roofline/bench.py --impl starter      # use your own functions

Bandwidth roof: a copy kernel (y.copy_(x)) over buffers far larger than L2. Compute roof:
the best large bf16 matmul. Every timing is the median of >= 20 repetitions after a 1 s timed
warm-up, with p10/p90 kept. Also measures saved activations and peak memory of a 16-block MLP
stack with and without activation checkpointing (the saved bytes are counted with
saved_activation_bytes of Lab 2, labs/lab02_pytorch). Writes roofline.json and roofline.md into
--out-dir; the chapter's measured-roofline figure is drawn from the JSON by figures.py.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import torch  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
HARDWARE = "2x A100-40GB (PCIe)"
DATASHEET = {"A100-40GB-PCIe": dict(flops=312e12, bw=1.555e12)}  # bf16 dense, HBM2


def load(which: str, lab_dir: str = HERE):
    spec = importlib.util.spec_from_file_location(
        f"{os.path.basename(lab_dir)}_{which}", os.path.join(lab_dir, which + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def versions() -> dict:
    drv = "n/a"
    try:
        drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.split()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        pass
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor()
    return dict(hardware=HARDWARE, device_name=gpu, torch=torch.__version__,
                cuda=torch.version.cuda, driver=drv, python=platform.python_version())


def matmul_sweep(lab, sizes, dtype, device, reps, data):
    """Square GEMMs. ``data='zeros'`` shows the data-dependent power draw of the tensor cores:
    the same instruction stream runs faster on a power-capped card when the bits do not toggle."""
    out = []
    for n in sizes:
        mk = torch.zeros if data == "zeros" else torch.randn
        a, b = mk(n, n, dtype=dtype, device=device), mk(n, n, dtype=dtype, device=device)
        st = lab.time_stats(lambda: a @ b, warmup_s=1.0, reps=reps)
        out.append(dict(n=n, data=data, flops=2 * n**3, flops_per_s=2 * n**3 / st["seconds"],
                        intensity=lab.matmul_intensity(n, n, n, a.element_size()), **st))
        del a, b
    return out


def checkpoint_memory(lab, lab02, device, dtype, scale):
    """Saved activations measured two ways for a 16-block MLP stack (d=4096, hidden 16384, 8,192
    tokens): bytes autograd saves (saved-tensor hooks, Lab 2's saved_activation_bytes) and the
    peak memory of one forward + backward, with and without per-block activation
    checkpointing."""
    import torch.nn as nn
    d, h, tokens = max(8, int(4096 * scale)), max(8, int(16384 * scale)), max(8, int(8192 * scale))
    torch.manual_seed(0)
    blocks = [nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d)).to(device, dtype)
              for _ in range(16)]
    x = torch.randn(tokens, d, device=device, dtype=dtype, requires_grad=True)
    out = dict(blocks=16, d=d, hidden=h, tokens=tokens,
               weight_bytes=sum(p.numel() * p.element_size() for b in blocks for p in b.parameters()))
    for ck in (False, True):
        key = "checkpoint" if ck else "full"
        out[key + "_saved_bytes"] = lab02.saved_activation_bytes(blocks, x, use_checkpoint=ck)
        if device != "cpu":
            out[key + "_peak_bytes"] = lab.peak_step_memory(blocks, x, use_checkpoint=ck)
            for b in blocks:
                b.zero_grad(set_to_none=True)
            x.grad = None
            st = lab.time_stats(lambda: lab.peak_step_memory(blocks, x, use_checkpoint=ck),
                                warmup_s=0.0, reps=3)
            out[key + "_step_s"] = st["seconds"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--impl", default="solution")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--scale", type=float, default=1.0, help="shrink every size (CPU runs)")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()
    lab = load(a.impl)
    lab02 = load(a.impl, os.path.join(os.path.dirname(HERE), "lab02_pytorch"))
    dt = DTYPES[a.dtype]
    gpu = a.device != "cpu"
    os.makedirs(a.out_dir, exist_ok=True)

    copy_sizes = [2**k for k in range(20, 31 if gpu else 27, 2)]          # 1 MiB .. 1 GiB
    copy = lab.measure_copy_bandwidth([max(1, int(s * a.scale)) for s in copy_sizes],
                                      device=a.device, reps=a.reps)
    mm_sizes = [max(16, int(n * a.scale)) for n in (1024, 2048, 4096, 8192, 16384)]
    mm = matmul_sweep(lab, mm_sizes, dt, a.device, a.reps, "randn")
    mm_zero = matmul_sweep(lab, mm_sizes[-2:], dt, a.device, a.reps, "zeros")
    pts = lab.measure_kernel_points(device=a.device, dtype=dt, scale=a.scale, reps=a.reps)

    ckpt = checkpoint_memory(lab, lab02, a.device, dt, a.scale)
    pf, pb = lab.empirical_roofline(mm, copy)
    res = dict(meta=dict(versions(), dtype=a.dtype, reps=a.reps, scale=a.scale),
               datasheet=DATASHEET["A100-40GB-PCIe"],
               roof=dict(peak_flops=pf, peak_bw=pb, ridge=lab.ridge_point(pf, pb)),
               copy=copy, matmul=mm, matmul_zeros=mm_zero, kernels=pts,
               checkpoint=ckpt)
    for p in pts:
        p["attainable"] = lab.roofline_attainable(p["intensity"], pf, pb)
        p["fraction_of_roof"] = p["flops_per_s"] / p["attainable"]
    with open(os.path.join(a.out_dir, "roofline.json"), "w") as f:
        json.dump(res, f, indent=1)

    m = res["meta"]
    lines = [f"# Empirical roofline ({a.dtype})", "",
             f"Measured on {m['hardware']} ({m['device_name']}); torch {m['torch']}, "
             f"CUDA {m['cuda']}, driver {m['driver']}. Median of {a.reps} repetitions after a "
             f"timed warm-up; p10-p90 in brackets.", "",
             f"Roof: {pf/1e12:.1f} TFLOP/s (best bf16 GEMM), {pb/1e9:.0f} GB/s (copy), "
             f"ridge {pf/pb:.0f} FLOP/byte.", "",
             "| copy buffer | GB/s |", "|---:|---:|"]
    lines += [f"| {r['n_bytes']/2**20:.0f} MiB | {r['bytes_per_s']/1e9:.0f} |" for r in copy]
    lines += ["", "| GEMM n | data | TFLOP/s (p90-p10 time) |", "|---:|:--|---:|"]
    for r in mm + mm_zero:
        lines.append(f"| {r['n']} | {r['data']} | {r['flops_per_s']/1e12:.1f} "
                     f"({r['flops']/r['p90']/1e12:.1f}-{r['flops']/r['p10']/1e12:.1f}) |")
    lines += ["", "| kernel | FLOP/byte | time (us) | TFLOP/s | GB/s | % of roof |",
              "|:--|---:|---:|---:|---:|---:|"]
    for p in pts:
        lines.append(f"| {p['name']} | {p['intensity']:.3g} | {p['seconds']*1e6:.1f} | "
                     f"{p['flops_per_s']/1e12:.2f} | {p['bytes_per_s']/1e9:.0f} | "
                     f"{100*p['fraction_of_roof']:.0f} |")
    c = ckpt
    lines += ["", f"Activation checkpointing, {c['blocks']} MLP blocks (d={c['d']}, hidden "
              f"{c['hidden']}, {c['tokens']} tokens, weights {c['weight_bytes']/1e9:.2f} GB):", "",
              "| variant | saved activations (GB) | peak fwd+bwd above weights (GB) | step (s) |",
              "|:--|---:|---:|---:|"]
    for k in ("full", "checkpoint"):
        lines.append(f"| {k} | {c[k + '_saved_bytes']/1e9:.2f} | "
                     f"{c.get(k + '_peak_bytes', float('nan'))/1e9:.2f} | "
                     f"{c.get(k + '_step_s', float('nan')):.3f} |")
    with open(os.path.join(a.out_dir, "roofline.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
