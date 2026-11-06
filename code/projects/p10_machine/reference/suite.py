"""Project 10 (reference solution): the whole suite, milestones 1-5 in order (Chapter 6).

    doc = run(load_config("configs/smoke.json"), device="cuda", out="results/smoke.json")

Every milestone returns raw rows (kept under ``artifacts``) and headline numbers (``metrics``);
milestone 4 builds a ``Machine`` from milestones 1-3 and predicts the 20 kernels of the zoo.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from typing import Dict, List, Optional, Sequence

import torch

from . import bandwidth, compute, env, kernels, latency, model, report, stats, timing


def load_config(path: str) -> Dict:
    with open(path) as f:
        cfg = json.load(f)
    for key in ("name", "reps", "m1", "m2", "m3", "m4"):
        if key not in cfg:
            raise ValueError(f"config {path} lacks {key!r}")
    return cfg


def _sizes(spec: Sequence[int]) -> List[int]:
    """[lo, hi, per_octave] -> power-of-two sizes; an explicit longer list is used as is."""
    if len(spec) == 3 and spec[0] < spec[1]:
        return bandwidth.log_sizes(int(spec[0]), int(spec[1]), int(spec[2]))
    return [int(v) for v in spec]


def _plateau(rows: Sequence[Dict], min_ws: float) -> Dict[str, float]:
    """Plateau of a cold sweep: median of the three largest working sets (all far beyond the
    cache), with their spread; ``converged`` is False if the curve still climbs by more than 3%
    across them, in which case the sweep should go further."""
    big = [r for r in rows if r["working_set"] >= min_ws][-3:] or list(rows[-3:])
    bws = [r["bw"] for r in big]
    return dict(
        bw=float(stats.summarize(bws)["median"]),
        p10=min(r["bw_p10"] for r in big),
        p90=max(r["bw_p90"] for r in big),
        converged=bool(bws[-1] <= 1.03 * bws[0]),
    )


def _l2_bytes(device) -> Optional[int]:
    if not timing.is_cuda(device):
        return None
    return getattr(torch.cuda.get_device_properties(torch.device(device)), "L2_cache_size", None)


# ----------------------------------------------------------------------------- milestone 1
def milestone1(cfg: Dict, device) -> Dict:
    c, reps = cfg["m1"], cfg["reps"]
    l2 = _l2_bytes(device) or 0
    out: Dict = {"stream": {}, "warm": {}, "host": {}, "p2p": {}, "numa": {}}
    for kind in c["stream_kinds"]:
        out["stream"][kind] = bandwidth.bandwidth_sweep(
            kind, _sizes(c["stream_sizes"]), device, "cold", reps
        )
    out["warm"]["copy"] = bandwidth.bandwidth_sweep(
        "copy", _sizes(c["warm_sizes"]), device, "warm", reps
    )
    m: Dict[str, float] = {}
    for kind in c["stream_kinds"]:
        pl = _plateau(out["stream"][kind], 4 * l2)
        m[f"hbm_{kind}_bw"] = pl["bw"]
        m[f"hbm_{kind}_bw_p10"], m[f"hbm_{kind}_bw_p90"] = pl["p10"], pl["p90"]
        m[f"hbm_{kind}_converged"] = float(pl["converged"])
    warm = out["warm"]["copy"]
    # a lower bound: every graph-replayed copy also pays the device's per-kernel gap (milestone 3
    # measures it; ``run`` adds a gap-corrected estimate)
    m["l2_bw"] = max(r["bw"] for r in warm)
    m["l2_bw_working_set"] = max(warm, key=lambda r: r["bw"])["working_set"]
    knee = bandwidth.find_knee([r["working_set"] for r in warm], [r["bw"] for r in warm])
    if knee is not None:
        m["l2_knee_bytes"] = knee
    if timing.is_cuda(device):
        hs = _sizes(c["host_sizes"])
        for direction in ("h2d", "d2h"):
            for pinned in (True, False):
                tag = f"{direction}_{'pinned' if pinned else 'pageable'}"
                # host buffers on the GPU's own NUMA node (the remote node costs up to 20%)
                rows = bandwidth.host_device_sweep(
                    hs, direction, pinned, device, reps, node=bandwidth.gpu_numa_node(device)
                )
                out["host"][tag] = rows
                best, fit = bandwidth.summarize_paths(rows)
                m[f"{tag}_bw"] = best
                m[f"{tag}_bw_plateau"] = float(
                    stats.summarize([r["bw"] for r in rows[-3:]])["median"]
                )
                m[f"{tag}_alpha"], m[f"{tag}_beta"] = fit["alpha"], fit["beta"]
        if c.get("numa"):
            # each placement in its own process, bound to one node; pages verified afterwards
            gpu = torch.device(device).index or 0
            nodes = sorted(
                int(d[4:])
                for d in os.listdir("/sys/devices/system/node")
                if d.startswith("node") and d[4:].isdigit()
            )
            for node in nodes:
                for pinned in (True, False):
                    tag = f"h2d_{'pinned' if pinned else 'pageable'}_node{node}"
                    out["numa"][tag] = bandwidth.numa_h2d(node, hs[-1], pinned, gpu, reps)
        if torch.cuda.device_count() >= 2 and c.get("p2p_sizes"):
            ps = _sizes(c["p2p_sizes"])
            src = torch.device(device).index or 0
            dst = 1 - src if src in (0, 1) else 0
            out["p2p"]["uni"] = bandwidth.p2p_sweep(ps, src, dst, False, reps)
            m["p2p_bw"], fit = bandwidth.summarize_paths(out["p2p"]["uni"])
            m["p2p_correct"] = float(all(r["correct"] for r in out["p2p"]["uni"]))
            if not m["p2p_correct"]:
                warnings.warn(
                    "PEER-TO-PEER COPIES DELIVERED WRONG DATA on this machine (see "
                    "versions.iommu and the chapter): never use tensor.to() between "
                    "these GPUs; use NCCL or a host-staged copy",
                    RuntimeWarning,
                )
            m["p2p_alpha"], m["p2p_beta"] = fit["alpha"], fit["beta"]
            for chunk in c.get("staged_chunks", []):
                r = bandwidth.staged_bandwidth(ps[-1], src, dst, int(chunk), reps=max(5, reps // 2))
                out["p2p"].setdefault("staged", []).append(r)
            if out["p2p"].get("staged"):
                m["staged_bw"] = max(r["bw"] for r in out["p2p"]["staged"])
                m["staged_correct"] = float(all(r["correct"] for r in out["p2p"]["staged"]))
            if c.get("p2p_bidirectional"):
                out["p2p"]["bi"] = bandwidth.p2p_sweep(ps[-2:], src, dst, True, reps)
                m["p2p_bidir_bw"] = max(r["bw"] for r in out["p2p"]["bi"])
    return dict(metrics=m, rows=out)


# ----------------------------------------------------------------------------- milestone 2
def milestone2(cfg: Dict, device) -> Dict:
    c, reps = cfg["m2"], cfg["reps"]
    rows: Dict = {"matmul": {}, "fma": [], "sustained": []}
    m: Dict[str, float] = {}
    for dt in c["matmul_dtypes"]:
        if not timing.is_cuda(device) and dt in ("tf32", "int8", "fp16"):
            continue
        best, rs = compute.matmul_peak(dt, c["matmul_sizes"], device, reps)
        rows["matmul"][dt] = rs
        top = max(rs, key=lambda r: r["flops_per_s"])
        m[f"matmul_{dt}"] = best
        m[f"matmul_{dt}_p10"], m[f"matmul_{dt}_p90"] = top["flops_p10"], top["flops_p90"]
    rows["fma"] = compute.fma_sweep(
        int(c["fma_numel"]), c["fma_iters"], device, reps, compile=c.get("fma_compile", True)
    )
    m["simt_fp32"] = max(r["flops_per_s"] for r in rows["fma"])
    if timing.is_cuda(device) and c.get("sustained_seconds", 0) > 0:
        rows["sustained"] = compute.sustained_matmul(
            c.get("sustained_n", 8192), "bf16", float(c["sustained_seconds"]), device
        )
        rate = rows["sustained"]["rate"]
        tail = [r["flops_per_s"] for r in rate[len(rate) // 2 :]]  # the steady second half
        m["matmul_bf16_sustained"] = float(stats.summarize(tail)["median"])
        m["matmul_bf16_first"] = rate[0]["flops_per_s"]
        clk = [r["clock_mhz"] for r in rate[len(rate) // 2 :] if r.get("clock_mhz")]
        if clk:
            m["sustained_sm_mhz"] = float(stats.summarize(clk)["median"])
            # throughput per clock, scaled to the boost clock: constant if FLOP/s tracks the clock
            boost = float(str(cfg.get("boost_mhz", 1410)))
            per = [
                r["flops_per_s"] * boost / r["clock_mhz"]
                for r in rate[len(rate) // 2 :]
                if r.get("clock_mhz")
            ]
            m["matmul_bf16_sustained_at_boost"] = float(stats.summarize(per)["median"])
    return dict(metrics=m, rows=rows)


# ----------------------------------------------------------------------------- milestone 3
def milestone3(cfg: Dict, device) -> Dict:
    c = cfg["m3"]
    lo = latency.launch_overhead(device, n=c["launch_n"], reps=cfg["reps"])
    sy = latency.sync_overhead(device, reps=c.get("sync_reps", 200))
    sc = latency.small_copy_latency(
        device, c.get("small_copy_bytes", 4), reps=c.get("sync_reps", 200)
    )
    gv = latency.graph_vs_eager(c["graph_n"], device, reps=cfg["reps"])
    m = dict(
        launch_s=lo["single_event"]["median"],
        enqueue_s=lo["enqueue"]["median"],
        empty_event_s=lo["empty_event"]["median"],
        launch_host_s=lo["single_host"]["median"],
        eager_kernel_s=lo["eager_burst"]["median"],
        sync_s=sy["synchronize_idle"]["median"],
        item_s=sy["item"]["median"],
        copy4_pinned_s=sc["h2d_pinned"]["median"],
        copy4_pageable_s=sc["h2d_pageable"]["median"],
        clock_resolution_s=latency.clock_resolution(2000),
    )
    if "graph_burst" in lo:
        m["graph_kernel_s"] = lo["graph_burst"]["median"]
    for key, summ in (
        ("launch_s", lo["single_event"]),
        ("enqueue_s", lo["enqueue"]),
        ("empty_event_s", lo["empty_event"]),
        ("sync_s", sy["synchronize_idle"]),
        ("item_s", sy["item"]),
        ("copy4_pinned_s", sc["h2d_pinned"]),
        ("graph_kernel_s", lo.get("graph_burst")),
    ):
        if summ:
            m[f"{key}_p10"], m[f"{key}_p90"] = summ["p10"], summ["p90"]
    return dict(metrics=m, rows=dict(launch=lo, sync=sy, small_copy=sc, graph_vs_eager=gv))


# ----------------------------------------------------------------------------- milestone 4
def build_machine(metrics: Dict[str, float], name: str) -> model.Machine:
    """The measured machine: the best ceilings of milestones 1-3."""
    peaks = {
        dt: metrics[f"matmul_{dt}"]
        for dt in ("fp32", "tf32", "fp16", "bf16", "int8")
        if f"matmul_{dt}" in metrics
    }
    for dt in ("tf32", "fp16", "bf16"):  # CPU runs: no tensor-core classes
        peaks.setdefault(dt, peaks.get("fp32", metrics.get("simt_fp32", 1e9)))
    peaks["simt"] = metrics.get("simt_fp32", peaks.get("fp32", 1e9))
    return model.Machine(
        name=name,
        peak_flops=peaks,
        hbm_bw=metrics["hbm_copy_bw"],
        launch_s=metrics.get("launch_s", 0.0),
        gap_s=metrics.get("graph_kernel_s", metrics.get("launch_s", 0.0)),
        l2_bw=metrics.get("l2_bw"),
        l2_bytes=metrics.get("l2_knee_bytes"),
    )


def datasheet_machine(key: str) -> model.Machine:
    ds = report.DATASHEETS[key]
    peaks = {k: ds[k] for k in ("fp32", "tf32", "fp16", "bf16", "int8", "simt")}
    return model.Machine(name=f"{key} (datasheet)", peak_flops=peaks, hbm_bw=ds["hbm_bw"])


def calibration_overlap(cfg: Dict, ks: Sequence[kernels.Kernel]) -> set:
    """Zoo kernels that coincide with a calibration probe of milestones 1-2 (a square GEMM of a
    swept size and format, or a copy of a swept size): predicting them is not a prediction."""
    sizes = set(int(v) for v in cfg["m2"].get("matmul_sizes", []))
    dtypes = set(cfg["m2"].get("matmul_dtypes", []))
    stream = (
        set(_sizes(cfg["m1"]["stream_sizes"])) if "copy" in cfg["m1"]["stream_kinds"] else set()
    )
    out = set()
    for k in ks:
        if k.shape and k.category == "gemm":
            m, n, kk, dt = k.shape
            if m == n == kk and m in sizes and dt in dtypes:
                out.add(k.name)
        if k.shape and k.name.startswith("copy_") and k.shape[0] in stream:
            out.add(k.name)
    return out


def milestone4(cfg: Dict, device, machine: model.Machine) -> Dict:
    c = cfg["m4"]
    ks = kernels.zoo(c.get("shapes", "gpu"), compile_ok=c.get("compile", True))
    flush = timing.L2Flusher(device) if timing.is_cuda(device) else None
    meas, timings, quality = [], {}, {}
    for k in ks:
        fn = k.make(torch.device(device))
        samples = timing.event_samples(
            fn, warmup=5, reps=c.get("reps", cfg["reps"]), device=device, flush=flush, ahead_us=100
        )
        s = stats.summarize(samples)
        lo, hi = stats.bootstrap_ci(samples, n_boot=500)
        quality[k.name] = dict(
            ci95=(lo, hi),
            steady_from=stats.steady_state_index(samples),
            outliers=len(samples) - len(stats.reject_outliers(samples)),
        )
        meas.append(s["median"])
        timings[k.name] = s
        del fn
        if timing.is_cuda(device):
            torch.cuda.empty_cache()
    preds = {
        "roofline": model.predict_all(machine, ks, launch=False),
        "roofline+launch": model.predict_all(machine, ks),
    }
    if cfg.get("datasheet") in report.DATASHEETS and timing.is_cuda(device):
        preds = {
            "datasheet": model.predict_all(datasheet_machine(cfg["datasheet"]), ks, launch=False),
            **preds,
        }
    fitted = model.fit_model(machine, ks, meas)
    if c.get("loo", True):
        preds["fitted_loo"] = model.loo_predictions(machine, ks, meas)
    band = c.get("band", 0.2)
    errors = {name: stats.error_metrics(p, meas, band) for name, p in preds.items()}
    # out-of-sample: drop zoo kernels whose exact shape was also a calibration probe
    in_sample = calibration_overlap(cfg, ks)
    keep = [i for i, k in enumerate(ks) if k.name not in in_sample]
    errors_oos = (
        {
            name: stats.error_metrics([p[i] for i in keep], [meas[i] for i in keep], band)
            for name, p in preds.items()
        }
        if keep
        else {}
    )
    bounds = [model.bound(machine, k) for k in ks]
    rows = report.prediction_rows([k.name for k in ks], meas, preds, bounds)
    kern = [
        dict(
            name=k.name,
            category=k.category,
            compute=k.compute,
            flops=k.flops,
            bytes=k.bytes,
            launches=k.launches,
            intensity=k.intensity,
            measured_s=t,
        )
        for k, t in zip(ks, meas)
    ]
    return dict(
        metrics={
            **{f"{n}_within": e["within"] for n, e in errors.items()},
            **{f"{n}_within_oos": e["within"] for n, e in errors_oos.items()},
        },
        rows=dict(
            kernels=kern,
            predictions=preds,
            errors=errors,
            errors_out_of_sample=errors_oos,
            in_sample=sorted(in_sample),
            quality=quality,
            prediction_rows=rows,
            fitted=vars(fitted),
        ),
        timings=timings,
    )


# ----------------------------------------------------------------------------- driver
def configure_timing(cfg: Dict, versions: Dict) -> None:
    """Machine-specific timer constants: the flush buffer covers twice the largest on-chip cache
    (config ``flush_bytes``, else 2x the datasheet's last-level cache, else 2x the L2 the driver
    reports), and the spin loop converts microseconds with the maximum SM clock."""
    ds = report.DATASHEETS.get(cfg.get("datasheet", ""), {})
    flush = cfg.get("flush_bytes") or (2 * ds["llc_bytes"] if "llc_bytes" in ds else None)
    mhz = None
    try:
        mhz = float(str(versions.get("max_sm_clock", "")).split()[0])
    except (ValueError, IndexError):
        pass
    timing.configure(flush_bytes=flush, spin_cycles_per_us=cfg.get("spin_cycles_per_us", mhz))


def run(
    cfg: Dict,
    device="cuda",
    out: Optional[str] = None,
    milestones: Sequence[int] = (1, 2, 3, 4),
    log=print,
) -> Dict:
    """Run the chosen milestones and (optionally) write the results JSON + Markdown spec sheet."""
    t_start = time.perf_counter()
    versions = env.collect()
    configure_timing(cfg, versions)
    metrics: Dict[str, float] = {}
    artifacts: Dict[str, object] = {}
    timings: Dict[str, Dict] = {}
    for i, fn in ((1, milestone1), (2, milestone2), (3, milestone3)):
        if i in milestones:
            t0 = time.perf_counter()
            res = fn(cfg, device)
            metrics.update(res["metrics"])
            artifacts[f"m{i}"] = res["rows"]
            if timing.is_cuda(device):
                artifacts.setdefault("clocks", {})[f"after_m{i}"] = env.gpu_clocks(
                    env.smi_index(device)
                )
            log(f"milestone {i}: {time.perf_counter() - t0:.1f} s")
    if "l2_bw" in metrics and metrics.get("graph_kernel_s"):
        # each replayed copy is t = gap + bytes / bw_true, so remove the measured gap
        moved = metrics["l2_bw_working_set"]
        t = moved / metrics["l2_bw"] - metrics["graph_kernel_s"]
        metrics["l2_bw_gap_corrected"] = moved / t if t > 0 else float("nan")
    if 4 in milestones:
        if "hbm_copy_bw" not in metrics or "launch_s" not in metrics:
            raise ValueError("milestone 4 needs milestones 1-3 (or their metrics) first")
        t0 = time.perf_counter()
        machine = build_machine(metrics, versions["hardware"])
        res = milestone4(cfg, device, machine)
        metrics.update(res["metrics"])
        artifacts["m4"] = res["rows"]
        artifacts["prediction_rows"] = res["rows"]["prediction_rows"]
        artifacts["model_errors"] = [
            dict(model=n, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()})
            for n, e in res["rows"]["errors"].items()
        ] + [
            dict(
                model=n + " (out of sample)",
                **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()},
            )
            for n, e in res["rows"]["errors_out_of_sample"].items()
        ]
        timings.update(
            {k: {q: v[q] for q in ("median", "p10", "p90")} for k, v in res["timings"].items()}
        )
        log(f"milestone 4: {time.perf_counter() - t0:.1f} s")
    if "m3" in artifacts:
        lat = artifacts["m3"]
        for name, s in (
            ("launch_single", lat["launch"]["single_event"]),
            ("sync_idle", lat["sync"]["synchronize_idle"]),
            ("item", lat["sync"]["item"]),
        ):
            timings[name] = {q: s[q] for q in ("median", "p10", "p90")}
    metrics["wall_seconds"] = time.perf_counter() - t_start
    doc = report.results_document(
        dict(cfg, device=str(device)), versions, metrics, timings, artifacts
    )
    if out:
        md = report.write(doc, out)
        log(f"wrote {out} and {md}")
    return doc
