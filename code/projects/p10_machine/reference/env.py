"""Project 10 (reference solution): record the environment every number depends on (Chapter 4).

A bandwidth or FLOP/s figure without the GPU model, driver, CUDA, PyTorch version, clocks, power
cap and topology cannot be compared with anything, including your own run next month.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from typing import Dict, List, Optional

import torch


def _cmd(args) -> Optional[str]:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def _cpu_model() -> Optional[str]:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def hardware_string() -> str:
    """'2x A100-40GB (PCIe)'-style label for plots and tables."""
    if not torch.cuda.is_available():
        return f"CPU ({_cpu_model() or platform.machine()})"
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    name = names[0].replace("NVIDIA ", "")
    return f"{len(names)}x {name}" if len(set(names)) == 1 else " + ".join(names)


def collect() -> Dict[str, object]:
    info: Dict[str, object] = dict(
        hardware=hardware_string(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cpu=_cpu_model(),
        cpus=os.cpu_count(),
    )
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception:
        info["triton"] = None
    if torch.cuda.is_available():
        q = _cmd(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,power.limit,clocks.max.sm,"
                "clocks.max.mem,pcie.link.gen.max,pcie.link.width.max",
                "--format=csv,noheader",
            ]
        )
        if q:
            first = [v.strip() for v in q.splitlines()[0].split(",")]
            keys = [
                "driver",
                "power_limit",
                "max_sm_clock",
                "max_mem_clock",
                "pcie_gen_max",
                "pcie_width_max",
            ]
            info.update(dict(zip(keys, first)))
        gpus = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append(
                dict(
                    name=p.name,
                    sms=p.multi_processor_count,
                    mem_bytes=p.total_memory,
                    cc=f"{p.major}.{p.minor}",
                    l2_bytes=getattr(p, "L2_cache_size", None),
                )
            )
        info["gpus"] = gpus
        info["peer_access"] = {
            f"{i}->{j}": torch.cuda.can_device_access_peer(i, j)
            for i in range(len(gpus))
            for j in range(len(gpus))
            if i != j
        }
        clean = lambda t: re.sub(r"\x1b\[[0-9;]*m", "", t) if t else None  # noqa: E731 (ANSI)
        info["topology"] = clean(_cmd(["nvidia-smi", "topo", "-m"]))
        # nvidia-smi's own peer-to-peer read/write status: "OK" is necessary, not sufficient
        info["p2p_read"] = clean(_cmd(["nvidia-smi", "topo", "-p2p", "r"]))
        info["p2p_write"] = clean(_cmd(["nvidia-smi", "topo", "-p2p", "w"]))
        info["iommu"] = [iommu_info(i) for i in range(len(gpus))]
    return info


def versions_line(info: Dict[str, object]) -> str:
    """One line for captions: hardware, driver, CUDA, torch."""
    return (
        f"{info.get('hardware')}, driver {info.get('driver', 'n/a')}, "
        f"CUDA {info.get('cuda')}, torch {info.get('torch')}"
    )


_SMI_INDEX: Dict[str, int] = {}


def pci_bus_id(device) -> Optional[str]:
    """'0000:17:00.0'-style PCI address of a CUDA device (None without CUDA)."""
    if not torch.cuda.is_available():
        return None
    p = torch.cuda.get_device_properties(torch.device("cuda", torch.device(device).index or 0))
    return "%04x:%02x:%02x.0" % (p.pci_domain_id, p.pci_bus_id, p.pci_device_id)


def smi_index(device) -> int:
    """nvidia-smi index of a CUDA device, matched by PCI bus id (nvidia-smi and CUDA may number
    devices differently, and CUDA_VISIBLE_DEVICES renumbers them again)."""
    bus = pci_bus_id(device)
    fallback = torch.device(device).index or 0
    if bus is None:
        return fallback
    if bus not in _SMI_INDEX:
        q = _cmd(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"])
        for line in (q or "").splitlines():
            idx, pci = [v.strip() for v in line.split(",")]
            _SMI_INDEX[pci.lower()[-12:]] = int(idx)  # '00000000:17:00.0' -> '0000:17:00.0'
    return _SMI_INDEX.get(bus.lower(), fallback)


def iommu_info(device) -> Dict[str, Optional[str]]:
    """IOMMU group and its type for a GPU. Type 'DMA' or 'DMA-FQ' means addresses are translated;
    NVIDIA does not support PCIe peer-to-peer through a translating IOMMU on bare-metal Linux
    (peer writes can be lost silently); 'identity' (iommu=pt) or no IOMMU is the supported case."""
    bus = pci_bus_id(device)
    out: Dict[str, Optional[str]] = dict(pci=bus, iommu_group=None, iommu_type=None, numa_node=None)
    if bus is None:
        return out
    base = f"/sys/bus/pci/devices/{bus}"
    try:
        group = os.path.realpath(os.path.join(base, "iommu_group"))
        out["iommu_group"] = os.path.basename(group)
        with open(os.path.join(group, "type")) as f:
            out["iommu_type"] = f.read().strip()
    except OSError:
        pass
    try:
        with open(os.path.join(base, "numa_node")) as f:
            out["numa_node"] = f.read().strip()
    except OSError:
        pass
    return out


def gpu_clocks(index: int = 0) -> Optional[Dict[str, str]]:
    """Current SM clock (MHz), power draw (W) and active clock-event (throttle) reasons."""
    q = _cmd(
        [
            "nvidia-smi",
            "--query-gpu=clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active",
            "--format=csv,noheader,nounits",
            "-i",
            str(index),
        ]
    )
    if not q:
        return None
    sm, mem, power, reasons = [v.strip() for v in q.split(",")]
    return dict(sm_mhz=sm, mem_mhz=mem, power_w=power, reasons=reasons)


def start_clock_sampler(index: int = 0, period_ms: int = 100):
    """Start ``nvidia-smi`` sampling SM clock, power and throttle reasons every ``period_ms`` in
    a background process. Each sample carries nvidia-smi's own wall-clock timestamp, which
    ``stop_clock_sampler`` converts to ``time.perf_counter`` time: the process takes a variable
    100-300 ms to start, so counting samples from the launch would misplace every one of them.
    (NVIDIA only; on AMD the same role is played by ``amd-smi``/``rocm-smi``.)"""
    import time

    try:
        proc = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,clocks.sm,power.draw,clocks_event_reasons.active",
                "--format=csv,noheader,nounits",
                f"-lms={period_ms}",
                "-i",
                str(index),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None
    return dict(proc=proc, wall_minus_perf=time.time() - time.perf_counter())


def parse_smi_timestamp(ts: str) -> float:
    """'2026/11/06 00:41:43.157' (local time) -> seconds since the epoch."""
    import datetime

    return datetime.datetime.strptime(ts.strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()


def stop_clock_sampler(sampler, t_ref: float) -> List[Dict[str, float]]:
    """Stop the sampler; return samples with ``t`` in seconds relative to ``t_ref``
    (a ``time.perf_counter`` value), from nvidia-smi's own timestamps."""
    if sampler is None:
        return []
    proc = sampler["proc"]
    proc.terminate()
    out, _ = proc.communicate(timeout=10)
    rows = []
    for line in out.splitlines():
        try:
            ts, sm, power, reasons = [v.strip() for v in line.split(",")]
            t_perf = parse_smi_timestamp(ts) - sampler["wall_minus_perf"]
            rows.append(
                dict(t=t_perf - t_ref, sm_mhz=float(sm), power_w=float(power), reasons=reasons)
            )
        except ValueError:
            continue
    return rows


def clocks_in_window(
    rows: List[Dict[str, float]], t0: float, t1: float, slack: float = 0.15
) -> Optional[float]:
    """Median SM clock of the samples inside [t0, t1]; if none fall inside (a window shorter
    than the sampling period), the nearest sample within ``slack`` seconds; else None."""
    inside = sorted(r["sm_mhz"] for r in rows if t0 <= r["t"] <= t1)
    if inside:
        return inside[len(inside) // 2]
    near = [r for r in rows if t0 - slack <= r["t"] <= t1 + slack]
    if not near:
        return None
    mid = 0.5 * (t0 + t1)
    return min(near, key=lambda r: abs(r["t"] - mid))["sm_mhz"]
