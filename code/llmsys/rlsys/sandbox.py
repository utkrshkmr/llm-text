"""Executing untrusted model-generated code to compute a reward.

This is a *teaching* sandbox: a fresh interpreter in an empty temp directory, an
empty environment, CPU-time / address-space / file-descriptor / file-size limits and a
wall-clock timeout. It is NOT a security boundary. Production reward services run
code inside containers or micro-VMs (gVisor, Firecracker, nsjail) with no network,
read-only images, cgroups, and a pool of pre-warmed workers.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Sequence, Tuple

try:
    import resource
except ImportError:  # pragma: no cover - Windows
    resource = None


@dataclass
class SandboxResult:
    status: str        # "pass" | "fail" | "timeout" | "error"
    reward: float
    elapsed: float
    stderr: str = ""


def _limits(cpu_sec: int, mem_mb: int):
    def fn():
        if resource is None:
            return
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec))
        mem = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
        os.setsid()  # own process group so a timeout can kill children too
    return fn


def run_python_reward(code: str, tests: str, timeout: float = 2.0, mem_mb: int = 512) -> SandboxResult:
    """Run ``code`` followed by ``tests`` (assert statements). Reward 1.0 iff exit code 0."""
    program = code + "\n\n" + tests + "\n"
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            p = subprocess.run([sys.executable, "-I", "-S", path], cwd=tmp, env={},
                               capture_output=True, text=True, timeout=timeout,
                               preexec_fn=_limits(int(timeout) + 1, mem_mb))
        except subprocess.TimeoutExpired:
            return SandboxResult("timeout", 0.0, time.perf_counter() - t0)
    dt = time.perf_counter() - t0
    if p.returncode == 0:
        return SandboxResult("pass", 1.0, dt)
    status = "fail" if "AssertionError" in p.stderr else "error"
    return SandboxResult(status, 0.0, dt, p.stderr[-500:])


def score_batch(items: Sequence[Tuple[str, str]], max_workers: int = 4,
                timeout: float = 2.0) -> List[SandboxResult]:
    """Score many (code, tests) pairs concurrently; order is preserved.

    Threads are fine here: each worker just waits on a subprocess.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(lambda ct: run_python_reward(ct[0], ct[1], timeout), items))
