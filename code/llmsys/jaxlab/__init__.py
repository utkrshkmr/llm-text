"""JAX and XLA companion code for PyTorch engineers (Appendix C).

The sub-modules are importable on a CPU-only machine and never need a GPU: the labs simulate a
device mesh with XLA's ``--xla_force_host_platform_device_count`` flag, which must be in the
environment *before* JAX is imported. Importing this package takes care of that (default: eight
host devices) and exposes ``HAS_JAX`` so that callers can skip gracefully when JAX is missing.

    basics      pure functions + parameter pytrees for a GPT-2 block; jit / grad / vmap / scan;
                converting an ``llmsys.gpt`` state_dict to JAX arrays
    sharding    meshes, NamedSharding/PartitionSpec, automatic sharding under jit, sharding
                constraints, shard_map with explicit collectives, the four sharded-matmul cases,
                collective (ring) matmuls
    hlo         lowering to StableHLO and optimised HLO, counting collectives and fusions,
                profiling with jax.profiler
"""
from __future__ import annotations

import os
import sys

DEFAULT_DEVICE_COUNT = 8


def ensure_simulated_devices(n: int = DEFAULT_DEVICE_COUNT) -> bool:
    """Ask XLA's CPU backend to expose ``n`` devices. Returns False if JAX was imported first.

    XLA reads ``XLA_FLAGS`` once, when the backend is initialised, so the flag has to be set
    before ``import jax`` anywhere in the process. An existing device-count flag is respected.
    The simulated mesh only exists on the CPU backend, so we also default ``JAX_PLATFORMS`` to
    ``cpu``; set it to ``gpu`` or ``tpu`` yourself to run the same code on real devices.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    flag = f"--xla_force_host_platform_device_count={n}"
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_force_host_platform_device_count" in flags:
        return True
    if "jax" in sys.modules:
        return False
    os.environ["XLA_FLAGS"] = f"{flags} {flag}".strip()
    return True


ensure_simulated_devices()

try:  # JAX is an optional dependency of the book's code: everything else is pure PyTorch
    import jax  # noqa: F401

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only on machines without jax
    HAS_JAX = False

__all__ = ["HAS_JAX", "DEFAULT_DEVICE_COUNT", "ensure_simulated_devices"]
