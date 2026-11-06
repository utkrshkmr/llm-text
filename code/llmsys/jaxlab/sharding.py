"""Sharding in JAX (Appendix C): automatic sharding under jit, sharding constraints, shard_map.

Three levels of control over the same device mesh:

1. ``jax.jit`` with ``in_shardings``/``out_shardings``: you say where the inputs and outputs
   live, the SPMD partitioner chooses every collective.
2. ``jax.lax.with_sharding_constraint``: pin an intermediate's layout inside a jitted function.
3. ``jax.shard_map``: write the per-device program yourself, with explicit ``psum``,
   ``all_gather``, ``psum_scatter`` and ``ppermute``.

The four sharded-matmul cases of Chapter 24 are written at levels 1 and 3, and the collective
(ring) matmuls at level 3. Everything runs on simulated CPU devices (see ``jaxlab/__init__``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

Spec = tuple  # a PartitionSpec written as a tuple, e.g. ("dp", None)


# --------------------------------------------------------------------------- mesh helpers
def make_mesh(shape: Sequence[int] = (2, 4), names: Sequence[str] = ("dp", "tp")) -> Mesh:
    """A device mesh with *automatic* axis types (the partitioner may propagate shardings).

    JAX 0.10 defaults ``jax.make_mesh`` to *explicit* axes, where shardings are part of an
    array's type and mismatches are errors. The appendix works in automatic mode, which is the
    GSPMD model: annotate inputs and outputs, let the compiler fill in the rest.
    """
    need = math.prod(shape)
    have = len(jax.devices())
    if have < need:
        raise RuntimeError(
            f"mesh {tuple(shape)} needs {need} devices but JAX sees {have}; set "
            "XLA_FLAGS=--xla_force_host_platform_device_count=8 before importing jax "
            "(importing llmsys.jaxlab first does it for you)")
    return jax.make_mesh(tuple(shape), tuple(names), axis_types=(AxisType.Auto,) * len(shape))


def named(mesh: Mesh, *spec) -> NamedSharding:
    return NamedSharding(mesh, P(*spec))


def shard(x, mesh: Mesh, *spec) -> jax.Array:
    """Place ``x`` on the mesh with the given PartitionSpec (``device_put`` moves the shards)."""
    return jax.device_put(jnp.asarray(x), named(mesh, *spec))


def spec_of(x: jax.Array) -> Spec:
    """The array's PartitionSpec as a plain tuple padded with ``None`` to ``x.ndim``."""
    spec = tuple(getattr(x.sharding, "spec", ()))
    return spec + (None,) * (x.ndim - len(spec))


def local_shape(x: jax.Array) -> tuple[int, ...]:
    """Shape of the piece one device holds (``x.addressable_shards`` are the local blocks)."""
    return tuple(x.addressable_shards[0].data.shape)


# --------------------------------------------------------------------------- level 1: jit
def sharded_matmul_auto(mesh: Mesh, a_spec: Spec, b_spec: Spec,
                        out_spec: Spec | None = None) -> Callable:
    """``jit(a @ b)`` with declared input (and optional output) shardings.

    XLA's SPMD partitioner decides the collective from the specs alone: it is the compiler's
    version of the calculus of Chapter 24. Inspect its choice with ``hlo.compiled_text``.
    """
    out = None if out_spec is None else named(mesh, *out_spec)
    return jax.jit(lambda a, b: a @ b,
                   in_shardings=(named(mesh, *a_spec), named(mesh, *b_spec)), out_shardings=out)


# --------------------------------------------------------------------------- level 2: constraints
def mlp_with_constraint(mesh: Mesh, hidden_spec: Spec) -> Callable:
    """Two-layer MLP where the hidden activation's layout is pinned with a sharding constraint.

    Without the constraint the partitioner picks the layout of ``h`` itself (usually well);
    with it you can force, e.g., Megatron's column-parallel hidden ``("dp", "tp")`` and see the
    second matmul become a row-parallel partial product followed by an all-reduce.
    """
    def mlp(x, w1, w2):
        h = jax.nn.gelu(x @ w1)
        h = jax.lax.with_sharding_constraint(h, named(mesh, *hidden_spec))
        return h @ w2

    return jax.jit(mlp)


# --------------------------------------------------------------------------- level 3: shard_map
@dataclass(frozen=True)
class MatmulCase:
    """One of the four sharded-matmul cases, with the specs that define it on a (dp, tp) mesh."""
    number: int
    name: str
    a_spec: Spec
    b_spec: Spec
    out_spec: Spec
    collective: str      # what a hand-written SPMD program must issue
    bytes_per_device: str  # ring cost, S = bytes of the communicated operand, n = tp size


CASES: tuple[MatmulCase, ...] = (
    MatmulCase(1, "local", ("dp", None), (None, "tp"), ("dp", "tp"),
               "none", "0"),
    MatmulCase(2, "one operand on K", ("dp", "tp"), (), ("dp", None),
               "all-gather A over tp, or partial product + all-reduce", "(n-1)/n * S"),
    MatmulCase(3, "both on K", ("dp", "tp"), ("tp", None), ("dp", None),
               "partial product + all-reduce (or reduce-scatter for a sharded output)",
               "2(n-1)/n * S  (all-reduce)  or  (n-1)/n * S  (reduce-scatter)"),
    MatmulCase(4, "conflict on tp", ("tp", None), (None, "tp"), ("tp", None),
               "all-gather one operand (B here)", "(n-1)/n * S"),
)


def case1_local(mesh: Mesh) -> Callable:
    """A (dp, -) @ B (-, tp) -> C (dp, tp). No axis is contracted over a sharded dimension."""
    return jax.shard_map(lambda a, b: a @ b, mesh=mesh,
                         in_specs=(P("dp", None), P(None, "tp")), out_specs=P("dp", "tp"))


def case2_gather(mesh: Mesh) -> Callable:
    """A (dp, tp) @ B (replicated) -> C (dp, -): all-gather A's K blocks, then a local matmul.

    ``check_vma=False``: shard_map's static check cannot infer that ``all_gather`` makes the
    product identical on every ``tp`` device, and would reject the replicated out_spec with a
    ValueError. We know the devices agree, so we switch the check off for this function.
    """
    def f(a, b):
        a_full = jax.lax.all_gather(a, "tp", axis=1, tiled=True)
        return a_full @ b

    return jax.shard_map(f, mesh=mesh, in_specs=(P("dp", "tp"), P()),
                         out_specs=P("dp", None), check_vma=False)


def case2_partial(mesh: Mesh) -> Callable:
    """Same case, the other way: each device multiplies by *its* rows of B, then all-reduce.

    Costs (n-1)/n * |C| * 2 bytes instead of (n-1)/n * |A|: cheaper when C is smaller than A,
    i.e. when K is large relative to N.
    """
    def f(a, b):
        k_local = a.shape[1]
        start = jax.lax.axis_index("tp") * k_local
        b_rows = jax.lax.dynamic_slice_in_dim(b, start, k_local, axis=0)
        return jax.lax.psum(a @ b_rows, "tp")

    return jax.shard_map(f, mesh=mesh, in_specs=(P("dp", "tp"), P()), out_specs=P("dp", None))


def case3_allreduce(mesh: Mesh) -> Callable:
    """A (dp, tp) @ B (tp, -) -> C (dp, -): local partial products, all-reduce over tp.

    Megatron's row-parallel linear (the second GEMM of an MLP or attention's output projection).
    """
    return jax.shard_map(lambda a, b: jax.lax.psum(a @ b, "tp"), mesh=mesh,
                         in_specs=(P("dp", "tp"), P("tp", None)), out_specs=P("dp", None))


def case3_reducescatter(mesh: Mesh) -> Callable:
    """Same inputs, output sharded (dp, tp): reduce-scatter instead of all-reduce (half the bytes).

    This is the sequence-parallel variant: the result lands already split along the columns.
    """
    def f(a, b):
        return jax.lax.psum_scatter(a @ b, "tp", scatter_dimension=1, tiled=True)

    return jax.shard_map(f, mesh=mesh, in_specs=(P("dp", "tp"), P("tp", None)),
                         out_specs=P("dp", "tp"))


def case4_gather(mesh: Mesh) -> Callable:
    """A (tp, -) @ B (-, tp) -> C (tp, -): both sharded over tp on non-contracting dims.

    Device i holds rows block i of A and column block i of B; a local product would give only
    the diagonal blocks of C. Gathering B makes the matmul local and C comes out row-sharded.
    (Gathering A instead gives C column-sharded: this is the sequence-parallel -> tensor-parallel
    all-gather in Megatron-LM.)
    """
    def f(a, b):
        b_full = jax.lax.all_gather(b, "tp", axis=1, tiled=True)
        return a @ b_full

    return jax.shard_map(f, mesh=mesh, in_specs=(P("tp", None), P(None, "tp")),
                         out_specs=P("tp", None))


SHARD_MAP_CASES: dict[str, Callable[[Mesh], Callable]] = {
    "case1_local": case1_local,
    "case2_gather": case2_gather,
    "case2_partial": case2_partial,
    "case3_allreduce": case3_allreduce,
    "case3_reducescatter": case3_reducescatter,
    "case4_gather": case4_gather,
}


# --------------------------------------------------------------------------- collective matmul
def _ring(n: int) -> list[tuple[int, int]]:
    """ppermute permutation: device j sends to j+1 (mod n)."""
    return [(j, (j + 1) % n) for j in range(n)]


def collective_matmul_allgather(mesh: Mesh, axis: str = "tp") -> Callable:
    """X (tp, -) @ W (-, tp) -> C (-, tp) with the all-gather of X folded into a ppermute ring.

    Instead of gathering all of X and then multiplying, device i multiplies the X block it
    currently holds by its W columns, writes that block of rows of C, and passes the X block to
    its neighbour. After n-1 passes every device has multiplied every block. The ppermute of
    step s overlaps with the matmul of step s, which is the point (Wang et al., 2023).
    """
    n = mesh.shape[axis]

    def f(x, w):
        me = jax.lax.axis_index(axis)
        rows = x.shape[0]
        out = jnp.zeros((n * rows, w.shape[1]), x.dtype)
        blk = x
        for s in range(n):
            src = (me - s) % n                       # whose rows does ``blk`` hold?
            out = jax.lax.dynamic_update_slice_in_dim(out, blk @ w, src * rows, axis=0)
            if s < n - 1:
                blk = jax.lax.ppermute(blk, axis, perm=_ring(n))
        return out

    return jax.shard_map(f, mesh=mesh, in_specs=(P(axis, None), P(None, axis)),
                         out_specs=P(None, axis))


def collective_matmul_reducescatter(mesh: Mesh, axis: str = "tp") -> Callable:
    """A (-, tp) @ B (tp, -) -> C (tp, -): the reduce-scatter of case 3 folded into a ring.

    A running partial sum travels around the ring; at each step the holder adds its own
    contribution for the destination block. After n steps the block for device i has visited
    every device and sits on device i.
    """
    n = mesh.shape[axis]

    def f(a, b):
        me = jax.lax.axis_index(axis)
        rows = a.shape[0] // n
        acc = None
        for s in range(n):
            dest = (me - s - 1) % n                  # block that ends on device me at step n-1
            a_rows = jax.lax.dynamic_slice_in_dim(a, dest * rows, rows, axis=0)
            part = a_rows @ b
            acc = part if acc is None else acc + part
            if s < n - 1:
                acc = jax.lax.ppermute(acc, axis, perm=_ring(n))
        return acc

    return jax.shard_map(f, mesh=mesh, in_specs=(P(None, axis), P(axis, None)),
                         out_specs=P(axis, None))


# --------------------------------------------------------------------------- cost model
def ring_collective_bytes(kind: str, total_bytes: int, n: int) -> int:
    """Bytes each device sends for a ring collective over ``n`` devices (Chapter 22's model)."""
    if n <= 1:
        return 0
    per_direction = (n - 1) * total_bytes // n
    return {"all-gather": per_direction, "reduce-scatter": per_direction,
            "all-reduce": 2 * per_direction, "ppermute": total_bytes}[kind]


def collective_time_s(kind: str, total_bytes: int, n: int, alpha_s: float = 25e-6,
                      beta_Bps: float = 22e9) -> float:
    """alpha-beta time of a ring collective; defaults are Chapter 24's PCIe link (two A100s)."""
    steps = {"all-gather": n - 1, "reduce-scatter": n - 1, "all-reduce": 2 * (n - 1),
             "ppermute": 1}[kind]
    return steps * alpha_s + ring_collective_bytes(kind, total_bytes, n) / beta_Bps


def ring_overlap_time_s(compute_s: float, total_bytes: int, n: int, alpha_s: float = 25e-6,
                        beta_Bps: float = 22e9) -> tuple[float, float]:
    """(sequential, overlapped) time of an all-gather followed by a matmul over ``n`` devices.

    ``compute_s`` is one device's time for its whole matmul after the gather. The ring runs n
    partial matmuls of compute_s/n; n-1 of them overlap one hop of total_bytes/n, the last one
    has nothing left to send. The gain tends to 2x when compute and communication balance at
    large n; at n=2 only one partial matmul overlaps, so the best case is 4/3.
    """
    hop = alpha_s + total_bytes / n / beta_Bps
    sequential = (n - 1) * hop + compute_s
    overlapped = (n - 1) * max(compute_s / n, hop) + compute_s / n
    return sequential, overlapped


# --------------------------------------------------------------------------- checks used by the lab
def replicated_like(mesh: Mesh, tree):
    """Place every leaf of a pytree on all devices (how we keep parameters under DDP-style dp)."""
    return jax.device_put(tree, named(mesh))


def gpt_data_parallel_step(n_heads: int) -> Callable:
    """``jit(value_and_grad(loss))`` for ``basics.loss_fn``, with no sharding annotation at all.

    Commit replicated parameters and a batch sharded on its first dimension: the shardings flow
    in with the arguments, the partitioner reduces every gradient, and the all-reduce combiner
    merges those into one tuple all-reduce, a DDP step derived by the compiler.
    """
    from . import basics

    return jax.jit(lambda params, idx, targets:
                   jax.value_and_grad(basics.loss_fn)(params, idx, targets, n_heads))


def matmul_grad(fn: Callable) -> Callable:
    """Jitted gradient of ``0.5 * ||fn(a, b)||^2`` with respect to both operands.

    The cotangents flow backwards through the collectives: JAX transposes all_gather to a
    reduce-scatter and ppermute to the reverse ring, without a line of ours.
    """
    return jax.jit(jax.grad(lambda a, b: 0.5 * jnp.sum(fn(a, b) ** 2), argnums=(0, 1)))


def calculus_vs_compiler(mesh: Mesh, a_spec: Spec, b_spec: Spec, shape_a: Sequence[int],
                         shape_b: Sequence[int]) -> dict:
    """Compare the plan of Chapter 24's ``matmul_plan`` with what XLA actually compiled.

    Returns the case, the calculus's collectives and output spec, and the collectives and output
    spec of the compiled ``jit(a @ b)`` with a free output sharding.
    """
    from llmsys.dist.sharding import Mesh as CalcMesh, ShardSpec, matmul_plan

    from . import hlo

    calc_mesh = CalcMesh(tuple(mesh.axis_names), tuple(mesh.shape[a] for a in mesh.axis_names))
    pad = lambda spec: tuple(spec) + (None,) * (2 - len(spec))
    plan = matmul_plan(ShardSpec(pad(a_spec)), ShardSpec(pad(b_spec)), calc_mesh, shape_a,
                       shape_b, dtype_bytes=4)
    a = shard(jnp.ones(shape_a, jnp.float32), mesh, *a_spec)
    b = shard(jnp.ones(shape_b, jnp.float32), mesh, *b_spec)
    fn = sharded_matmul_auto(mesh, a_spec, b_spec)
    counts = hlo.count_collectives(hlo.compiled_text(fn, a, b))
    return {"case": plan.case, "calculus": [k.replace("_", "-") for k in plan.kinds()],
            "calculus_out": tuple(plan.out.dims),
            "compiler": sorted(k for k, v in counts.items() for _ in range(v)),
            "compiler_out": spec_of(fn(a, b))}
