"""Lab C (reference solution): JAX and XLA for PyTorch engineers.

Everything runs on the CPU with eight simulated XLA devices. The flag that creates them must be
in the environment before JAX is imported, which is why this module sets it first thing.

    LAB_IMPL=solution python -m pytest -q labs/labC_jax
    python labs/labC_jax/run.py --out /tmp/labC_hlo

Tasks (the functions a student fills in; the rest is given):
    1. layer_norm, causal_attention, transformer_block, gpt_forward   (the model as pure functions)
    2. torch_state_dict_to_jax                                        (PyTorch -> pytree bridge)
    3. per_example_grads                                              (vmap of grad)
    4. sharded_matmul_auto                                            (jit with in/out shardings)
    5. case1_local .. case4_gather                                    (shard_map + collectives)
    6. collective_matmul_allgather                                    (ppermute ring)
    7. count_collectives                                              (reading HLO text)
    8. ring_overlap_time_s                                            (the overlap cost model)
"""
from __future__ import annotations

import math
import os
import re
from typing import Any, Callable, Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                               + " --xla_force_host_platform_device_count=8").strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

Params = dict[str, Any]
Spec = tuple


# =========================================================================== given helpers
def linear(p: Params, x: jax.Array) -> jax.Array:
    """``x @ W (+ b)``; W is stored (in, out), the transpose of ``nn.Linear.weight``."""
    y = x @ p["w"]
    return y + p["b"] if "b" in p else y


def gelu(x: jax.Array) -> jax.Array:
    """GPT-2's tanh approximation, identical to ``llmsys.layers.GELU``."""
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def feed_forward(p: Params, x: jax.Array) -> jax.Array:
    return linear(p["fc2"], gelu(linear(p["fc1"], x)))


def cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    return -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0].mean()


def make_mesh(shape: Sequence[int] = (2, 4), names: Sequence[str] = ("dp", "tp")) -> Mesh:
    """A (dp, tp) mesh over the simulated devices, with automatic (GSPMD-style) axis types."""
    if len(jax.devices()) < math.prod(shape):
        raise RuntimeError("set XLA_FLAGS=--xla_force_host_platform_device_count=8 before "
                           "importing jax")
    return jax.make_mesh(tuple(shape), tuple(names), axis_types=(AxisType.Auto,) * len(shape))


def named(mesh: Mesh, *spec) -> NamedSharding:
    return NamedSharding(mesh, P(*spec))


def shard(x, mesh: Mesh, *spec) -> jax.Array:
    return jax.device_put(jnp.asarray(x), named(mesh, *spec))


def spec_of(x: jax.Array) -> Spec:
    """The array's PartitionSpec as a tuple padded with ``None`` to ``x.ndim``."""
    spec = tuple(getattr(x.sharding, "spec", ()))
    return spec + (None,) * (x.ndim - len(spec))


def matmul_grad(fn: Callable) -> Callable:
    """Jitted gradient of ``0.5 * ||fn(a, b)||^2`` w.r.t. both operands (used by the tests)."""
    return jax.jit(jax.grad(lambda a, b: 0.5 * jnp.sum(fn(a, b) ** 2), argnums=(0, 1)))


def _ring(n: int) -> list[tuple[int, int]]:
    """ppermute permutation for a ring: device j sends to j+1 (mod n)."""
    return [(j, (j + 1) % n) for j in range(n)]


# =========================================================================== Task 1: the model
def layer_norm(p: Params, x: jax.Array, eps: float = 1e-5) -> jax.Array:
    """LayerNorm with biased variance, as ``llmsys.layers.LayerNorm``: p has "scale", "shift"."""
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return p["scale"] * (x - mean) / jnp.sqrt(var + eps) + p["shift"]


def causal_attention(p: Params, x: jax.Array, n_heads: int) -> jax.Array:
    """Multi-head causal attention on (B, T, D); p has "q", "k", "v", "out" linear pytrees."""
    B, T, D = x.shape
    hd = D // n_heads
    split = lambda t: t.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
    q, k, v = split(linear(p["q"], x)), split(linear(p["k"], x)), split(linear(p["v"], x))
    scores = q @ k.swapaxes(-1, -2) / math.sqrt(hd)
    scores = jnp.where(jnp.tril(jnp.ones((T, T), dtype=bool)), scores, -jnp.inf)
    weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
    ctx = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return linear(p["out"], ctx)


def transformer_block(p: Params, x: jax.Array, n_heads: int) -> jax.Array:
    """Pre-LN block: x + Attn(LN1(x)); x + FFN(LN2(x)). Keys: norm1, att, norm2, ff."""
    x = x + causal_attention(p["att"], layer_norm(p["norm1"], x), n_heads)
    return x + feed_forward(p["ff"], layer_norm(p["norm2"], x))


def gpt_forward(params: Params, idx: jax.Array, n_heads: int) -> jax.Array:
    """Logits (B, T, V) from token ids (B, T): tok_emb, pos_emb, blocks, final_norm, out_head."""
    T = idx.shape[-1]
    x = params["tok_emb"][idx] + params["pos_emb"][:T]
    for blk in params["blocks"]:
        x = transformer_block(blk, x, n_heads)
    return layer_norm(params["final_norm"], x) @ params["out_head"]


def loss_fn(params: Params, idx: jax.Array, targets: jax.Array, n_heads: int) -> jax.Array:
    return cross_entropy(gpt_forward(params, idx, n_heads), targets)


# =========================================================================== Task 2: the bridge
def torch_state_dict_to_jax(sd: dict) -> Params:
    """Convert an ``llmsys.gpt.GPTModel`` state_dict into the pytree ``gpt_forward`` expects.

    Linear weights are transposed to (in, out). Torch is not imported: ``.detach().cpu()
    .float().numpy()`` is all we need from the tensors.
    """
    arr = lambda name: jnp.asarray(sd[name].detach().cpu().float().numpy())

    def lin(prefix: str) -> Params:
        p = {"w": arr(prefix + ".weight").T}
        if prefix + ".bias" in sd:
            p["b"] = arr(prefix + ".bias")
        return p

    norm = lambda prefix: {"scale": arr(prefix + ".scale"), "shift": arr(prefix + ".shift")}
    n_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("trf_blocks."))
    blocks = []
    for i in range(n_layers):
        b = f"trf_blocks.{i}."
        blocks.append({
            "norm1": norm(b + "norm1"),
            "att": {"q": lin(b + "att.W_query"), "k": lin(b + "att.W_key"),
                    "v": lin(b + "att.W_value"), "out": lin(b + "att.out_proj")},
            "norm2": norm(b + "norm2"),
            "ff": {"fc1": lin(b + "ff.layers.0"), "fc2": lin(b + "ff.layers.2")},
        })
    return {"tok_emb": arr("tok_emb.weight"), "pos_emb": arr("pos_emb.weight"),
            "blocks": blocks, "final_norm": norm("final_norm"),
            "out_head": arr("out_head.weight").T}


# =========================================================================== Task 3: vmap(grad)
def per_example_grads(params: Params, idx: jax.Array, targets: jax.Array,
                      n_heads: int) -> Params:
    """Gradient of every example's own loss: a pytree whose leaves have a leading batch axis.

    ``vmap`` over the batch of ``grad`` of a one-example loss; the parameters are not mapped
    (``in_axes=None``), the token arrays are mapped along axis 0.
    """
    one = lambda p, i, t: loss_fn(p, i[None], t[None], n_heads)
    return jax.vmap(jax.grad(one), in_axes=(None, 0, 0))(params, idx, targets)


# =========================================================================== Task 4: jit shardings
def sharded_matmul_auto(mesh: Mesh, a_spec: Spec, b_spec: Spec,
                        out_spec: Spec | None = None) -> Callable:
    """``jit(a @ b)`` with declared input shardings and an optional output sharding.

    The SPMD partitioner reads the collective off the specs; when ``out_spec`` is None it also
    chooses where the result lives.
    """
    out = None if out_spec is None else named(mesh, *out_spec)
    return jax.jit(lambda a, b: a @ b,
                   in_shardings=(named(mesh, *a_spec), named(mesh, *b_spec)), out_shardings=out)


# =========================================================================== Task 5: shard_map
def case1_local(mesh: Mesh) -> Callable:
    """A (dp, -) @ B (-, tp) -> C (dp, tp): no contraction over a sharded dimension."""
    return jax.shard_map(lambda a, b: a @ b, mesh=mesh,
                         in_specs=(P("dp", None), P(None, "tp")), out_specs=P("dp", "tp"))


def case2_gather(mesh: Mesh) -> Callable:
    """A (dp, tp) @ B (replicated) -> C (dp, -): all-gather A along K, then multiply locally.

    ``check_vma=False`` because shard_map's static check cannot prove the result is identical
    on every tp device (it is: the gathered A and the replicated B are the same everywhere).
    """
    def f(a, b):
        return jax.lax.all_gather(a, "tp", axis=1, tiled=True) @ b

    return jax.shard_map(f, mesh=mesh, in_specs=(P("dp", "tp"), P()),
                         out_specs=P("dp", None), check_vma=False)


def case3_allreduce(mesh: Mesh) -> Callable:
    """A (dp, tp) @ B (tp, -) -> C (dp, -): local partial products, all-reduce over tp."""
    return jax.shard_map(lambda a, b: jax.lax.psum(a @ b, "tp"), mesh=mesh,
                         in_specs=(P("dp", "tp"), P("tp", None)), out_specs=P("dp", None))


def case3_reducescatter(mesh: Mesh) -> Callable:
    """Same inputs, C sharded (dp, tp): reduce-scatter along the columns instead of all-reduce."""
    def f(a, b):
        return jax.lax.psum_scatter(a @ b, "tp", scatter_dimension=1, tiled=True)

    return jax.shard_map(f, mesh=mesh, in_specs=(P("dp", "tp"), P("tp", None)),
                         out_specs=P("dp", "tp"))


def case4_gather(mesh: Mesh) -> Callable:
    """A (tp, -) @ B (-, tp) -> C (tp, -): both sharded on tp along non-contracting dims.

    A local product would only give the diagonal blocks of C; gathering B's column blocks
    makes the product local and C comes out row-sharded like A.
    """
    def f(a, b):
        return a @ jax.lax.all_gather(b, "tp", axis=1, tiled=True)

    return jax.shard_map(f, mesh=mesh, in_specs=(P("tp", None), P(None, "tp")),
                         out_specs=P("tp", None))


# =========================================================================== Task 6: ring matmul
def collective_matmul_allgather(mesh: Mesh, axis: str = "tp") -> Callable:
    """X (tp, -) @ W (-, tp) -> C (-, tp) with the all-gather of X replaced by a ppermute ring.

    At step s device ``me`` holds the X block that started on device ``(me - s) % n``; it
    multiplies it by its own W columns, writes those rows of C, and passes the block on. The
    ppermute of one step overlaps with the matmul of the same step.
    """
    n = mesh.shape[axis]

    def f(x, w):
        me = jax.lax.axis_index(axis)
        rows = x.shape[0]
        out = jnp.zeros((n * rows, w.shape[1]), x.dtype)
        blk = x
        for s in range(n):
            src = (me - s) % n
            out = jax.lax.dynamic_update_slice_in_dim(out, blk @ w, src * rows, axis=0)
            if s < n - 1:
                blk = jax.lax.ppermute(blk, axis, perm=_ring(n))
        return out

    return jax.shard_map(f, mesh=mesh, in_specs=(P(axis, None), P(None, axis)),
                         out_specs=P(None, axis))


# =========================================================================== Task 7: reading HLO
COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "collective-permute")
_OPS = "|".join(COLLECTIVES)
_HLO_RE = re.compile(rf"^\s*(?:ROOT\s+)?%?\S+\s*=\s*.*?\s({_OPS})(-start|-done)?\(")
_SHLO_RE = re.compile(rf"stablehlo\.({_OPS.replace('-', '_')})\b")


def count_collectives(hlo_text: str) -> dict[str, int]:
    """Count collective *instructions* in HLO or StableHLO text, one entry per kind.

    Match the instruction form ``%name = shape op(...)`` so that ``metadata={op_name=...}``
    and instruction names do not count; an async ``-start``/``-done`` pair counts once.
    """
    counts = {c: 0 for c in COLLECTIVES}
    for line in hlo_text.splitlines():
        m = _HLO_RE.match(line)
        if m is not None:
            if m.group(2) != "-done":
                counts[m.group(1)] += 1
            continue
        for op in _SHLO_RE.findall(line):
            counts[op.replace("_", "-")] += 1
    return counts


def compiled_text(fn: Callable, *args) -> str:
    """Optimised HLO after SPMD partitioning: what actually runs."""
    fn = fn if hasattr(fn, "lower") else jax.jit(fn)
    return fn.lower(*args).compile().as_text()


# =========================================================================== Task 8: cost model
def ring_overlap_time_s(compute_s: float, total_bytes: int, n: int, alpha_s: float = 25e-6,
                        beta_Bps: float = 22e9) -> tuple[float, float]:
    """(sequential, overlapped) seconds for all-gather + matmul over ``n`` devices.

    One hop moves total_bytes/n and costs alpha + bytes/beta. Sequential: n-1 hops then the
    whole matmul. Ring: n partial matmuls of compute_s/n, and n-1 of them each hide one hop.
    """
    hop = alpha_s + total_bytes / n / beta_Bps
    sequential = (n - 1) * hop + compute_s
    overlapped = (n - 1) * max(compute_s / n, hop) + compute_s / n
    return sequential, overlapped
