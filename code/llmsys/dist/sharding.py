"""A calculus for sharded computation: meshes, shard specs and the matmul rules (Chapter 19).

The notation. A ``Mesh`` names the device axes (``("dp", "tp")`` with sizes ``(8, 2)``). A
``ShardSpec`` says, for every dimension of a tensor, which mesh axis it is split over (or ``None``
for replicated), plus the set of mesh axes over which the tensor is a *partial sum* (each device
holds one summand of the logical value).

The rules. ``matmul_plan`` takes the specs of ``A (M, K)`` and ``B (K, N)`` and derives the
output spec and the collectives needed, following the four cases of the chapter:

    I   no mesh axis touches the contracting dimension        -> local product
    II  one operand is sharded on K                            -> all-gather it, or slice the other
                                                                  operand and produce a partial sum
    III both operands are sharded on K over the same axis     -> local partial products, then
                                                                  all-reduce or reduce-scatter
    IV  one mesh axis appears on two free dimensions           -> conflict: all-gather one operand

``reshard_plan`` derives the collective that moves a tensor from one spec to another (all-gather,
reduce-scatter, all-reduce, all-to-all or a free local slice), ``transpose_collective`` gives the
adjoint used by the backward pass, and the alpha-beta ``Link`` presets turn byte counts into time.
Everything here is pure Python so the same plans can be checked on a laptop and used to predict
measurements on GPUs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# =========================================================================== mesh and specs
@dataclass(frozen=True)
class Mesh:
    """Named device axes. ``Mesh(("dp", "tp"), (4, 2))`` is 8 devices arranged 4 x 2."""

    axes: Tuple[str, ...]
    sizes: Tuple[int, ...]

    def __post_init__(self):
        if len(self.axes) != len(self.sizes) or len(set(self.axes)) != len(self.axes):
            raise ValueError("mesh axes must be unique and match sizes")

    def size(self, axis: str) -> int:
        return self.sizes[self.axes.index(axis)]

    @property
    def ndevices(self) -> int:
        return math.prod(self.sizes)

    def group_size(self, axes: Iterable[str]) -> int:
        return math.prod(self.size(a) for a in axes)


@dataclass(frozen=True)
class ShardSpec:
    """How one tensor is laid out: ``dims[i]`` is the mesh axis that splits tensor dim ``i``
    (``None`` = replicated along that dim); ``partial`` lists mesh axes over which the local
    tensors are summands of the logical value.

    ``ShardSpec.parse("-,tp")`` is a (M, N) tensor with N split over ``tp``;
    ``ShardSpec.parse("-,-|tp")`` is a replicated-shape tensor that is a partial sum over ``tp``.
    """

    dims: Tuple[Optional[str], ...]
    partial: frozenset = field(default_factory=frozenset)

    @classmethod
    def parse(cls, text: str) -> "ShardSpec":
        body, _, part = text.partition("|")
        dims = tuple(None if d.strip() in ("-", "") else d.strip() for d in body.split(","))
        partial = frozenset(p.strip() for p in part.split(",") if p.strip())
        return cls(dims, partial)

    @classmethod
    def replicated(cls, ndim: int) -> "ShardSpec":
        return cls((None,) * ndim)

    def __str__(self) -> str:
        body = ",".join("-" if d is None else d for d in self.dims)
        return f"[{body}]" + (f"+partial{{{','.join(sorted(self.partial))}}}" if self.partial else "")

    @property
    def ndim(self) -> int:
        return len(self.dims)

    @property
    def shard_axes(self) -> Tuple[str, ...]:
        return tuple(d for d in self.dims if d is not None)

    def validate(self, mesh: Mesh) -> None:
        axes = self.shard_axes
        if len(set(axes)) != len(axes):
            raise ValueError(f"{self}: a mesh axis may split at most one tensor dimension")
        for a in axes + tuple(self.partial):
            if a not in mesh.axes:
                raise ValueError(f"{self}: unknown mesh axis {a!r}")
        if set(axes) & self.partial:
            raise ValueError(f"{self}: an axis cannot both shard and partial-sum the tensor")

    def dim_of(self, axis: str) -> Optional[int]:
        """Tensor dimension split over ``axis`` or ``None``."""
        return self.dims.index(axis) if axis in self.dims else None

    def with_dim(self, i: int, axis: Optional[str]) -> "ShardSpec":
        dims = list(self.dims)
        dims[i] = axis
        return ShardSpec(tuple(dims), self.partial)

    def with_partial(self, *axes: str) -> "ShardSpec":
        return ShardSpec(self.dims, self.partial | frozenset(axes))

    def without_partial(self, *axes: str) -> "ShardSpec":
        return ShardSpec(self.dims, self.partial - frozenset(axes))

    def local_shape(self, shape: Sequence[int], mesh: Mesh) -> Tuple[int, ...]:
        """Shape of one device's piece (dimensions must divide evenly)."""
        out = []
        for n, a in zip(shape, self.dims):
            if a is None:
                out.append(n)
            else:
                p = mesh.size(a)
                if n % p:
                    raise ValueError(f"dimension {n} does not divide over {a} ({p})")
                out.append(n // p)
        return tuple(out)

    def local_bytes(self, shape: Sequence[int], mesh: Mesh, dtype_bytes: int = 2) -> int:
        return math.prod(self.local_shape(shape, mesh)) * dtype_bytes

    def replication(self, mesh: Mesh) -> int:
        """How many devices hold each element (1 = fully sharded over the mesh)."""
        used = set(self.shard_axes) | self.partial
        return mesh.group_size(a for a in mesh.axes if a not in used)


# =========================================================================== cost model
@dataclass(frozen=True)
class Link:
    """alpha-beta link: a message of n bytes costs alpha + n / bandwidth seconds."""

    name: str
    alpha: float            # seconds per message (launch + first byte)
    bandwidth: float        # bytes per second, per direction, per device


PCIE_GEN4 = Link("PCIe Gen4 x16, no NVLink (2x A100 box)", alpha=25e-6, bandwidth=22e9)
NVLINK_A100 = Link("NVLink 3 (A100 SXM, 300 GB/s per direction)", alpha=10e-6, bandwidth=300e9)
NVLINK_H100 = Link("NVLink 4 (H100 SXM, 450 GB/s per direction)", alpha=10e-6, bandwidth=450e9)
IB_400G = Link("400 Gb/s InfiniBand", alpha=20e-6, bandwidth=50e9)

COLLECTIVES = ("all_gather", "reduce_scatter", "all_reduce", "all_to_all", "slice")
# A "slice" is any free local re-layout: keeping one block of a replicated tensor, or zero-padding
# one block into the summand of a partial tensor (its transpose).


def bytes_per_rank(kind: str, nbytes: int, p: int) -> float:
    """Bytes each device sends for a collective over ``p`` devices with ring algorithms.

    ``nbytes`` is the size of the *logical* (full) tensor for all-gather, reduce-scatter and
    all-reduce, and the size of one device's buffer for all-to-all. A local slice is free.
    """
    if p == 1 or kind == "slice":
        return 0.0
    if kind in ("all_gather", "reduce_scatter", "all_to_all"):
        return (p - 1) / p * nbytes
    if kind == "all_reduce":            # reduce-scatter followed by all-gather
        return 2 * (p - 1) / p * nbytes
    raise ValueError(kind)


def collective_time(kind: str, nbytes: int, p: int, link: Link) -> float:
    """Ring alpha-beta time: (p-1) latency terms, or 2(p-1) for an all-reduce."""
    if p == 1 or kind == "slice":
        return 0.0
    steps = 2 * (p - 1) if kind == "all_reduce" else (p - 1)
    return steps * link.alpha + bytes_per_rank(kind, nbytes, p) / link.bandwidth


def transpose_collective(kind: str) -> str:
    """The adjoint of each collective, i.e. what the backward pass issues in its place.

    all-gather <-> reduce-scatter (A^T of 'copy my slice to everyone' is 'sum everyone's
    contribution to my slice'); all-reduce and all-to-all are self-adjoint (J^T = J, a permutation's
    transpose is its inverse); the adjoint of a free slice is a zero-padded sum, which the backward
    pass realises as a reduce-scatter only when the gradient arrives as a partial sum.
    """
    return {"all_gather": "reduce_scatter", "reduce_scatter": "all_gather",
            "all_reduce": "all_reduce", "all_to_all": "all_to_all", "slice": "slice"}[kind]


@dataclass(frozen=True)
class Collective:
    kind: str
    axis: str                # mesh axis whose devices take part
    tensor: str              # which tensor moves ("A", "B", "C", or a user name)
    nbytes: int              # logical tensor bytes (see bytes_per_rank)
    note: str = ""

    def bytes_per_rank(self, mesh: Mesh) -> float:
        return bytes_per_rank(self.kind, self.nbytes, mesh.size(self.axis))

    def time(self, mesh: Mesh, link: Link) -> float:
        return collective_time(self.kind, self.nbytes, mesh.size(self.axis), link)

    def transpose(self) -> "Collective":
        return Collective(transpose_collective(self.kind), self.axis, "d" + self.tensor, self.nbytes,
                          "backward")

    def __str__(self) -> str:
        return f"{self.kind}({self.tensor}, axis={self.axis}, {self.nbytes / 2**20:.2f} MiB)"


# =========================================================================== resharding
def reshard_plan(src: ShardSpec, dst: ShardSpec, mesh: Mesh, shape: Sequence[int],
                 dtype_bytes: int = 2, name: str = "X") -> List[Collective]:
    """Collectives that move a tensor from layout ``src`` to layout ``dst``, one mesh axis at a time.

    partial -> replicated          all-reduce
    partial -> sharded on dim j    reduce-scatter along j
    sharded on i -> replicated     all-gather along i
    sharded on i -> sharded on j   all-to-all (each device keeps 1/p of its data, sends the rest)
    replicated -> sharded on j     local slice, no bytes
    sharded on i -> partial        local zero-pad, no bytes (the transpose of the slice above)
    replicated -> partial          not a layout change anyone wants (raises)
    """
    src.validate(mesh)
    dst.validate(mesh)
    if len(shape) != src.ndim or src.ndim != dst.ndim:
        raise ValueError("shape and specs must have the same rank")
    nbytes = math.prod(shape) * dtype_bytes
    steps: List[Collective] = []
    cur = src
    for axis in mesh.axes:
        p = mesh.size(axis)
        src_dim, dst_dim = cur.dim_of(axis), dst.dim_of(axis)
        src_part, dst_part = axis in cur.partial, axis in dst.partial
        # the tensor as the *group over this axis* sees it: all other axes already applied
        other = cur.local_bytes(shape, mesh, dtype_bytes) * (p if src_dim is not None else 1)
        if src_part and not dst_part:
            if dst_dim is None:
                steps.append(Collective("all_reduce", axis, name, other))
            else:
                steps.append(Collective("reduce_scatter", axis, name, other, f"along dim {dst_dim}"))
            cur = cur.without_partial(axis).with_dim(dst_dim, axis) if dst_dim is not None \
                else cur.without_partial(axis)
        elif not src_part and dst_part:
            if src_dim is None:
                raise ValueError(f"{src} -> {dst}: nothing turns a replicated tensor into a partial sum")
            # sharded -> partial: zero-pad my block into a full-size summand (the transpose of slice)
            steps.append(Collective("slice", axis, name, 0, f"zero-pad dim {src_dim} into a summand"))
            cur = cur.with_dim(src_dim, None).with_partial(axis)
        elif src_dim is not None and dst_dim is None:
            steps.append(Collective("all_gather", axis, name, other, f"along dim {src_dim}"))
            cur = cur.with_dim(src_dim, None)
        elif src_dim is not None and dst_dim is not None and src_dim != dst_dim:
            local = cur.local_bytes(shape, mesh, dtype_bytes)
            steps.append(Collective("all_to_all", axis, name, local, f"dim {src_dim} -> dim {dst_dim}"))
            cur = cur.with_dim(src_dim, None).with_dim(dst_dim, axis)
        elif src_dim is None and dst_dim is not None and not src_part:
            steps.append(Collective("slice", axis, name, 0, f"keep 1/{p} of dim {dst_dim}"))
            cur = cur.with_dim(dst_dim, axis)
    assert cur == dst, (cur, dst)
    return steps


def cotangent_spec(spec: ShardSpec, mesh: Mesh) -> ShardSpec:
    """Layout of the gradient of a tensor with layout ``spec``, axis by axis.

    A sharded dimension stays sharded (each device owns the gradient of its block). A partial sum
    becomes replicated: the derivative of f(sum_r S_r) with respect to any summand S_r is the same
    full gradient on every device. A replicated axis becomes partial: the devices along it consumed
    the tensor in different local programs, so each holds one summand of the total gradient. (When
    every device ran the same downstream program the gradient is already replicated and the
    partial -> replicated all-reduce is a no-op; Megatron's ``f``/``g`` operators exploit that.)
    """
    sharded = set(spec.shard_axes)
    return ShardSpec(spec.dims, frozenset(a for a in mesh.axes
                                          if a not in sharded and a not in spec.partial))


def transpose_reshard(src: ShardSpec, dst: ShardSpec, mesh: Mesh, shape: Sequence[int],
                      dtype_bytes: int = 2, name: str = "dX") -> List[Collective]:
    """Collectives of the backward pass of a resharding ``src -> dst``.

    The backward moves the gradient from the cotangent layout of ``dst`` to that of ``src``; the
    chapter's derivation ("transpose the forward program") says nothing else. The result agrees
    with ``transpose_collective`` step by step: an all-gather (shard -> replicated) becomes a
    reduce-scatter (partial -> shard), a reduce-scatter becomes an all-gather, an all-reduce and an
    all-to-all stay what they are, and a free slice becomes a free zero-pad.
    """
    return reshard_plan(cotangent_spec(dst, mesh), cotangent_spec(src, mesh), mesh, shape,
                        dtype_bytes, name)


# =========================================================================== the matmul rules
@dataclass
class MatmulPlan:
    case: str                       # "I", "II-gather", "II-partial", "III", "IV", ...
    steps: List[Collective]         # in program order (forward)
    out: ShardSpec                  # spec of C after the steps
    explanation: str = ""

    def comm_bytes(self, mesh: Mesh) -> float:
        return sum(s.bytes_per_rank(mesh) for s in self.steps)

    def time(self, mesh: Mesh, link: Link) -> float:
        return sum(s.time(mesh, link) for s in self.steps)

    def kinds(self) -> List[str]:
        return [s.kind for s in self.steps if s.kind != "slice"]

    def backward_steps(self) -> List[Collective]:
        """Collectives of the transposed program, in backward order."""
        return [s.transpose() for s in reversed(self.steps)]

    def __str__(self) -> str:
        body = "; ".join(str(s) for s in self.steps) or "no communication"
        return f"case {self.case}: {body} -> C {self.out}"


def _finish_partial(out: ShardSpec, axis: str, finish: str, shape_c: Sequence[int], mesh: Mesh,
                    dtype_bytes: int, steps: List[Collective]) -> ShardSpec:
    """Resolve a partial sum over ``axis``: leave it, all-reduce it, or reduce-scatter it."""
    nbytes = out.local_bytes(shape_c, mesh, dtype_bytes)      # one summand = the group's view of C
    if finish == "partial":
        return out
    if finish == "all_reduce":
        steps.append(Collective("all_reduce", axis, "C", nbytes))
        return out.without_partial(axis)
    if finish.startswith("reduce_scatter"):
        dim = int(finish.split(":")[1]) if ":" in finish else 0
        if out.dims[dim] is not None:
            raise ValueError(f"cannot reduce-scatter along dim {dim}: already sharded over {out.dims[dim]}")
        steps.append(Collective("reduce_scatter", axis, "C", nbytes, f"along dim {dim}"))
        return out.without_partial(axis).with_dim(dim, axis)
    raise ValueError(finish)


def matmul_plan(a: ShardSpec, b: ShardSpec, mesh: Mesh, shape_a: Sequence[int],
                shape_b: Sequence[int], dtype_bytes: int = 2, finish: str = "all_reduce",
                strategy: str = "auto") -> MatmulPlan:
    """Derive the collectives and output spec of ``C = A @ B`` for 2-D ``A (M, K)``, ``B (K, N)``.

    ``finish`` says what to do with a partial-sum output: ``"partial"`` (leave it for the next
    op), ``"all_reduce"`` (replicate it over the axis) or ``"reduce_scatter:d"`` (shard output
    dim ``d`` over the axis). ``strategy`` picks the case-II variant: ``"gather"`` the operand that
    is sharded on K, ``"partial"`` slice the other operand and defer the reduction, or ``"auto"``
    take whichever moves fewer bytes (counting the deferred reduction).

    Byte counts are always *the tensor as the communicating group sees it*: the local block times
    the group size, so that a collective over one axis of a 2-D mesh is not charged for the
    sharding the other axis already did.
    """
    a.validate(mesh)
    b.validate(mesh)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul_plan handles 2-D operands; fold batch dims into M first")
    M, K = shape_a
    K2, N = shape_b
    if K != K2:
        raise ValueError(f"inner dimensions differ: {K} vs {K2}")
    shape_c = (M, N)
    steps: List[Collective] = []

    def gbytes(spec: ShardSpec, shape: Sequence[int], axis: str) -> int:
        return spec.local_bytes(shape, mesh, dtype_bytes) * mesh.size(axis)

    def gather(name: str, dim: int, axis: str, note: str) -> None:
        """Append the all-gather of operand ``name`` along ``dim`` over ``axis`` and update it."""
        nonlocal a, b
        spec, shape = (a, shape_a) if name == "A" else (b, shape_b)
        steps.append(Collective("all_gather", axis, name, gbytes(spec, shape, axis), note))
        if name == "A":
            a = a.with_dim(dim, None)
        else:
            b = b.with_dim(dim, None)

    def smaller(axis: str) -> str:
        """Which operand costs fewer bytes to all-gather over ``axis``."""
        return "A" if gbytes(a, shape_a, axis) <= gbytes(b, shape_b, axis) else "B"

    # --- partial-sum operands: (sum_r A_r) B = sum_r (A_r B) holds as long as B is not itself a
    # partial sum and does not use that axis; otherwise resolve the partial operand first.
    for name in ("A", "B"):
        spec, other, shape = (a, b, shape_a) if name == "A" else (b, a, shape_b)
        for axis in sorted(spec.partial):
            if other.partial or axis in other.shard_axes:
                steps.append(Collective("all_reduce", axis, name, gbytes(spec, shape, axis),
                                        "resolve partial operand"))
                if name == "A":
                    a = a.without_partial(axis)
                else:
                    b = b.without_partial(axis)
    carried_partial = a.partial | b.partial

    # --- contracting dimension sharded on both operands but over different axes: gather the
    # cheaper operand along K so that one axis remains
    if a.dims[1] is not None and b.dims[0] is not None and a.dims[1] != b.dims[0]:
        if gbytes(a, shape_a, a.dims[1]) <= gbytes(b, shape_b, b.dims[0]):
            gather("A", 1, a.dims[1], "mismatched K axes")
        else:
            gather("B", 0, b.dims[0], "mismatched K axes")
    aM, aK = a.dims
    bK, bN = b.dims

    # --- case III: both sharded on K over the same axis -> local partial products
    if aK is not None and aK == bK:
        axis = aK
        if aM is not None and aM == bN:         # free dims collide on another axis (case IV inside)
            name = smaller(aM)
            gather(name, 0 if name == "A" else 1, aM, f"conflict on {aM}: gather {name}")
            aM, bN = a.dims[0], b.dims[1]
        out = ShardSpec((aM, bN), carried_partial | {axis})
        out = _finish_partial(out, axis, finish, shape_c, mesh, dtype_bytes, steps)
        return MatmulPlan("III", steps, out,
                          f"both operands split K over {axis}: local partial products, then {finish}")

    # --- case II: exactly one operand sharded on K
    if (aK is None) != (bK is None):
        axis = aK or bK
        name = "A" if aK else "B"
        other_free = bN if aK else aM
        if other_free == axis:
            # the other operand's free dim is split over the same axis: no local slice of it
            # matches our K block on every device, so the only option is to gather our operand
            gather(name, 1 if aK else 0, axis, "along K")
            return MatmulPlan("II-gather", steps, ShardSpec((aM, bN), carried_partial),
                              f"{name} is split on K over {axis} and the free dim of the other "
                              f"operand uses {axis} too: all-gather {name}")
        if aM is not None and aM == bN:         # free dims collide on another axis
            n2 = smaller(aM)
            gather(n2, 0 if n2 == "A" else 1, aM, f"conflict on {aM}: gather {n2}")
            aM, bN = a.dims[0], b.dims[1]
        spec, shape = (a, shape_a) if aK else (b, shape_b)
        gather_cost = bytes_per_rank("all_gather", gbytes(spec, shape, axis), mesh.size(axis))
        c_group = ShardSpec((aM, bN)).local_bytes(shape_c, mesh, dtype_bytes)
        reduce_kind = "all_reduce" if finish == "all_reduce" else "reduce_scatter"
        partial_cost = 0.0 if finish == "partial" else \
            bytes_per_rank(reduce_kind, c_group, mesh.size(axis))
        if strategy == "gather" or (strategy == "auto" and gather_cost <= partial_cost):
            gather(name, 1 if aK else 0, axis, "along K")
            return MatmulPlan("II-gather", steps, ShardSpec((aM, bN), carried_partial),
                              f"{name} is split on K over {axis}: all-gather it, then local")
        steps.append(Collective("slice", axis, "B" if aK else "A", 0, "take the matching K block"))
        out = ShardSpec((aM, bN), carried_partial | {axis})
        out = _finish_partial(out, axis, finish, shape_c, mesh, dtype_bytes, steps)
        return MatmulPlan("II-partial", steps, out,
                          f"{name} is split on K over {axis}: slice the other operand, "
                          f"partial products, then {finish}")

    # --- no axis on K: case I (local) unless the two free dims collide (case IV)
    if aM is not None and aM == bN:
        axis = aM
        name = smaller(axis)
        gather(name, 0 if name == "A" else 1, axis, f"conflict on {axis}: gather {name}")
        return MatmulPlan("IV", steps, ShardSpec((a.dims[0], b.dims[1]), carried_partial),
                          f"{axis} splits both M and N: a device would only hold diagonal blocks "
                          f"of C; all-gather {name} (the smaller operand)")
    return MatmulPlan("I", steps, ShardSpec((aM, bN), carried_partial),
                      "no mesh axis touches K: every device multiplies its own blocks")


# =========================================================================== layer programs
@dataclass
class ProgramStep:
    op: str
    plan: Optional[MatmulPlan] = None
    reshard: Optional[List[Collective]] = None

    def collectives(self) -> List[Collective]:
        if self.plan is not None:
            return list(self.plan.steps)
        return list(self.reshard or [])


def mlp_program(x: ShardSpec, w1: ShardSpec, w2: ShardSpec, mesh: Mesh, tokens: int, d_model: int,
                d_ff: int, dtype_bytes: int = 2, finish2: str = "all_reduce",
                x_out: Optional[ShardSpec] = None, finish1: str = "all_reduce") -> List[ProgramStep]:
    """The two matmuls of an MLP, ``H = X W1`` then ``Y = act(H) W2``, as a sharded program.

    ``finish1``/``finish2`` resolve each product's partial sum if it has one (``"all_reduce"`` is
    Megatron TP, ``"reduce_scatter:0"`` is sequence parallelism, and the 2-D weight-stationary
    layout of the inference section reduce-scatters H over one mesh axis). ``x_out`` optionally
    re-shards Y to the layout the next layer expects, so that the per-layer traffic is complete.
    The activation between the two products is elementwise, so it needs no collective of its own.
    """
    steps = []
    p1 = matmul_plan(x, w1, mesh, (tokens, d_model), (d_model, d_ff), dtype_bytes, finish=finish1)
    steps.append(ProgramStep("H = X @ W1", plan=p1))
    p2 = matmul_plan(p1.out, w2, mesh, (tokens, d_ff), (d_ff, d_model), dtype_bytes, finish=finish2)
    steps.append(ProgramStep("Y = act(H) @ W2", plan=p2))
    if x_out is not None and p2.out != x_out:
        steps.append(ProgramStep("re-shard Y", reshard=reshard_plan(p2.out, x_out, mesh,
                                                                    (tokens, d_model), dtype_bytes, "Y")))
    return steps


def program_summary(steps: Sequence[ProgramStep], mesh: Mesh, link: Link) -> Dict[str, float]:
    """Forward bytes per device, forward time and the collectives of forward and backward."""
    fwd = [c for s in steps for c in s.collectives() if c.kind != "slice"]
    bwd = [c.transpose() for c in reversed(fwd)]
    return {
        "forward": [c.kind for c in fwd],
        "backward": [c.kind for c in bwd],
        "fwd_bytes_per_rank": sum(c.bytes_per_rank(mesh) for c in fwd),
        "fwd_time": sum(c.time(mesh, link) for c in fwd),
        "bwd_time": sum(c.time(mesh, link) for c in bwd),
    }
