"""The four sharded matmul cases with PyTorch DTensor (Chapter 24).

``torch.distributed.tensor`` is the production form of the chapter's notation: a ``DeviceMesh`` is
our ``Mesh``, and a tuple of placements, one per mesh axis (``Shard(d)``, ``Replicate()``,
``Partial()``), is our ``ShardSpec`` with the roles reversed (a ``ShardSpec`` lists, per tensor
dimension, the mesh axis that splits it; a placement list says, per mesh axis, which tensor
dimension it splits). ``spec_to_placements`` / ``placements_to_spec`` convert between the two.

DTensor applies the same rules we derived: ``a @ b`` on DTensors chooses the local matmul when no
mesh axis touches K, produces a ``Partial`` when both operands are split on K, and inserts the
redistributions the rules demand otherwise. ``issued_collectives`` records which collectives a
block of DTensor code actually ran (through ``CommDebugMode``), so that the plan from
``llmsys.dist.sharding`` can be checked against the real thing.

Everything runs on gloo with ``init_device_mesh("cpu", ...)`` as well as on NCCL.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist

from .sharding import Mesh, ShardSpec

try:
    from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
    from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
    from torch.distributed.tensor.debug import CommDebugMode
    HAS_DTENSOR = True
except ImportError:                                   # pragma: no cover - old torch
    HAS_DTENSOR = False


# =========================================================================== mesh + placements
def make_mesh(mesh: Mesh, device_type: str = "cpu") -> "DeviceMesh":
    """A DeviceMesh with the axes and sizes of our ``Mesh`` (must match the world size)."""
    if mesh.ndevices != dist.get_world_size():
        raise ValueError(f"mesh has {mesh.ndevices} devices, world size is {dist.get_world_size()}")
    return init_device_mesh(device_type, tuple(mesh.sizes), mesh_dim_names=tuple(mesh.axes))


def spec_to_placements(spec: ShardSpec, mesh: Mesh) -> Tuple:
    """Per mesh axis: Shard(d) if the spec splits tensor dim d over it, Partial() if the spec is a
    partial sum over it, Replicate() otherwise."""
    out = []
    for axis in mesh.axes:
        d = spec.dim_of(axis)
        if d is not None:
            out.append(Shard(d))
        elif axis in spec.partial:
            out.append(Partial())
        else:
            out.append(Replicate())
    return tuple(out)


def placements_to_spec(placements: Sequence, mesh: Mesh, ndim: int) -> ShardSpec:
    dims: List = [None] * ndim
    partial = set()
    for axis, p in zip(mesh.axes, placements):
        if isinstance(p, Shard):
            dims[p.dim] = axis
        elif isinstance(p, Partial):
            partial.add(axis)
    return ShardSpec(tuple(dims), frozenset(partial))


def shard_like(full: torch.Tensor, spec: ShardSpec, mesh: Mesh, dmesh: "DeviceMesh") -> "DTensor":
    """Distribute a (replicated, identical on every rank) full tensor according to ``spec``."""
    return distribute_tensor(full, dmesh, list(spec_to_placements(spec, mesh)))


def spec_of(t: "DTensor", mesh: Mesh) -> ShardSpec:
    return placements_to_spec(t.placements, mesh, t.ndim)


# =========================================================================== what DTensor ran
_NAMES = {"all_gather_into_tensor": "all_gather", "all_gather": "all_gather",
          "reduce_scatter_tensor": "reduce_scatter", "all_reduce": "all_reduce",
          "all_to_all_single": "all_to_all", "all_to_all": "all_to_all"}


def issued_collectives(fn: Callable[[], object]) -> Tuple[object, Dict[str, int]]:
    """Run ``fn()`` under CommDebugMode and return (result, {collective kind: count})."""
    with CommDebugMode() as mode:
        out = fn()
    counts: Dict[str, int] = {}
    for op, n in mode.get_comm_counts().items():
        name = str(op).split(".")[-1].replace("'", "").replace(")", "").split("=")[-1]
        kind = _NAMES.get(name, name)
        counts[kind] = counts.get(kind, 0) + int(n)
    return out, counts


# =========================================================================== the four cases
def dtensor_matmul(a: "DTensor", b: "DTensor", finish: str = "partial") -> "DTensor":
    """``a @ b`` with DTensor, then resolve a Partial output: ``"partial"`` leaves it,
    ``"all_reduce"`` replicates it, ``"reduce_scatter:d"`` shards dim d over the partial axis."""
    c = a @ b
    if finish == "partial" or not any(isinstance(p, Partial) for p in c.placements):
        return c
    if finish == "all_reduce":
        new = [Replicate() if isinstance(p, Partial) else p for p in c.placements]
    elif finish.startswith("reduce_scatter"):
        d = int(finish.split(":")[1]) if ":" in finish else 0
        new = [Shard(d) if isinstance(p, Partial) else p for p in c.placements]
    else:
        raise ValueError(finish)
    return c.redistribute(c.device_mesh, new)


def four_cases(mesh: Mesh, M: int = 8, K: int = 16, N: int = 12) -> Dict[str, Tuple[ShardSpec, ShardSpec, str]]:
    """Operand specs of the chapter's four cases on a one-axis mesh ``axis``."""
    ax = mesh.axes[-1]
    S = ShardSpec.parse
    return {
        "I": (S("-,-"), S(f"-,{ax}"), "partial"),              # local, C [-,tp]
        "II-gather": (S(f"-,{ax}"), S(f"-,{ax}"), "partial"),  # gather A along K, C [-,tp]
        "II-partial": (S("-,-"), S(f"{ax},-"), "all_reduce"),  # slice A, partial, all-reduce
        "III": (S(f"-,{ax}"), S(f"{ax},-"), "all_reduce"),     # partial products, all-reduce
        "III-rs": (S(f"-,{ax}"), S(f"{ax},-"), "reduce_scatter:0"),
        "IV": (S(f"{ax},-"), S(f"-,{ax}"), "partial"),         # conflict: gather one operand
    }
