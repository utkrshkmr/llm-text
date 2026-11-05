"""Lab 18 starter: sharded data parallelism from scratch (Chapter 18).

Fill in every ``raise NotImplementedError("TODO ...")``. Tasks:
  1. flat helpers: padded_numel, flatten_tensors, shard_range, unflatten_views
  2. allreduce_flat: one all-reduce over all gradients
  3. FlatShardedOptimizer.step / state_numel: element-wise ZeRO around any optimizer
  4. FullyShardedDP: the _FlatUnit storage methods, _make_unit, _on_grad,
     finish_gradient_synchronization (the hooks are provided)
  5. accounting: layer_parallel_costs, fsdp_persistent_bytes (max_degree is provided)

Run:  LAB_IMPL=starter python -m pytest -q labs/lab18_fsdp
"""
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type
import math

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


# =========================================================================== flat helpers
def padded_numel(numel: int, world: int) -> int:
    """Smallest multiple of ``world`` that is >= ``numel``."""
    raise NotImplementedError("TODO (padded_numel): round numel up to a multiple of world")


def flatten_tensors(tensors: Sequence[torch.Tensor], world: int,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Concatenate tensors into one 1-D vector padded with zeros to a multiple of ``world``."""
    raise NotImplementedError("TODO (flatten_tensors): cat flattened tensors (cast to dtype) and zero-pad to a multiple of world")


def shard_range(total: int, rank: int, world: int) -> Tuple[int, int]:
    """[start, end) of rank's equal-size shard of a padded flat vector of length ``total``."""
    raise NotImplementedError("TODO (shard_range): equal contiguous slices of a padded vector")


def unflatten_views(flat: torch.Tensor, shapes: Sequence[torch.Size]) -> List[torch.Tensor]:
    """Split a flat vector back into views with the given shapes (padding is ignored)."""
    raise NotImplementedError("TODO (unflatten_views): return VIEWS (no copies) of flat with the given shapes")


# =========================================================================== one unit
class _FlatUnit:
    """Bookkeeping for one FSDP unit: which module attributes it owns and its shard."""

    def __init__(self, name: str, module: nn.Module,
                 slots: List[Tuple[nn.Module, str, int]], params: List[nn.Parameter],
                 rank: int, world: int, compute_dtype: Optional[torch.dtype]):
        self.name, self.module = name, module
        self.slots = slots              # (owning module, attribute name, index into params)
        self.shapes = [p.shape for p in params]
        self.numel = sum(p.numel() for p in params)
        self.rank, self.world = rank, world
        self.compute_dtype = compute_dtype or params[0].dtype
        flat = flatten_tensors(params, world, torch.float32)
        self.padded = flat.numel()
        a, b = shard_range(self.padded, rank, world)
        self.shard = nn.Parameter(flat[a:b].clone())          # fp32 master shard
        # gathered full flat buffer in compute dtype; its storage is freed when unused
        self.flat = torch.empty(self.padded, dtype=self.compute_dtype,
                                device=self.shard.device)
        self.flat.untyped_storage().resize_(0)
        self.full: Optional[torch.Tensor] = None              # autograd leaf over self.flat
        self.pending: Optional[Tuple[torch.Tensor, object]] = None   # prefetch (buf, work)
        self.grad_work: Optional[Tuple[object, torch.Tensor]] = None
        self.backward_ready = False
        self.stats = {"all_gathers": 0, "reduce_scatters": 0, "peak_gathered": 0}

    # ------------------------------------------------------------------ storage helpers
    @property
    def gathered(self) -> bool:
        return self.flat.untyped_storage().size() > 0

    def _launch_gather(self, async_op: bool):
        raise NotImplementedError("TODO (_launch_gather): cast the shard to compute_dtype, all_gather_into_tensor into a new buffer; return (buf, work)")

    def _fill_storage(self, buf: torch.Tensor):
        # Copy at the *storage* level: this does not bump autograd version counters, so
        # tensors saved for backward (views of self.flat) stay valid after a re-gather.
        raise NotImplementedError("TODO (_fill_storage): resize self.flat's untyped storage and copy buf's storage into it (do not bump version counters)")

    def prefetch(self):
        if not self.gathered and self.pending is None:
            self.pending = self._launch_gather(async_op=True)

    def unshard(self):
        raise NotImplementedError("TODO (unshard): no-op if gathered; else wait for a pending prefetch or gather synchronously, then _fill_storage")

    def reshard(self):
        raise NotImplementedError("TODO (reshard): free the storage of self.flat (resize_(0))")

    def attach_views(self, requires_grad: bool):
        """Expose the gathered buffer as the module's parameters (plain tensors)."""
        raise NotImplementedError("TODO (attach_views): detach self.flat into a leaf (requires_grad if asked), setattr views onto every slot")


# =========================================================================== the wrapper
class FullyShardedDP(nn.Module):
    """Wrap ``module`` so that every unit's parameters are sharded across ranks.

    Args:
        module: the full model (identical initialisation is NOT required; rank 0's
            weights are broadcast before sharding).
        units: submodules that become their own FSDP unit (e.g. ``model.trf_blocks``).
            Everything else forms the root unit.
        compute_dtype: dtype for communication and compute (e.g. ``torch.bfloat16``);
            master shards, gradients after reduce-scatter and optimizer state stay fp32.
        reshard_after_forward: free gathered parameters after each unit's forward
            (ZeRO-3). ``False`` keeps them until backward (more memory, fewer gathers).
        prefetch: overlap the next unit's all-gather with the current unit's compute.

    Use exactly like a module, then call :meth:`finish_gradient_synchronization` after
    ``loss.backward()`` and before ``optimizer.step()``; build the optimizer over
    ``fsdp.parameters()`` (the shards).
    """

    def __init__(self, module: nn.Module, units: Iterable[nn.Module] = (),
                 compute_dtype: Optional[torch.dtype] = None,
                 reshard_after_forward: bool = True, prefetch: bool = True):
        super().__init__()
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.reshard_after_forward = reshard_after_forward
        self.prefetch_enabled = prefetch
        self.compute_dtype = compute_dtype
        # broadcast rank 0's initial weights so all ranks shard the same model
        with torch.no_grad():
            for t in list(module.parameters()) + list(module.buffers()):
                dist.broadcast(t.data, src=0)
        unit_modules = list(units)
        self._units: List[_FlatUnit] = []
        claimed: Dict[int, nn.Parameter] = {}
        for i, um in enumerate(unit_modules):
            u = self._make_unit(f"unit{i}", um, claimed)
            if u is not None:
                self._units.append(u)
        root = self._make_unit("root", module, claimed, root=True)
        if root is not None:
            self._units.append(root)
        else:                         # no root parameters: still end the recording pass
            module.register_forward_hook(lambda *a: setattr(self, "_recording", False))
        self.module = module
        # shards are the only parameters the optimizer sees
        self.shards = nn.ParameterList([u.shard for u in self._units])
        self._fwd_order: List[_FlatUnit] = []
        self._recording = True
        for u in self._units:
            self._install_hooks(u)

    # ------------------------------------------------------------------ construction
    def _make_unit(self, name, um: nn.Module, claimed: Dict[int, nn.Parameter],
                   root: bool = False) -> _FlatUnit:
        raise NotImplementedError("TODO (_make_unit): collect this unit's not-yet-claimed params (dedupe tied ones), record slots, delete them from module._parameters, build a _FlatUnit")

    def set_compute_dtype(self, dtype: torch.dtype):
        for u in self._units:
            u.compute_dtype = dtype
            u.flat = torch.empty(u.padded, dtype=dtype, device=u.shard.device)
            u.flat.untyped_storage().resize_(0)

    # ------------------------------------------------------------------ hooks
    def _install_hooks(self, u: _FlatUnit):
        is_root = u.name == "root"

        def pre_forward(mod, args, kwargs):
            if self._recording and not is_root:
                self._fwd_order.append(u)
            u.unshard()
            u.stats["peak_gathered"] = max(u.stats["peak_gathered"],
                                           self.gathered_numel())
            full = u.attach_views(requires_grad=torch.is_grad_enabled())
            if full.requires_grad:
                u.backward_ready = False
                full.register_post_accumulate_grad_hook(lambda t, u=u: self._on_grad(u))
            if self.prefetch_enabled and not is_root:
                nxt = self._next_in_order(u, +1)
                if nxt is not None:
                    nxt.prefetch()
            return None

        def post_forward(mod, args, kwargs, out):
            grad_on = torch.is_grad_enabled()
            if not is_root and (self.reshard_after_forward or not grad_on):
                u.reshard()
            if grad_on:
                for t in _tensors(out):
                    if t.requires_grad:
                        t.register_hook(lambda g, u=u: self._pre_backward(u))
            if is_root:
                self._recording = False
                if not grad_on and self.reshard_after_forward:
                    u.reshard()
            return None

        u.module.register_forward_pre_hook(pre_forward, with_kwargs=True)
        u.module.register_forward_hook(post_forward, with_kwargs=True)

    def _next_in_order(self, u: _FlatUnit, step: int) -> Optional[_FlatUnit]:
        if u not in self._fwd_order:
            return None
        i = self._fwd_order.index(u) + step
        return self._fwd_order[i] if 0 <= i < len(self._fwd_order) else None

    def _pre_backward(self, u: _FlatUnit):
        if not u.backward_ready:
            u.backward_ready = True
            u.unshard()                     # refill the storage saved views point into
            if self.prefetch_enabled:
                prev = self._next_in_order(u, -1)
                if prev is not None:
                    prev.prefetch()

    def _on_grad(self, u: _FlatUnit):
        raise NotImplementedError("TODO (_on_grad): reduce_scatter_tensor the fp32 full gradient into a shard-sized tensor asynchronously, store (work, shard_grad, g) in u.grad_work, drop the full grad, reshard")

    # ------------------------------------------------------------------ public API
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """Wait for all reduce-scatters and install averaged gradients on the shards."""
        raise NotImplementedError("TODO (finish_gradient_synchronization): wait for every unit's reduce-scatter, divide by world, set/accumulate shard.grad, reshard anything still gathered")

    def gathered_numel(self) -> int:
        return sum(u.padded for u in self._units if u.gathered)

    def sharded_numel(self) -> int:
        return sum(u.shard.numel() for u in self._units)

    def unit_names(self) -> List[str]:
        return [u.name for u in self._units]

    def comm_stats(self) -> Dict[str, Dict[str, int]]:
        return {u.name: dict(u.stats) for u in self._units}

    @torch.no_grad()
    def full_state(self) -> Dict[str, torch.Tensor]:
        """All-gather every unit and return {qualified attr name: fp32 tensor} on every rank.

        Used for checkpointing and for testing equivalence with single-process training.
        """
        names = {id(m): n for n, m in self.module.named_modules()}
        out = {}
        for u in self._units:
            buf = torch.empty(u.padded, dtype=torch.float32, device=u.shard.device)
            dist.all_gather_into_tensor(buf, u.shard.detach())
            views = unflatten_views(buf, u.shapes)
            for mod, attr, idx in u.slots:
                key = (names[id(mod)] + "." if names[id(mod)] else "") + attr
                out[key] = views[idx].clone()
        return out


def _tensors(x):
    if torch.is_tensor(x):
        yield x
    elif isinstance(x, (list, tuple)):
        for y in x:
            yield from _tensors(y)
    elif isinstance(x, dict):
        for y in x.values():
            yield from _tensors(y)


# =========================================================================== accounting
def fsdp_persistent_bytes(n_params: float, world: int, master_bytes: int = 4,
                          optim_bytes: int = 8, grad_bytes: int = 4) -> float:
    """Per-rank bytes that live for the whole step: master shard + grad shard + Adam m, v."""
    raise NotImplementedError("TODO (fsdp_persistent_bytes): (master + grad + optimizer bytes) * n_params / world")


def fsdp_transient_bytes(unit_params: Sequence[float], compute_bytes: int = 2,
                         units_in_flight: int = 2) -> float:
    """Peak gathered-parameter memory: the largest ``units_in_flight`` units in compute dtype
    (current unit + one prefetched), plus their full gradients during backward."""
    biggest = sorted(unit_params, reverse=True)[:units_in_flight]
    return sum(biggest) * compute_bytes * 2       # params + full grad of the same unit


def layer_parallel_costs(n_params: float, tokens: float, d_model: int, n: int,
                         flops: float, bw: float, strategy: str = "dp",
                         bytes_per: int = 2, tp_allreduces: int = 1) -> Dict[str, float]:
    """Compute vs. communication time for ONE dense layer (or block) under a strategy.

    The layer has ``n_params`` weights with input/output width ``d_model`` and processes ``tokens`` tokens per step across ``n`` devices. ``flops`` is
    per-device FLOP/s and ``bw`` per-device egress bandwidth in bytes/s. Ring collectives
    move (n-1)/n of the data per rank (all-reduce: twice that).

    dp   : batch split; backward all-reduces the weight gradients.
    fsdp : batch split; weights all-gathered in forward and again in backward, gradients
           reduce-scattered.
    tp   : weights split Megatron-style; every column->row group ends in an all-reduce of a
           (tokens, d_model) activation, ``tp_allreduces`` times per forward (1 for an
           attention block or an MLP), and as many again in backward.
    """
    raise NotImplementedError("TODO (layer_parallel_costs): compute vs ring-collective time for dp / fsdp / tp (see Chapter 18)")


def max_degree(n_params: float, tokens: float, d_model: int, flops: float, bw: float,
               strategy: str, bytes_per: int = 2, tp_allreduces: int = 1,
               limit: int = 1 << 16) -> int:
    """Largest power-of-two degree n for which the backward pass is not communication bound."""
    best, n = 1, 2
    while n <= limit:
        c = layer_parallel_costs(n_params, tokens, d_model, n, flops, bw, strategy,
                                 bytes_per, tp_allreduces)
        if c["bwd_comm_bound"] or c["fwd_comm_bound"]:
            break
        best, n = n, n * 2
    return best


def allreduce_flat(module: nn.Module) -> int:
    """Average all gradients with ONE all-reduce on a flattened buffer and copy the result
    back into every ``p.grad``. Return the number of elements communicated."""
    raise NotImplementedError("TODO (allreduce_flat): torch.cat grads, dist.all_reduce, divide by world, copy back")


# ======================================================================= Task 3

from typing import Any, Dict, Iterable, List, Type

import torch
import torch.distributed as dist
import torch.nn as nn



class FlatShardedOptimizer:
    """Wrap ``optimizer_cls`` so that its state is sharded element-wise across ranks.

    Example::

        opt = FlatShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=3e-4,
                                   weight_decay=0.1)
        loss.backward(); opt.step(); opt.zero_grad()

    Parameters must already be identical across ranks (broadcast rank 0's weights first,
    e.g. with :func:`llmsys.dist.ddp.broadcast_parameters`). Gradients do NOT need to be
    synchronised beforehand: :meth:`step` does it with a reduce-scatter.
    """

    def __init__(self, params: Iterable, optimizer_cls: Type[torch.optim.Optimizer],
                 **defaults: Any):
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        groups = list(params)
        if not groups:
            raise ValueError("no parameters")
        if not isinstance(groups[0], dict):
            groups = [{"params": groups}]
        self._groups: List[Dict[str, Any]] = []
        inner_groups = []
        for g in groups:
            ps = [p for p in g["params"] if p.requires_grad]
            flat = flatten_tensors(ps, self.world, torch.float32)
            a, b = shard_range(flat.numel(), self.rank, self.world)
            shard = nn.Parameter(flat[a:b].clone())
            self._groups.append(dict(params=ps, shapes=[p.shape for p in ps],
                                     numel=flat.numel(), shard=shard))
            hp = {k: v for k, v in g.items() if k != "params"}
            inner_groups.append({"params": [shard], **hp})
        self.inner = optimizer_cls(inner_groups, **defaults)

    # mimic the parts of the Optimizer API that schedulers and loops use
    @property
    def param_groups(self):
        return self.inner.param_groups

    @property
    def state(self):
        return self.inner.state

    def zero_grad(self, set_to_none: bool = True):
        for g in self._groups:
            for p in g["params"]:
                p.grad = None if set_to_none else (p.grad.zero_() if p.grad is not None else None)
            g["shard"].grad = None

    @torch.no_grad()
    def step(self, closure=None):
        raise NotImplementedError("TODO (step): reduce-scatter flattened grads into the shard (average), inner.step(), all-gather shards and copy back into params")

    # ------------------------------------------------------------------ accounting
    def state_numel(self) -> int:
        """Elements of optimizer state held by THIS rank (e.g. Adam m + v)."""
        raise NotImplementedError("TODO (state_numel): sum numel of tensor-valued (dim>0) entries in self.inner.state")

    def shard_numel(self) -> int:
        return sum(g["shard"].numel() for g in self._groups)

    def state_dict(self):
        """Rank-local (sharded) state; save one file per rank, as DeepSpeed does."""
        return {"rank": self.rank, "world": self.world, "inner": self.inner.state_dict(),
                "shards": [g["shard"].detach().clone() for g in self._groups]}

    def load_state_dict(self, sd):
        assert sd["world"] == self.world, "resharding to a new world size is not supported"
        self.inner.load_state_dict(sd["inner"])
        for g, s in zip(self._groups, sd["shards"]):
            g["shard"].data.copy_(s)
