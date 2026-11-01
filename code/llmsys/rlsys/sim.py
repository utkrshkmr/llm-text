"""A small event-driven simulator of an RL training pipeline.

It models the two expensive stages of an RLVR step:

* **generation**: ``n_slots`` concurrent decode slots (continuous batching across all
  rollout engines). Each slot decodes one response at ``tokens_per_sec`` tokens/s.
  Response lengths are heavy-tailed (log-normal), which is the source of the
  "long-tail" problem.
* **training**: cost proportional to the number of tokens trained on.

and four scheduling policies:

``colocated``  generation and training time-share the same GPUs (sequential).
``sync``       separate generation and training pools, but strictly on-policy:
               each waits for the other (idle bubbles in both pools).
``async``      separate pools, generation of batch k+1 overlaps training on batch k
               (one-step off-policy, staleness <= ``max_staleness``).
``partial``    like ``sync`` but ``oversample * batch_size`` responses are kept in
               flight and each generation phase stops once ``batch_size`` of them
               are finished; unfinished responses keep their progress and resume
               in the next phase under *newer* weights (mixed-policy sequences).

Everything is deterministic given the seed so results are reproducible in tests.
"""
from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SimConfig:
    batch_size: int = 128           # responses consumed per optimizer step
    n_steps: int = 20
    n_slots: int = 128              # concurrent decode slots in the generation pool
    tokens_per_sec: float = 50.0    # per-slot decode speed (memory-bound regime)
    mean_len: float = 1000.0        # mean response length (tokens)
    sigma: float = 1.0              # log-normal shape; larger = heavier tail
    max_len: int = 16_384           # generation cap
    train_sec_per_token: float = 2e-4
    switch_sec: float = 2.0         # colocated: offload/onload engines, re-shard weights
    sync_sec: float = 1.0           # weight broadcast trainer -> rollout engines
    max_staleness: int = 1
    oversample: float = 2.0         # partial: responses kept in flight per consumed batch
    seed: int = 0


@dataclass
class Rollout:
    rid: int
    length: int
    remaining: float
    versions: List[int] = field(default_factory=list)  # policy versions that produced it


@dataclass
class SimResult:
    policy: str
    wall_time: float
    steps: int
    samples: int
    gen_busy: float          # slot-seconds spent decoding
    gen_capacity: float      # slot-seconds available
    train_busy: float
    staleness: List[int]
    mixed_policy_fraction: float

    @property
    def samples_per_sec(self) -> float:
        return self.samples / self.wall_time

    @property
    def gen_utilization(self) -> float:
        return self.gen_busy / self.gen_capacity if self.gen_capacity else 0.0

    @property
    def train_utilization(self) -> float:
        return self.train_busy / self.wall_time

    @property
    def mean_staleness(self) -> float:
        return sum(self.staleness) / max(1, len(self.staleness))

    def summary(self) -> Dict[str, float]:
        return dict(policy=self.policy, wall_time=round(self.wall_time, 1),
                    samples_per_sec=round(self.samples_per_sec, 3),
                    gen_util=round(self.gen_utilization, 3),
                    train_util=round(self.train_utilization, 3),
                    mean_staleness=round(self.mean_staleness, 3),
                    mixed_policy_frac=round(self.mixed_policy_fraction, 3))


def sample_lengths(n: int, cfg: SimConfig, rng: random.Random) -> List[int]:
    """Log-normal lengths with the requested mean, clipped to ``max_len``."""
    mu = math.log(cfg.mean_len) - cfg.sigma ** 2 / 2
    return [max(1, min(cfg.max_len, int(rng.lognormvariate(mu, cfg.sigma)))) for _ in range(n)]


def run_generation(pending: List[Rollout], n_slots: int, tps: float, stop_after: Optional[int],
                   version: int):
    """Continuous batching on ``n_slots`` slots, FIFO admission.

    Runs until every rollout finishes or until ``stop_after`` have finished.
    Returns (elapsed, finished, unfinished, busy_slot_seconds).
    Unfinished rollouts have ``remaining`` reduced by the work done.
    """
    queue = list(pending)
    heap: List = []          # (finish_time, seq, rollout, start_time)
    t, busy, seq = 0.0, 0.0, 0
    finished: List[Rollout] = []

    def admit(now):
        nonlocal seq
        while queue and len(heap) < n_slots:
            r = queue.pop(0)
            if not r.versions or r.versions[-1] != version:
                r.versions.append(version)
            heapq.heappush(heap, (now + r.remaining / tps, seq, r, now))
            seq += 1

    admit(0.0)
    while heap:
        t, _, r, start = heapq.heappop(heap)
        busy += t - start
        r.remaining = 0.0
        finished.append(r)
        if stop_after is not None and len(finished) >= stop_after:
            break
        admit(t)
    unfinished = []
    for (ft, _, r, start) in heap:       # still decoding: account partial progress
        busy += t - start
        r.remaining = max(0.0, (ft - t) * tps)
        unfinished.append(r)
    return t, finished, unfinished + queue, busy


def simulate(policy: str, cfg: SimConfig = SimConfig()) -> SimResult:
    rng = random.Random(cfg.seed)
    rid = 0

    def new_rollouts(n):
        nonlocal rid
        out = []
        for L in sample_lengths(n, cfg, rng):
            out.append(Rollout(rid, L, float(L)))
            rid += 1
        return out

    wall = gen_busy = gen_cap = train_busy = 0.0
    stale: List[int] = []
    samples = mixed = 0
    version = 0

    if policy in ("colocated", "sync"):
        for _ in range(cfg.n_steps):
            t_gen, fin, _, busy = run_generation(new_rollouts(cfg.batch_size), cfg.n_slots,
                                                 cfg.tokens_per_sec, None, version)
            t_train = cfg.train_sec_per_token * sum(r.length for r in fin)
            overhead = cfg.switch_sec if policy == "colocated" else cfg.sync_sec
            wall += t_gen + t_train + overhead
            gen_busy += busy
            gen_cap += cfg.n_slots * (t_gen + t_train + overhead)
            train_busy += t_train
            stale += [0] * len(fin)
            samples += len(fin)
            version += 1
        if policy == "colocated":   # trainer and generator are the same hardware
            gen_cap = cfg.n_slots * wall
    elif policy == "async":
        # Generation of batch k+1 runs while batch k trains. With staleness bound s,
        # the generator may run at most s versions ahead of the trainer.
        gen_free_at = 0.0
        train_free_at = 0.0
        batch_ready: List[float] = []
        batch_version: List[int] = []
        train_done_at: List[float] = []
        for k in range(cfg.n_steps):
            # cannot start generating batch k until trainer finished step k - s - 1
            dep = train_done_at[k - cfg.max_staleness - 1] if k - cfg.max_staleness - 1 >= 0 else 0.0
            start = max(gen_free_at, dep)
            # weights available at `start` = number of finished train steps by then
            v = sum(1 for d in train_done_at if d <= start)
            t_gen, fin, _, busy = run_generation(new_rollouts(cfg.batch_size), cfg.n_slots,
                                                 cfg.tokens_per_sec, None, v)
            gen_free_at = start + t_gen + cfg.sync_sec
            gen_busy += busy
            t_train = cfg.train_sec_per_token * sum(r.length for r in fin)
            t0 = max(train_free_at, start + t_gen)
            train_free_at = t0 + t_train
            train_done_at.append(train_free_at)
            train_busy += t_train
            stale += [k - v] * len(fin)
            samples += len(fin)
        wall = max(train_free_at, gen_free_at)
        gen_cap = cfg.n_slots * wall
    elif policy == "partial":
        carry: List[Rollout] = []
        for _ in range(cfg.n_steps):
            target = int(cfg.oversample * cfg.batch_size)
            pending = carry + new_rollouts(max(0, target - len(carry)))
            t_gen, fin, carry, busy = run_generation(pending, cfg.n_slots, cfg.tokens_per_sec,
                                                     cfg.batch_size, version)
            t_train = cfg.train_sec_per_token * sum(r.length for r in fin)
            wall += t_gen + t_train + cfg.sync_sec
            gen_busy += busy
            gen_cap += cfg.n_slots * (t_gen + t_train + cfg.sync_sec)
            train_busy += t_train
            for r in fin:
                stale.append(version - min(r.versions))
                mixed += len(set(r.versions)) > 1
            samples += len(fin)
            version += 1
    else:
        raise ValueError(policy)

    return SimResult(policy, wall, cfg.n_steps, samples, gen_busy, gen_cap, train_busy,
                     stale, mixed / max(1, samples))


def generation_occupancy(lengths: List[int], n_slots: int, tps: float, n_points: int = 50):
    """Fraction of decode slots that are busy over time for one static batch.

    With a single batch and no new admissions, occupancy decays as short responses
    finish -- the curve whose area-above is the long-tail waste.
    """
    finish = sorted(L / tps for L in lengths[:n_slots])
    T = finish[-1]
    pts = []
    for i in range(n_points + 1):
        t = T * i / n_points
        alive = sum(1 for f in finish if f > t)
        pts.append((t, alive / len(finish)))
    return pts


def weight_sync_seconds(n_params: float, bytes_per_param: float, link_GBps: float,
                        n_receivers: int = 1, pipelined_broadcast: bool = True) -> float:
    """Time to push weights from trainer to rollout engines.

    A pipelined (chunked) broadcast costs ~ one transfer of the full payload regardless
    of receiver count; a naive point-to-point fan-out costs ``n_receivers`` transfers
    from the same NIC.
    """
    payload = n_params * bytes_per_param
    transfers = 1 if pipelined_broadcast else n_receivers
    return transfers * payload / (link_GBps * 1e9)
