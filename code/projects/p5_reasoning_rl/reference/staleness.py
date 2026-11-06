"""Rollout versioning and staleness control for asynchronous RL (Chapter 53, after Chapter 46).

A rollout batch carries the *policy version* that generated it. The trainer at version ``v``
may consume a batch of version ``u`` only if ``v - u <= K`` (``K = --async-staleness``);
``K = 0`` is strictly on-policy, ``K = 1`` is the classic one-step-overlapped pipeline. Stale
batches are rejected (counted, never trained on). Because a consumed batch may still be up to
``K`` versions old, the trainer treats the behaviour log-probs recorded at generation time as
``lp_old``, so the clipped ratio pi_theta / pi_behaviour does the off-policy correction.

``AsyncRolloutWorker`` runs generation in a background thread on the rollout device; the
trainer publishes new weights after each optimizer step and the worker picks them up before
its next batch. The queue is bounded by ``K + 1`` so the generator can never run further ahead
than the staleness bound allows.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch


@dataclass
class RolloutBatch:
    version: int                        # policy version that produced the rollouts
    groups: List[List]                  # P groups of G llmsys.rl.rollout.Rollout
    examples: List[Dict]                # the P prompts' records (answers for grading)
    lp_behaviour: Optional[torch.Tensor] = None     # (N, T-1) log-probs under the generator
    meta: Dict = field(default_factory=dict)


class StalenessController:
    """Decides whether a batch is fresh enough and keeps the accept/reject tally."""

    def __init__(self, max_staleness: int):
        if max_staleness < 0:
            raise ValueError("max_staleness must be >= 0")
        self.max_staleness = max_staleness
        self.accepted = 0
        self.rejected = 0
        self.staleness_seen: List[int] = []

    def staleness(self, batch_version: int, trainer_version: int) -> int:
        return trainer_version - batch_version

    def accept(self, batch_version: int, trainer_version: int) -> bool:
        s = self.staleness(batch_version, trainer_version)
        if s < 0:
            raise ValueError("a batch cannot come from a future policy")
        ok = s <= self.max_staleness
        if ok:
            self.accepted += 1
            self.staleness_seen.append(s)
        else:
            self.rejected += 1
        return ok

    def summary(self) -> Dict[str, float]:
        n = max(1, len(self.staleness_seen))
        return dict(accepted=self.accepted, rejected=self.rejected,
                    mean_staleness=sum(self.staleness_seen) / n,
                    max_staleness_seen=max(self.staleness_seen, default=0))


class RolloutQueue:
    """FIFO of rollout batches; ``pop_fresh`` discards stale batches until a fresh one appears."""

    def __init__(self, controller: StalenessController, maxsize: int):
        self.controller = controller
        self.q: "queue.Queue[RolloutBatch]" = queue.Queue(maxsize=maxsize)

    def put(self, batch: RolloutBatch, timeout: Optional[float] = None) -> None:
        self.q.put(batch, timeout=timeout)

    def pop_fresh(self, trainer_version: int, timeout: Optional[float] = None) -> RolloutBatch:
        while True:
            b = self.q.get(timeout=timeout)
            if self.controller.accept(b.version, trainer_version):
                return b


class AsyncRolloutWorker(threading.Thread):
    """Background generator: syncs to the newest published weights, generates, enqueues.

    ``generate_fn(version) -> RolloutBatch`` does the sampling with the worker's own model copy;
    ``sync_fn(state_dict)`` loads published weights into that copy. The trainer calls
    ``publish(version, state_dict)`` after every optimizer step.
    """

    def __init__(self, generate_fn: Callable[[int], RolloutBatch], sync_fn: Callable[[Dict], None],
                 out_queue: RolloutQueue, n_batches: int):
        super().__init__(daemon=True)
        self.generate_fn, self.sync_fn, self.out = generate_fn, sync_fn, out_queue
        self.n_batches = n_batches
        self._lock = threading.Lock()
        self._pending: Optional[tuple] = None        # (version, state_dict) not yet loaded
        self._version = 0
        self._stop = threading.Event()
        self.error: Optional[BaseException] = None

    def publish(self, version: int, state_dict: Dict) -> None:
        with self._lock:
            self._pending = (version, state_dict)

    def stop(self) -> None:
        self._stop.set()

    def _maybe_sync(self) -> None:
        with self._lock:
            pending, self._pending = self._pending, None
        if pending is not None:
            self._version = pending[0]
            self.sync_fn(pending[1])

    def run(self) -> None:
        try:
            produced = 0
            while produced < self.n_batches and not self._stop.is_set():
                self._maybe_sync()
                batch = self.generate_fn(self._version)
                while not self._stop.is_set():
                    try:
                        self.out.put(batch, timeout=0.05)
                        break
                    except queue.Full:
                        continue
                produced += 1
        except BaseException as e:      # surfaced to the trainer thread by ``check``
            self.error = e

    def check(self) -> None:
        if self.error is not None:
            raise RuntimeError("rollout worker failed") from self.error
