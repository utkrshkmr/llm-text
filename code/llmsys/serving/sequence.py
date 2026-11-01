"""Request/sequence bookkeeping for the serving engine."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class SamplingParams:
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    eos_id: Optional[int] = None
    seed: Optional[int] = None


class Status(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Sequence:
    rid: int
    prompt: List[int]
    params: SamplingParams
    arrival: float = 0.0
    output: List[int] = field(default_factory=list)
    status: Status = Status.WAITING
    block_table: List[int] = field(default_factory=list)
    block_hashes: List[int] = field(default_factory=list)  # hashes of registered full blocks
    num_computed: int = 0          # tokens whose K/V are already in the paged cache
    prefix_hit_tokens: int = 0
    num_preemptions: int = 0
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    token_times: List[float] = field(default_factory=list)
    generator: Optional[torch.Generator] = None

    @property
    def tokens(self) -> List[int]:
        return self.prompt + self.output

    @property
    def num_tokens(self) -> int:
        return len(self.prompt) + len(self.output)

    @property
    def remaining(self) -> int:
        """Tokens that still have to be pushed through the model before we can sample."""
        return self.num_tokens - self.num_computed

    def is_finished(self) -> bool:
        if not self.output:
            return False
        p = self.params
        return len(self.output) >= p.max_new_tokens or (p.eos_id is not None and self.output[-1] == p.eos_id)
