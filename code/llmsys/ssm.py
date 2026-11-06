"""A minimal selective state-space model (Mamba-style) for Chapter 11.

The core recurrence for each channel d and state dimension n is

    h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t        (discretised, input-dependent)
    y_t = <C_t, h_t> + D * x_t

where dt_t, B_t, C_t depend on the input ("selective"). Because the update
h_t = a_t * h_{t-1} + b_t is a first-order *linear* recurrence, it can be computed with an
associative (parallel prefix) scan in O(log T) sequential steps, which is what makes SSMs
trainable in parallel on GPUs while still decoding with O(1) state.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- scans
def scan_sequential(a: torch.Tensor, b: torch.Tensor, h0: Optional[torch.Tensor] = None):
    """h_t = a_t * h_{t-1} + b_t along dim 1. a, b: (B, T, ...). Returns all h_t."""
    h = torch.zeros_like(b[:, 0]) if h0 is None else h0
    hs = []
    for t in range(a.shape[1]):
        h = a[:, t] * h + b[:, t]
        hs.append(h)
    return torch.stack(hs, dim=1)


def scan_parallel(a: torch.Tensor, b: torch.Tensor):
    """Hillis-Steele inclusive scan with the associative operator
        (a1, b1) . (a2, b2) = (a1 * a2, a2 * b1 + b2).
    log2(T) rounds, each fully parallel over t (O(T log T) work).
    """
    a, b = a.clone(), b.clone()
    T = a.shape[1]
    step = 1
    while step < T:
        a_prev = torch.ones_like(a)
        b_prev = torch.zeros_like(b)
        a_prev[:, step:] = a[:, :-step]
        b_prev[:, step:] = b[:, :-step]
        # combine element t with element t-step (earlier segment on the left)
        b = a * b_prev + b
        a = a * a_prev
        step *= 2
    return b


def selective_scan(x, dt, A, Bm, Cm, D, parallel: bool = True, h0=None):
    """x, dt: (B,T,E); A: (E,N) negative; Bm, Cm: (B,T,N); D: (E,).
    Returns y (B,T,E) and the last state (B,E,N)."""
    dA = torch.exp(dt[..., None] * A)                    # (B,T,E,N)   decay per step
    dBx = dt[..., None] * Bm[:, :, None, :] * x[..., None]   # (B,T,E,N) input injection
    if parallel and h0 is None:
        h = scan_parallel(dA, dBx)
    else:
        h = scan_sequential(dA, dBx, h0)
    y = torch.einsum("bten,btn->bte", h, Cm) + D * x
    return y, h[:, -1]


# --------------------------------------------------------------------------- block
class MambaBlock(nn.Module):
    """in_proj -> (x, z); x -> causal depthwise conv -> SiLU -> selective SSM; y * SiLU(z) -> out.

    d_state N is the per-channel state size; expand E = expand * d_model channels.
    """

    def __init__(self, d_model: int, d_state: int = 8, expand: int = 2, d_conv: int = 4,
                 dt_rank: Optional[int] = None):
        super().__init__()
        E = expand * d_model
        self.E, self.N, self.k = E, d_state, d_conv
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))
        self.in_proj = nn.Linear(d_model, 2 * E, bias=False)
        self.conv = nn.Conv1d(E, E, d_conv, groups=E, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(E, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, E, bias=True)
        # S4D-real style init: A_n = -(n+1); dt initialised in [1e-3, 1e-1] (log-uniform)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(E, 1))
        dt0 = torch.exp(torch.rand(E) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt0 + torch.log(-torch.expm1(-dt0)))   # softplus^-1
        self.D = nn.Parameter(torch.ones(E))
        self.out_proj = nn.Linear(E, d_model, bias=False)

    def _ssm_params(self, xc):
        dbc = self.x_proj(xc)
        dt, Bm, Cm = torch.split(dbc, [self.dt_rank, self.N, self.N], dim=-1)
        return F.softplus(self.dt_proj(dt)), Bm, Cm

    def forward(self, u, parallel: bool = True):
        B, T, _ = u.shape
        x, z = self.in_proj(u).chunk(2, dim=-1)
        xc = self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)   # causal: drop right pad
        xc = F.silu(xc)
        dt, Bm, Cm = self._ssm_params(xc)
        y, _ = selective_scan(xc, dt, -torch.exp(self.A_log), Bm, Cm, self.D, parallel)
        return self.out_proj(y * F.silu(z))

    def init_state(self, batch: int, device=None):
        return (torch.zeros(batch, self.E, self.k - 1, device=device),   # conv window
                torch.zeros(batch, self.E, self.N, device=device))       # SSM state

    @torch.no_grad()
    def step(self, u_t, state):
        """One decoding step. u_t: (B, 1, d_model). State size is independent of position."""
        conv_s, h = state
        x, z = self.in_proj(u_t).chunk(2, dim=-1)                       # (B,1,E)
        window = torch.cat([conv_s, x.transpose(1, 2)], dim=-1)        # (B,E,k)
        xc = (window * self.conv.weight.squeeze(1)).sum(-1) + self.conv.bias
        xc = F.silu(xc)[:, None]                                        # (B,1,E)
        dt, Bm, Cm = self._ssm_params(xc)
        y, h = selective_scan(xc, dt, -torch.exp(self.A_log), Bm, Cm, self.D, parallel=False, h0=h)
        return self.out_proj(y * F.silu(z)), (window[..., 1:], h)

    def state_numel(self) -> int:
        return self.E * (self.k - 1) + self.E * self.N
