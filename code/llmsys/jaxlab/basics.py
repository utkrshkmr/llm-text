"""JAX basics for PyTorch engineers (Appendix C): a GPT-2 model as pure functions on pytrees.

The model mirrors ``llmsys.gpt.GPTModel`` operation for operation (pre-LN blocks, GPT-2
tanh-GELU MLP, learned positions, tied-or-not output head), so that a PyTorch state_dict can be
converted with :func:`torch_state_dict_to_jax` and the logits compared to ``1e-5``. Nothing here
holds state: parameters are a nested dict of arrays (a *pytree*) that every function receives
as its first argument, and randomness comes from explicit PRNG keys.
"""
from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp

Params = dict[str, Any]

# How PyTorch concepts map onto JAX ones. Printed by the lab's run.py; the appendix has the prose.
MENTAL_MODEL: tuple[tuple[str, str], ...] = (
    ("nn.Module with parameters inside", "pure function + parameter pytree passed in"),
    ("torch.manual_seed / global RNG", "explicit jax.random.key, split per use"),
    ("in-place ops (x.add_(), x[i] = v)", "immutable arrays: x.at[i].set(v) returns a new array"),
    ("loss.backward(); p.grad", "grads = jax.grad(loss_fn)(params, ...)"),
    ("torch.compile (Dynamo bytecode capture)", "jax.jit (tracing to jaxpr, one trace per shape)"),
    ("torch.func.vmap", "jax.vmap"),
    ("for layer in layers: x = layer(x)", "jax.lax.scan over stacked layer params"),
    ("DTensor + DeviceMesh (Shard/Replicate)", "NamedSharding(mesh, PartitionSpec)"),
    ("dist.all_reduce inside your code", "shard_map + jax.lax.psum, or let jit insert it"),
    ("torch.cuda.synchronize()", "jax.block_until_ready(x) (dispatch is asynchronous)"),
)


# --------------------------------------------------------------------------- building blocks
def linear(p: Params, x: jax.Array) -> jax.Array:
    """``x @ W (+ b)`` with W stored as (in, out), the transpose of ``nn.Linear.weight``."""
    y = x @ p["w"]
    return y + p["b"] if "b" in p else y


def layer_norm(p: Params, x: jax.Array, eps: float = 1e-5) -> jax.Array:
    """GPT-2 LayerNorm written out, biased variance like ``llmsys.layers.LayerNorm``."""
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return p["scale"] * (x - mean) / jnp.sqrt(var + eps) + p["shift"]


def gelu(x: jax.Array) -> jax.Array:
    """GPT-2's tanh approximation (identical to ``llmsys.layers.GELU``)."""
    return 0.5 * x * (1.0 + jnp.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def feed_forward(p: Params, x: jax.Array) -> jax.Array:
    return linear(p["fc2"], gelu(linear(p["fc1"], x)))


def causal_attention(p: Params, x: jax.Array, n_heads: int) -> jax.Array:
    """Multi-head causal self-attention on (B, T, D). ``n_heads`` must be a Python int."""
    B, T, D = x.shape
    hd = D // n_heads
    split = lambda t: t.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)   # (B, H, T, hd)
    q, k, v = split(linear(p["q"], x)), split(linear(p["k"], x)), split(linear(p["v"], x))
    scores = q @ k.swapaxes(-1, -2) / math.sqrt(hd)
    allowed = jnp.tril(jnp.ones((T, T), dtype=bool))
    scores = jnp.where(allowed, scores, -jnp.inf)
    weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
    ctx = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return linear(p["out"], ctx)


def transformer_block(p: Params, x: jax.Array, n_heads: int) -> jax.Array:
    """Pre-LN block: x + Attn(LN(x)), then x + FFN(LN(x)). Same maths as ``gpt.TransformerBlock``."""
    x = x + causal_attention(p["att"], layer_norm(p["norm1"], x), n_heads)
    return x + feed_forward(p["ff"], layer_norm(p["norm2"], x))


def gpt_forward(params: Params, idx: jax.Array, n_heads: int) -> jax.Array:
    """Logits (B, T, V) for token ids (B, T). A Python loop over blocks: unrolled when traced."""
    T = idx.shape[-1]
    x = params["tok_emb"][idx] + params["pos_emb"][:T]
    for blk in params["blocks"]:
        x = transformer_block(blk, x, n_heads)
    return layer_norm(params["final_norm"], x) @ params["out_head"]


def stack_blocks(blocks: list[Params]) -> Params:
    """Turn a list of per-layer pytrees into one pytree whose leaves have a leading layer axis."""
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *blocks)


def gpt_forward_scan(params: Params, idx: jax.Array, n_heads: int) -> jax.Array:
    """Same model, but the layer loop is a ``lax.scan``: one traced body, L iterations at run time.

    The compiled program has one copy of the block instead of L; compile time and program size
    stop growing with depth. This is how every production JAX LLM stacks its layers.
    """
    T = idx.shape[-1]
    x = params["tok_emb"][idx] + params["pos_emb"][:T]

    def body(x, blk):
        return transformer_block(blk, x, n_heads), None

    x, _ = jax.lax.scan(body, x, stack_blocks(params["blocks"]))
    return layer_norm(params["final_norm"], x) @ params["out_head"]


# --------------------------------------------------------------------------- initialisation
def _linear_params(key: jax.Array, d_in: int, d_out: int, bias: bool, std: float) -> Params:
    p = {"w": std * jax.random.normal(key, (d_in, d_out), jnp.float32)}
    if bias:
        p["b"] = jnp.zeros((d_out,), jnp.float32)
    return p


def init_block_params(key: jax.Array, d: int, qkv_bias: bool, std: float = 0.02) -> Params:
    kq, kk, kv, ko, k1, k2 = jax.random.split(key, 6)
    norm = lambda: {"scale": jnp.ones((d,), jnp.float32), "shift": jnp.zeros((d,), jnp.float32)}
    return {
        "norm1": norm(),
        "att": {"q": _linear_params(kq, d, d, qkv_bias, std),
                "k": _linear_params(kk, d, d, qkv_bias, std),
                "v": _linear_params(kv, d, d, qkv_bias, std),
                "out": _linear_params(ko, d, d, True, std)},
        "norm2": norm(),
        "ff": {"fc1": _linear_params(k1, d, 4 * d, True, std),
               "fc2": _linear_params(k2, 4 * d, d, True, std)},
    }


def init_gpt_params(key: jax.Array, cfg: dict, std: float = 0.02) -> Params:
    """Random GPT parameters for an ``llmsys.gpt`` config dict. Same key, same parameters."""
    d, L = cfg["emb_dim"], cfg["n_layers"]
    k_tok, k_pos, k_head, *k_blocks = jax.random.split(key, 3 + L)
    return {
        "tok_emb": std * jax.random.normal(k_tok, (cfg["vocab_size"], d), jnp.float32),
        "pos_emb": std * jax.random.normal(k_pos, (cfg["context_length"], d), jnp.float32),
        "blocks": [init_block_params(k, d, cfg["qkv_bias"], std) for k in k_blocks],
        "final_norm": {"scale": jnp.ones((d,), jnp.float32),
                       "shift": jnp.zeros((d,), jnp.float32)},
        "out_head": std * jax.random.normal(k_head, (d, cfg["vocab_size"]), jnp.float32),
    }


def count_params(params: Params) -> int:
    return sum(leaf.size for leaf in jax.tree.leaves(params))


# --------------------------------------------------------------------------- loss and grads
def cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean token cross-entropy, computed from log-softmax in fp32."""
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    picked = jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    return -picked.mean()


def loss_fn(params: Params, idx: jax.Array, targets: jax.Array, n_heads: int) -> jax.Array:
    return cross_entropy(gpt_forward(params, idx, n_heads), targets)


def per_example_grads(params: Params, idx: jax.Array, targets: jax.Array,
                      n_heads: int) -> Params:
    """Gradient of each example's loss: ``vmap`` over the batch of ``grad`` of a 1-example loss.

    Returns a pytree whose leaves have a leading batch axis. The vectorised version runs as one
    batched program (one matmul per layer with an extra batch dimension), not B programs.
    """
    one = lambda p, i, t: loss_fn(p, i[None], t[None], n_heads)   # (T,) -> (1, T)
    return jax.vmap(jax.grad(one), in_axes=(None, 0, 0))(params, idx, targets)


def dropout(key: jax.Array, x: jax.Array, rate: float) -> jax.Array:
    """Explicit-key dropout: the same key always gives the same mask."""
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


# --------------------------------------------------------------------------- PyTorch bridge
def torch_state_dict_to_jax(sd: dict) -> Params:
    """Convert an ``llmsys.gpt.GPTModel`` state_dict into the pytree used by :func:`gpt_forward`.

    Linear weights are transposed to (in, out). Works on any object with ``.detach().cpu()``
    and ``.numpy()`` (i.e. torch tensors); torch itself is not imported here.
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


def max_logit_diff(torch_model, idx) -> float:
    """Largest |logit difference| between a PyTorch GPTModel and its JAX conversion."""
    import numpy as np
    import torch

    torch_model.eval()
    with torch.no_grad():
        ref = torch_model(torch.as_tensor(np.asarray(idx))).float().numpy()
    params = torch_state_dict_to_jax(torch_model.state_dict())
    ours = gpt_forward(params, jnp.asarray(np.asarray(idx)), torch_model.cfg["n_heads"])
    return float(np.max(np.abs(np.asarray(ours) - ref)))
