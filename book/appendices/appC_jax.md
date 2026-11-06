# JAX and XLA for PyTorch Engineers {#app:jax}

Every system in this book is written in PyTorch, and for good reasons: it is the language of the
open-model ecosystem, of the serving engines of [@ch:serving], and of the custom kernels of
[@ch:cuda]. But a large share of the frontier training runs we have reasoned about, and
almost everything that runs on TPUs ([@app:accel]), is written in JAX (Bradbury et al., 2018)
and compiled by XLA. The two frameworks arrive at the same collectives, the same rooflines and the
same sharded-matmul cases of [@ch:shardcalc] from opposite directions. PyTorch gives you an
imperative program and lets you add parallelism by hand or through DTensor; JAX gives you a pure
function and a compiler that partitions it for you, with an escape hatch when you want to write the
per-device program yourself. Reading JAX fluently is therefore not a change of allegiance but a
second view of the same machine, and the view in which the compiler's decisions are the most
visible.

This appendix is written for a reader who knows PyTorch well and JAX not at all. We build the mental
model first: pure functions, parameter pytrees, explicit random keys, and tracing. We then rewrite
the tiny GPT of [@ch:gpt] as JAX functions and show that it reproduces the PyTorch model's
logits on the same weights to $6\cdot10^{-7}$. The second half is about sharding: the device mesh,
named shardings, the three levels of control (let `jit` decide, pin an intermediate, or write the
SPMD program in `shard_map`), the four matmul cases and the collective matmul of
[@ch:shardcalc] written in JAX, and finally how to read the StableHLO and optimised HLO that
XLA produces, so that you can see which collective it chose and why. We close with a profile and
with an honest comparison of when each framework is the better tool.

Nothing here needs an accelerator. XLA's CPU backend can be asked to pretend it has eight devices
(`XLA_FLAGS=--xla_force_host_platform_device_count=8`), and every sharded program in this appendix
and in Lab C runs on that simulated mesh, producing the same HLO, the same collectives and the same
shardings a real eight-chip host would. The reference code is in `llmsys/jaxlab/`.

![Roadmap of this appendix. Stages 1--3 build the programming model and the compiler picture; stages
4--8 redo the sharded computation of [@ch:shardcalc] in JAX and read the compiler's
output.](figures/appC_jax-1.svg){#fig:jax:roadmap}

## Why a PyTorch engineer should read JAX {#sec:jax:why}

Three things are easier to see in JAX than in PyTorch, and they are exactly the things this book
cares about. First, the *program* is a first-class object: `jax.jit` traces a Python function into
an intermediate representation that you can print, lower to StableHLO, compile, and read back as
optimised HLO. In PyTorch the equivalent artefacts (an `fx` graph, Inductor's generated Triton)
exist but are a layer removed from the code you wrote. Second, *sharding is a property of arrays*,
not of modules: an array carries a `NamedSharding` that says which mesh axis each of its dimensions
is split over, and the compiler's SPMD partitioner derives every collective from those annotations.
This is [@ch:shardcalc]'s calculus implemented as a compiler pass, and we can check the two
against each other. Third, when the compiler's choice is not the one we want, `shard_map` lets us
write the per-device program with explicit `psum`, `all_gather` and `ppermute`, while keeping `jit`,
`grad` and `vmap` working across it. Writing the four matmul cases once at each level is the fastest
way we know to understand what GSPMD (Xu et al., 2021) does on your behalf.

::: keyidea
[The same machine, two directions]{.box-title}

PyTorch starts from an imperative program and adds parallelism (DDP wrappers, FSDP, DTensor
placements, hand-written collectives). JAX starts from a pure function and a declaration of where
its inputs and outputs live, and a compiler fills in the collectives. Both end in the same HLO-level
program: matmuls on local shards joined by all-gathers, reduce-scatters and all-reduces whose bytes
are fixed by the shardings alone. Learning to read that program is the skill; the framework is the
notation.
:::

## The functional model {#sec:jax:functional}

### Pure functions and parameter pytrees

An `nn.Module` bundles parameters, buffers and a `forward` method; calling it mutates nothing
visible but the module *owns* state, and that state is what `load_state_dict`, `.to(device)` and the
optimizer operate on. A JAX model is a pure function of `(params, inputs)` where `params` is a
*pytree*: any nesting of Python dicts, lists and tuples whose leaves are arrays. The function has no
attributes and no device; the parameters are just a value that the caller passes in and gets new
versions of back. [@lst:jax:block] is the transformer block of [@ch:gpt] in this
style; compare it with `TransformerBlock` in `llmsys/gpt.py`.

``` {#lst:jax:block .python caption="A pre-LN transformer block as pure functions on a parameter pytree (`llmsys/jaxlab/basics.py`). Linear weights are stored $(d_\\text{in}, d_\\text{out})$, the transpose of `nn.Linear.weight`."}
def linear(p, x):                      # p = {"w": (in, out), "b": (out,)} or no "b"
    y = x @ p["w"]
    return y + p["b"] if "b" in p else y

def layer_norm(p, x, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return p["scale"] * (x - mean) / jnp.sqrt(var + eps) + p["shift"]

def causal_attention(p, x, n_heads):   # n_heads is a Python int, not a traced value
    B, T, D = x.shape
    hd = D // n_heads
    split = lambda t: t.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
    q, k, v = split(linear(p["q"], x)), split(linear(p["k"], x)), split(linear(p["v"], x))
    scores = q @ k.swapaxes(-1, -2) / math.sqrt(hd)
    scores = jnp.where(jnp.tril(jnp.ones((T, T), dtype=bool)), scores, -jnp.inf)
    weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(x.dtype)
    ctx = (weights @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return linear(p["out"], ctx)

def transformer_block(p, x, n_heads):
    x = x + causal_attention(p["att"], layer_norm(p["norm1"], x), n_heads)
    return x + feed_forward(p["ff"], layer_norm(p["norm2"], x))

def gpt_forward(params, idx, n_heads):
    x = params["tok_emb"][idx] + params["pos_emb"][: idx.shape[-1]]
    for blk in params["blocks"]:       # a Python loop: unrolled when traced
        x = transformer_block(blk, x, n_heads)
    return layer_norm(params["final_norm"], x) @ params["out_head"]
```

The parameter pytree for this model is a dict with keys `tok_emb`, `pos_emb`, `blocks` (a list, one
dict per layer), `final_norm` and `out_head` ([@fig:jax:pytree]). Pytrees are what every
JAX transformation operates on: `jax.grad` returns a pytree of gradients with the same structure as
the parameters, `jax.tree.map(lambda p, g: p - lr * g, params, grads)` is an SGD step, and
`jax.tree.leaves(params)` flattens the whole model for counting or checkpointing. The same structure
is what a PyTorch `state_dict` is, minus the module that owns it, and [@sec:jax:bridge]
converts one into the other mechanically.

![A module owns its parameters; a JAX model receives them. The pytree has the same shape as a
`state_dict`, which is why converting between the two is
mechanical.](figures/appC_jax-2.svg){#fig:jax:pytree}

### Immutability

JAX arrays cannot be modified in place. `x[i] = v` is an error; `x.at[i].set(v)` returns a new array
with the update applied. Under `jit` this costs nothing, because XLA owns the buffers and turns the
functional update into an in-place one when the old value is dead (*buffer donation* lets you
promise this across a `jit` boundary too, with `donate_argnums`). What you lose is the PyTorch idiom
of preallocating a KV cache and writing into it from the outside; what you gain is that every
function is safe to retrace, reorder, differentiate and partition, because no operation has a hidden
effect on anything else. The compiler's freedom to fuse, schedule and shard rests entirely on this
guarantee.

### Explicit randomness

PyTorch has a global generator: `torch.manual_seed(0)` and every subsequent `torch.randn` advances
hidden state. A pure function cannot do that, so JAX threads randomness through explicitly. A *key*
is a value, `jax.random.key(0)`; `jax.random.split(key, n)` derives $n$ independent keys; each
sampling call consumes a key and the same key always gives the same numbers
([@lst:jax:rng]). The discipline looks pedantic until you shard a model: with explicit
keys, dropout on device 3 uses a key that is a pure function of (step, layer, device), so a run is
reproducible across any number of devices and any re-partitioning, which is not true of a
per-process global generator.

``` {#lst:jax:rng .python caption="Explicit PRNG keys: dropout as a pure function of its key."}
def dropout(key, x, rate):
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)

key = jax.random.key(0)
k_init, k_drop = jax.random.split(key)          # never reuse a key for two purposes
params = init_gpt_params(k_init, cfg)
h = dropout(k_drop, h, 0.1)                     # same k_drop -> same mask, on any device
```

### A translation table

[@tab:jax:mental] collects the correspondences we use in the rest of the appendix. The
right column is deliberately short: most of JAX's API is NumPy's (`jax.numpy` is a drop-in for the
subset you need), and the framework-specific part is a handful of transformations and the sharding
types.

::: {#tab:jax:mental .table-float}
| PyTorch | JAX |
|:---|:---|
| `nn.Module` holding parameters | pure function + parameter pytree passed in |
| `torch.manual_seed`, global RNG | explicit `jax.random.key`, split per use |
| in-place ops (`x.add_()`, `x[i] = v`) | immutable arrays: `x.at[i].set(v)` returns a new array |
| `loss.backward(); p.grad` | `grads = jax.grad(loss_fn)(params, ...)` |
| `torch.compile` (Dynamo bytecode capture, guards) | `jax.jit` (tracing to jaxpr, one trace per shape signature) |
| `torch.func.vmap` | `jax.vmap` |
| `for layer in layers: x = layer(x)` | `jax.lax.scan` over stacked layer parameters |
| `DeviceMesh` + DTensor `Shard(i)`/`Replicate()` | `Mesh` + `NamedSharding(mesh, PartitionSpec(...))` |
| `dist.all_reduce` written by you | `shard_map` + `jax.lax.psum`, or let `jit` insert it |
| `torch.cuda.synchronize()` | `jax.block_until_ready(x)` (dispatch is asynchronous) |
| `torch.fx` graph / Inductor Triton | jaxpr / StableHLO / optimised HLO ([@sec:jax:xla]) |

: The PyTorch-to-JAX mental model used in this appendix (`basics.MENTAL_MODEL` prints it).
:::

::: systemslens
[Purity is what the compiler is paid in]{.box-title}

Every optimisation XLA performs---fusing a chain of elementwise ops into one kernel, choosing memory
layouts, deciding which device computes which block of a matmul, overlapping a collective with the
matmul that follows it---requires knowing that reordering two operations cannot change the result.
In an imperative framework the compiler has to *prove* this (Dynamo guards, aliasing analysis, graph
breaks when it cannot); in JAX the programming model guarantees it. That is the trade: you give up
in-place updates and global state, and in return the whole program, including its parallelisation,
becomes the compiler's to rearrange.
:::

## Transformations: jit, grad, vmap, scan {#sec:jax:transforms}

### jit: tracing, not capturing

`jax.jit(f)` returns a function that, on its first call with arrays of a given shape and dtype, runs
`f` once with *tracers* in place of the arrays. Every `jnp` operation on a tracer records itself
into a *jaxpr*, a small typed IR ([@lst:jax:jaxpr]). Python control flow executes during
tracing: an `if` on a Python value picks a branch at trace time, a `for` loop over a Python range is
unrolled, and a Python `int` like `n_heads` is a compile-time constant. Anything that depends on the
*value* of a traced array (`if x.sum() > 0`) is an error, because the tracer has a shape but no
value; `jax.lax.cond` and `jax.lax.while_loop` are the traced alternatives. The jaxpr is then
lowered to StableHLO, compiled by XLA, and cached under the *signature*: the pytree structure,
shapes and dtypes of the arguments plus any `static_argnums`. A later call with the same signature
skips Python entirely and dispatches the compiled executable.

``` {#lst:jax:jaxpr caption="The jaxpr of `jax.nn.gelu(x @ w)` for $x\\in\\mathbb{R}^{8\\times16}$, $w\\in\\mathbb{R}^{16\\times8}$ (`jax.make_jaxpr`). Each line is one primitive with typed inputs and outputs."}
{ lambda ; a:f32[8,16] b:f32[16,8]. let
    c:f32[8,8] = dot_general[dimension_numbers=(([1], [0]), ([], []))] a b
    d:f32[8,8] = integer_pow[y=3] c
    e:f32[8,8] = mul 0.044714998453855515:f32[] d
    f:f32[8,8] = add c e
    g:f32[8,8] = mul 0.7978845834732056:f32[] f
    h:f32[8,8] = tanh g
    i:f32[8,8] = add 1.0:f32[] h
    j:f32[8,8] = mul 0.5:f32[] i
    k:f32[8,8] = mul c j
  in (k,) }
```

This is a different mechanism from `torch.compile`. Dynamo captures a graph by symbolically
executing Python *bytecode*, installing guards on the values it specialised on, and falling back to
eager Python at a graph break; it can handle data-dependent control flow by splitting the graph
around it, and with dynamic shapes it compiles a symbolic program once for many sizes. JAX's tracing
is simpler and stricter: there are no graph breaks (a function either traces or raises), and shapes
are always static, so a new shape means a new trace and a new compile ([@fig:jax:tracing]).
[@lst:jax:traces] makes the cache visible: the Python body runs once per new shape and
never again for a shape it has seen.

![What a call to a jitted function does. The cost of the "no" branch is the shape-specialisation
cost of [@sec:jax:xla]; the "yes" branch is a hash lookup and an asynchronous
dispatch.](figures/appC_jax-3.svg){#fig:jax:tracing}

``` {#lst:jax:traces .python caption="Counting traces: the Python body runs only on a cache miss (`basics.count_traces`)."}
def count_traces(fn, inputs):
    calls = [0]
    def counted(*args):
        calls[0] += 1                       # Python runs only while tracing
        return fn(*args)
    jitted = jax.jit(counted)
    history = []
    for args in inputs:
        jax.block_until_ready(jitted(*args))
        history.append(calls[0])
    return history

count_traces(lambda x: x * 2, [(jnp.ones(3),), (jnp.ones(3),), (jnp.ones(4),)])
# -> [1, 1, 2]: same shape reuses the executable, a new shape retraces and recompiles
```

::: pitfall
[Python values versus traced values]{.box-title}

Inside a jitted function, `x.shape` is a tuple of Python ints and may drive control flow; `x[0]` is
a tracer and may not. Passing a model hyper-parameter such as `n_heads` as an ordinary argument
makes it a traced `int32` scalar, and `D // n_heads` then fails to produce a static shape for the
reshape. Mark it `static_argnums` (or close over it), and remember that every distinct static value
is a separate compile. The same rule applies to `jax.lax.scan` lengths, mesh shapes and
`PartitionSpec`s: all are compile-time constants.
:::

### grad and value_and_grad

`jax.grad(loss_fn)` is a function that returns $\partial\text{loss}/\partial(\text{first
argument})$ as a pytree shaped like that argument. There is no `.backward()`, no `.grad` attribute
and no `zero_grad()`: gradients are return values. JAX implements this by *tracing* the function to
a jaxpr, applying a forward-mode rule (JVP) to every primitive and then *transposing* the resulting
linear program to obtain reverse mode (Frostig et al., 2018). The transposition step is the one to
remember: it is why, in [@sec:jax:shardmap], the gradient of an `all_gather` comes out as a
`psum_scatter` and the gradient of a `ppermute` as the reverse permutation, with no code of ours.

``` {#lst:jax:grad .python caption="Loss and gradient for the JAX GPT. `value_and_grad` returns both in one traced program."}
def cross_entropy(logits, targets):
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    return -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0].mean()

def loss_fn(params, idx, targets, n_heads):
    return cross_entropy(gpt_forward(params, idx, n_heads), targets)

step = jax.jit(jax.value_and_grad(loss_fn), static_argnums=3)
loss, grads = step(params, idx, targets, 4)          # grads: same pytree as params
params = jax.tree.map(lambda p, g: p - 1e-3 * g, params, grads)   # SGD, functionally
```

### vmap: batching as a transformation

`jax.vmap(f, in_axes)` turns a function of one example into a function of a batch by adding a batch
dimension to every primitive in its jaxpr; the result is a single batched program, not a loop.
Combined with `grad` it gives per-example gradients ([@lst:jax:vmap]), the primitive behind
gradient clipping per sample, differential privacy, and the influence computations of
[@ch:data]; in PyTorch the same thing is `torch.func.vmap(torch.func.grad(...))` and is a
more recent addition built on the same idea. `in_axes=(None, 0, 0)` says: do not map over the
parameters, map the two token arrays along their first axis.

``` {#lst:jax:vmap .python caption="Per-example gradients: `vmap` of `grad` of a one-example loss (`basics.per_example_grads`)."}
def per_example_grads(params, idx, targets, n_heads):
    one = lambda p, i, t: loss_fn(p, i[None], t[None], n_heads)    # (T,) -> (1, T)
    return jax.vmap(jax.grad(one), in_axes=(None, 0, 0))(params, idx, targets)
# every leaf of the result has a leading axis of size B; one batched program runs
```

### scan: a loop the compiler can see {#sec:jax:scan}

The Python loop in [@lst:jax:block] is unrolled during tracing: a 32-layer model becomes 32
copies of the block in the jaxpr, in the StableHLO and in the compiled program, and compile time
grows with depth. `jax.lax.scan` traces the body *once* and emits a loop whose carried state is the
activation and whose per-iteration input is layer $\ell$'s slice of a *stacked* parameter pytree
([@lst:jax:scan]). Production JAX LLMs store their layers stacked for exactly this reason,
and it is why a `state_dict` converted from PyTorch needs a `stack_blocks` step before it fits such
a model.

``` {#lst:jax:scan .python caption="The layer loop as `lax.scan`: one traced body, stacked parameters (`basics.gpt_forward_scan`)."}
def stack_blocks(blocks):            # list of per-layer pytrees -> one pytree with a layer axis
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *blocks)

def gpt_forward_scan(params, idx, n_heads):
    x = params["tok_emb"][idx] + params["pos_emb"][: idx.shape[-1]]
    def body(x, blk):                # blk is layer l's slice of the stacked pytree
        return transformer_block(blk, x, n_heads), None
    x, _ = jax.lax.scan(body, x, params["blocks"])      # params["blocks"] is stacked
    return layer_norm(params["final_norm"], x) @ params["out_head"]
```

::: napkinmath
[Program size with and without scan]{.box-title}

For the tiny GPT (`GPT_CONFIG_TINY`, $d_{\text{model}}=64$) on the CPU backend we measured the
lowered program for 2, 8 and 16 layers. Unrolled: 317, 1067 and 2067 lines of StableHLO and 1086,
3480 and 6672 lines of optimised HLO, growing by about 125 StableHLO and 400 HLO lines per layer.
With `scan`: 253 and 865 lines at every depth. Compile time followed the same pattern (0.31, 0.38,
0.59 s unrolled versus a flat 0.73 s with scan: the loop costs a little fixed overhead and the
unrolled program only overtakes it beyond about 20 layers at this size). At production width the
unrolled program is tens of thousands of HLO instructions and compiles for minutes; scan keeps it at
one layer's worth. The price is that the compiler cannot specialise layer $\ell$ differently from
layer $\ell'$, which matters for the first and last layers of pipelines and for models with
heterogeneous layers (the hybrids of [@ch:attnalt]): those are written as a scan over the
homogeneous middle plus explicit first and last blocks.
:::

### The PyTorch bridge {#sec:jax:bridge}

To be sure that [@lst:jax:block] is the *same* model as `llmsys/gpt.py` and not merely a
similar one, we convert the PyTorch model's `state_dict` into the pytree and compare logits
([@lst:jax:bridge]). The conversion is a renaming plus one transpose per linear layer; the
tensors cross over as NumPy arrays, so `torch` is never imported on the JAX side. On the tiny
configuration the largest logit difference is $6\cdot10^{-7}$ in fp32, which is the accumulated
rounding difference of two libraries' matmul orderings and nothing else.

``` {#lst:jax:bridge .python caption="Converting an `llmsys.gpt.GPTModel` state_dict to the pytree of [@lst:jax:block] (`basics.torch_state_dict_to_jax`, abridged)."}
def torch_state_dict_to_jax(sd):
    arr = lambda name: jnp.asarray(sd[name].detach().cpu().float().numpy())
    def lin(prefix):                              # nn.Linear.weight is (out, in)
        p = {"w": arr(prefix + ".weight").T}
        if prefix + ".bias" in sd:
            p["b"] = arr(prefix + ".bias")
        return p
    norm = lambda prefix: {"scale": arr(prefix + ".scale"), "shift": arr(prefix + ".shift")}
    blocks = [{"norm1": norm(f"trf_blocks.{i}.norm1"),
               "att": {"q": lin(f"trf_blocks.{i}.att.W_query"), ...},
               "norm2": norm(f"trf_blocks.{i}.norm2"),
               "ff": {"fc1": lin(f"trf_blocks.{i}.ff.layers.0"),
                      "fc2": lin(f"trf_blocks.{i}.ff.layers.2")}} for i in range(n_layers)]
    return {"tok_emb": arr("tok_emb.weight"), "pos_emb": arr("pos_emb.weight"),
            "blocks": blocks, "final_norm": norm("final_norm"),
            "out_head": arr("out_head.weight").T}

params = torch_state_dict_to_jax(model.state_dict())
jax_logits = gpt_forward(params, jnp.asarray(idx), model.cfg["n_heads"])
# max |jax_logits - model(idx)| = 6e-7 on GPT_CONFIG_TINY
```

## XLA: from jaxpr to machine code {#sec:jax:xla}

### The pipeline

XLA (Accelerated Linear Algebra) is a domain-specific compiler whose input is HLO (High-Level
Operations), a small graph language of about a hundred ops: elementwise arithmetic, `dot`, `reduce`,
`broadcast`, `reshape`, `dynamic-slice`, `while`, `conditional`, the collectives, and a few more.
JAX lowers a jaxpr to *StableHLO*, the versioned MLIR dialect that is XLA's public input format, and
hands it to XLA, which runs a long sequence of passes ([@fig:jax:pipeline]) and emits an
executable for the target: LLVM IR for CPU, PTX for NVIDIA GPUs, and the proprietary backend for
TPUs. Two of those passes are the subject of the rest of this appendix: the *SPMD partitioner*,
which turns one program with sharding annotations into the per-device program with collectives, and
*fusion*. Two more decide most of the single-device performance: *layout assignment* and *scheduling
with buffer assignment*.

![From a Python function to an executable. The two texts a JAX user can read are StableHLO (what JAX
asked for) and optimised HLO (what XLA decided to run); the lab dumps both for every sharded matmul
case.](figures/appC_jax-4.svg){#fig:jax:pipeline}

### Reading StableHLO and HLO

[@lst:jax:stablehlo] is the StableHLO of `jax.nn.gelu(x @ w)`: one op per jaxpr primitive,
in MLIR syntax, with scalar constants broadcast explicitly. [@lst:jax:hlo] is the optimised
HLO of the same program. Three things happened. The ten elementwise ops became *one* loop fusion
(`kind=kLoop`): a single kernel that reads the matmul output once and writes the GELU once, which is
exactly the fusion we wrote by hand in Triton in [@app:triton]. The `dot` became a `kCustom`
fusion that calls the backend's matmul library (on CPU, XLA's own tiled kernels; on GPU, cuBLAS or a
Triton GEMM). And every array acquired a *layout*: `f32[8,8]{1,0}` means an $8\times8$ array whose
dimension 1 is minor-most (row-major). Layout assignment chooses these per instruction so that the
matmul library gets the layout it wants and inserts transposes only where it must; on TPU this is
where the $(8,128)$ tiling of [@app:accel] enters.

``` {#lst:jax:stablehlo caption="StableHLO for `gelu(x @ w)` (`hlo.stablehlo_text`, abridged). Every jaxpr primitive is one op; nothing has been fused yet."}
func.func public @main(%arg0: tensor<8x16xf32>, %arg1: tensor<16x8xf32>) -> tensor<8x8xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0]
  %1 = stablehlo.multiply %0, %0 : tensor<8x8xf32>
  %2 = stablehlo.multiply %1, %0 : tensor<8x8xf32>
  %cst = stablehlo.constant dense<4.471500e-02> : tensor<f32>
  %3 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<8x8xf32>
  %4 = stablehlo.multiply %3, %2 : tensor<8x8xf32>
  %5 = stablehlo.add %0, %4 : tensor<8x8xf32>
  ...                                  # tanh, 1 + ., 0.5 * ., x * .
  return %13 : tensor<8x8xf32>
}
```

``` {#lst:jax:hlo caption="Optimised HLO for the same program (`hlo.compiled_text`, metadata stripped). Two fusions: the matmul and one loop fusion for the whole GELU."}
  %param_0.1 = f32[8,8]{1,0} parameter(0)
  %integer_pow.1 = f32[8,8]{1,0} multiply(%param_0.1, %param_0.1)
  %integer_pow.0 = f32[8,8]{1,0} multiply(%integer_pow.1, %param_0.1)
  ...                                  # the remaining elementwise ops
  ROOT %mul.0 = f32[8,8]{1,0} multiply(%param_0.1, %mul.1)
}

ENTRY %main.1 (x.1: f32[8,16], w.1: f32[16,8]) -> f32[8,8] {
  %x.1 = f32[8,16]{1,0} parameter(0)
  %w.1 = f32[16,8]{1,0} parameter(1)
  %ynn_fusion = f32[8,8]{1,0} fusion(%x.1, %w.1), kind=kCustom, calls=%fused_computation
  ROOT %multiply_multiply_fusion = f32[8,8]{1,0} fusion(%ynn_fusion), kind=kLoop,
        calls=%fused_computation.1
}
```

The HLO grammar is regular enough to parse with a regular expression, which is what
`hlo.count_fusions` and `hlo.count_collectives` do: every instruction is
`%name = type op(operands), attributes`, and the `ENTRY` computation is the program's top level.
Fusions come in four kinds. `kLoop` is a fused elementwise loop; `kInput` is a reduction (softmax's
row max and sum, a norm's mean of squares) with its producers fused in; `kOutput` fuses an
elementwise epilogue into a matmul or convolution (the bias-and-activation epilogue of
[@app:triton]); `kCustom` calls a library or a hand-written kernel. Reading the fusion list
of a transformer layer tells you in seconds whether the norm, the activation and the residual add
became one kernel each or one kernel together.

### The shape-specialisation trade-off

XLA compiles a program for *exactly* the shapes it was traced with. This is the source of both its
strength and its most-cited weakness. With every shape known, every buffer's size is known, so XLA
can assign all temporaries to a single preplanned arena (buffer assignment), schedule instructions
to minimise peak memory, decide fusion profitability exactly, and emit collectives whose byte counts
are compile-time constants. The price is one compile per distinct shape, and compiles of large
programs take minutes.

::: napkinmath
[How many shapes can you afford?]{.box-title}

Suppose a training-step program compiles in 90  and a step runs in 2 . A single shape amortises its
compile over the first 45 steps of a run that has $10^5$ of them: negligible. Now consider serving
with variable prompt lengths. Compiling one program per length from 1 to 4096 is
$4096\times90\,\mathrm{\second}\approx100\,\mathrm{\hour}$, which is absurd; compiling for 13
power-of-two *buckets* (1, 2, 4, ..., 4096) and padding each prompt up to the next bucket is 20  of
compile and at most $2\times$ wasted prefill compute, $1.4\times$ on average. This is exactly the
bucketing that TPU serving systems do, and it is why the paged KV cache of [@ch:serving],
whose block table has a data-dependent shape, needs a fixed maximum number of blocks per sequence in
a JAX implementation. The compile cache can be persisted to disk across processes
(`jax.config.update("jax_compilation_cache_dir", ...)`), which turns the 20  into a one-off.
:::

::: pitfall
[Dynamic shapes are a redesign, not a flag]{.box-title}

A PyTorch program that grows a KV cache by concatenation, drops finished sequences from a batch, or
routes a variable number of tokens to each expert will retrace and recompile on every step under
`jit`. The fixes are the ones [@ch:inference] and [@ch:moe] already motivated for
performance reasons: preallocate the cache at maximum length and write into it with
`dynamic_update_slice`; keep the batch at a fixed size with masked padding slots; give every expert
a fixed capacity and drop or pad. XLA simply refuses to let you postpone these decisions.
:::

::: systemslens
[Static shapes buy a memory plan]{.box-title}

The reason a JAX training run reports a single peak-memory number at compile time, and a PyTorch run
has to be probed with `max_memory_allocated`, is buffer assignment: with every shape fixed, XLA lays
out all temporaries of the step in one arena and knows the high-water mark before the first step
executes. `compiled.memory_analysis()` returns it (`hlo.cost_summary` prints it alongside XLA's FLOP
and byte estimates). The same property is what lets XLA schedule collectives to overlap with compute
without a runtime allocator getting in the way, which we return to in [@sec:jax:cmm].
:::
