# Modern Architectures: Qwen3 from Scratch {#ch:modern}

GPT-2 was released in 2019. Six years later the dominant open-weight models---Llama, Mistral, Gemma,
DeepSeek, Qwen---are still decoder-only transformers with the same overall shape as the model of
[@ch:gpt]: embed, stack $L$ residual blocks of attention plus an MLP, normalize, project to
the vocabulary. What changed are the parts inside the block, and a striking number of those changes
were driven not by modeling accuracy but by *systems* concerns: cheaper normalization, attention
whose KV cache is several times smaller, position encodings that need no learned table and
extrapolate to longer contexts, and sparse MLPs that grow the parameter count without growing the
FLOPs per token.

This chapter rebuilds one representative modern model, Qwen3 (Qwen Team, 2025), component by
component, and explains each design change from both angles: what it does to the model and what it
does to the machine. Qwen3 is the base model for the whole second half of this book---inference
([@ch:inference]), serving ([@ch:serving]), evaluation ([@ch:eval]),
reinforcement learning ([@ch:grpo]) and distillation ([@ch:distill])---so the code we
write here is the code we will run for the next fourteen chapters. We finish with the larger members
of the dense Qwen3 family, how to load and plan memory for them, and with mixture-of-experts layers,
the main architectural lever for scaling past what a dense model can afford.

![Roadmap of this chapter: we replace GPT-2's components one at a time (1--5), assemble and load the
real Qwen3-0.6B (6--8), then scale up to larger dense models and to sparse mixture-of-experts layers
(9--10).](figures/ch06_modern_arch-1.svg){#fig:modern:roadmap}

## From GPT-2 to Qwen3 {#sec:modern:diff}

[@fig:modern:sidebyside] places the two architectures next to each other. The skeleton is
unchanged: token embedding, $L$ pre-norm residual blocks, final norm, linear head. The differences
are all local and are summarized in [@tab:modern:diff].

![GPT-2 small and Qwen3-0.6B side by side. Both are pre-norm decoder-only transformers; Qwen3 swaps
every normalization for RMSNorm, the GELU MLP for a gated SwiGLU MLP, learned absolute positions for
rotary embeddings applied inside attention, and multi-head for grouped-query attention with per-head
query/key normalization. Dropout is gone.](figures/ch06_modern_arch-2.svg){#fig:modern:sidebyside}

::: {#tab:modern:diff .table-float}
| Component | GPT-2 | Qwen3 (dense) |
|:---|:---|:---|
| Normalization | LayerNorm (mean, variance, scale, shift) | RMSNorm (RMS, scale only) |
| MLP | $W_2\,\operatorname{GELU}(W_1x)$, biases, $d_{\text{ff}}=4d$ | $W_3(\operatorname{SiLU}(W_1x)\odot W_2x)$, no biases |
| Positions | learned absolute table added to input | rotary, applied to $q,k$ in every layer |
| Attention | MHA, $H$ K/V heads | GQA, 8 K/V heads shared by $H$ query heads |
| Extra norms | -- | RMSNorm on each query and key head |
| Biases | everywhere | none |
| Dropout | 0.1 | none (pretraining on enough data to not overfit) |
| Vocabulary | 50257 | 151936 (padded) |
| Context | 1024 | 32768 native, 40960 positions configured |
| Weights dtype | fp32 checkpoint | bf16 checkpoint |

: What changed between GPT-2 and Qwen3.
:::

Notice the absence of dropout. GPT-2 was trained for a few epochs over a 40 GB corpus; modern models
train for a single pass over tens of trillions of tokens, so they barely overfit and dropout would
only add noise and a random-mask kernel per sublayer. Removing biases simplifies tensor parallelism
(a bias on a row-parallel layer must be added exactly once after the all-reduce,
[@ch:distributed]) and has no measurable cost in quality.

## RMSNorm {#sec:modern:rmsnorm}

Root-mean-square normalization (Zhang and Sennrich, 2019) keeps only the rescaling half of layer
norm. It divides each token vector by its root mean square and applies a learned per-feature gain,
without subtracting the mean and without a shift: $$\begin{equation}
\mathrm{RMSNorm}(\boldsymbol{x}) = \boldsymbol{\gamma}\odot\frac{\boldsymbol{x}}{\sqrt{\tfrac1d\sum_i x_i^2+\epsilon}} .
\label{eq:modern:rms}
\end{equation}$$ The output is not centered: its mean is whatever the input's mean was, scaled.
Empirically this does not hurt---re-centering turns out to be unnecessary for the optimization
benefits of normalization, which come mainly from controlling the scale.

![The same four-feature token normalized by LayerNorm and by RMSNorm ($\gamma=1$). Both put the
vector on a fixed scale; only LayerNorm also removes the
mean.](figures/ch06_modern_arch-3.svg){#fig:modern:rmsvsln}

``` {#lst:modern:rms .python caption="RMSNorm with an fp32 accumulation (cf. `llmsys/layers.py`)."}
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))   # gamma; no shift

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)                    # square and mean in fp32
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.scale.to(torch.float32)).to(in_dtype)
```

The explicit up-cast matters. The model runs in bf16, whose 8-bit mantissa gives only about three
significant decimal digits; summing $d_{\text{model}}=1024$ squares in bf16 accumulates rounding
error, and squaring large activations can overflow fp16. Doing the reduction in fp32 and casting
back is what every production kernel does, and it is required to reproduce the official logits.

::: systemslens
[Why RMSNorm is faster]{.box-title}

LayerNorm needs two row statistics, the mean and the variance; RMSNorm needs one, $\sum_i x_i^2$. In
a fused kernel both read the row once, so the bandwidth cost is identical; the win is in the
reduction itself (one warp-level tree reduction instead of two, fewer registers, no dependency
between mean and variance) and in having one parameter vector instead of two. Reported end-to-end
speedups are modest (Zhang and Sennrich report 7--64% for the norm itself depending on the model),
but at the scale of thousands of GPUs every elementwise microsecond counts. More importantly for
fusion, RMSNorm's output depends on $\boldsymbol{x}$ only through one scalar per row, so it fuses
especially cleanly into the preceding residual add ("add + RMSNorm" kernels are standard in vLLM and
TensorRT-LLM) and even into the following GEMM's prologue.
:::

## Gated feed-forward networks: SwiGLU {#sec:modern:swiglu}

Qwen3's MLP is a *gated linear unit* (Dauphin et al., 2017; Shazeer, 2020). Instead of one
up-projection followed by a nonlinearity, it computes two up-projections of the same input; one
passes through a nonlinearity and acts as a soft, input-dependent *gate* on the other, and the
elementwise product is projected back down: $$\begin{equation}
\mathrm{SwiGLU}(\boldsymbol{x}) = \mathbf{W}_3\big(\operatorname{SiLU}(\mathbf{W}_1\boldsymbol{x})\odot \mathbf{W}_2\boldsymbol{x}\big),\qquad
\operatorname{SiLU}(z)=z\,\sigma(z) .
\label{eq:modern:swiglu}
\end{equation}$$ The name combines Swish (= SiLU with unit temperature) and GLU.
[@fig:modern:swiglu] shows the data flow; in the code we keep the names `fc1` (gate), `fc2`
(up) and `fc3` (down) for the three bias-free linear layers.

![The SwiGLU feed-forward block: two parallel up-projections, one gated by SiLU, multiplied
elementwise and projected back down.](figures/ch06_modern_arch-4.svg){#fig:modern:swiglu}

``` {#lst:modern:swiglu .python caption="SwiGLU feed-forward network."}
class SwiGLUFeedForward(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dtype=None):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)  # gate
        self.fc2 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)  # up
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False, dtype=dtype)  # down

    def forward(self, x):
        return self.fc3(nn.functional.silu(self.fc1(x)) * self.fc2(x))
```

##### Same budget, different shape.

A GELU MLP with $d_{\text{ff}}=4d$ has $8d^2$ weights; a SwiGLU MLP with hidden width $h$ has $3dh$.
Matching the parameter (and FLOP) budget therefore means $h=\tfrac83 d$, which is why Llama-style
models often have hidden sizes that look odd (e.g. $\tfrac83\cdot4096\approx11008$ in Llama 2-7B).
It is a common misconception that the gated MLP is "more expensive": at equal parameters it costs
the same, and Shazeer (2020) found it trains to lower perplexity. Qwen3 spends a bit more than
parity: $h=3d$ for 0.6B, 1.7B and 8B, and $h=5d$ for 32B ([@tab:modern:family]).

::: systemslens
[Three GEMMs or two?]{.box-title}

Written naively, SwiGLU launches three GEMMs, a SiLU kernel and a multiply kernel, and stores two
$h$-wide intermediates for the backward pass. Production implementations concatenate $\mathbf{W}_1$
and $\mathbf{W}_2$ into one $d\times 2h$ matrix, so a single GEMM with twice the output width
produces both branches (better tile utilization, one weight read), then apply a fused
"SiLU-and-multiply" kernel that reads $2h$ and writes $h$ values per token. Under tensor parallelism
([@ch:distributed]) the gate and up matrices are split column-wise across GPUs with the
*same* partition, so each GPU can apply the gate locally and only the down projection needs an
all-reduce---one reason gated MLPs fit Megatron-style parallelism so naturally.
:::

::: deepdive
[GELU, SiLU, and friends]{.box-title}

GELU weights its input by a Gaussian CDF; SiLU weights it by a logistic sigmoid; their curves differ
by a few percent ([@fig:gpt:gelu]). SiLU is cheaper to compute exactly (one exponential),
which is one reason it became the default in gated MLPs. Variants with GELU as the gate (GeGLU, used
in Gemma) behave almost identically. What matters far more than the choice of smooth activation is
the multiplicative gate, which lets the network represent products of features directly.
:::

## Rotary position embeddings {#sec:modern:rope}

GPT-2 adds a learned vector $\boldsymbol{p}_t$ to each token embedding. That has three drawbacks:
the table has a fixed size (GPT-2 cannot see position 1024 at all), the position signal must survive
through every layer's residual stream, and attention scores depend on absolute positions even though
language is mostly translation invariant ("the cat" means the same at position 5 and 5000). Rotary
position embeddings (RoPE; Su et al., 2021) inject position *inside* attention, by rotating queries
and keys, in such a way that the score between a query at position $m$ and a key at position $n$
depends only on their offset $m-n$.

##### The idea in two dimensions.

Take a 2-D query $\boldsymbol{q}$ and key $\boldsymbol{k}$. Rotate the query by angle $m\theta$ and
the key by $n\theta$, where $R(\alpha)$ is the $2\times2$ rotation matrix: $$\begin{equation}
\langle R(m\theta)\boldsymbol{q},\ R(n\theta)\boldsymbol{k}\rangle = \boldsymbol{q}^\top R(m\theta)^\top R(n\theta)\boldsymbol{k}
= \boldsymbol{q}^\top R\big((n-m)\theta\big)\boldsymbol{k} .
\label{eq:modern:roperel}
\end{equation}$$ Rotations compose by adding angles and $R^\top=R^{-1}$, so the dot product sees
only the relative rotation. In $d_{h}$ dimensions, RoPE splits the head vector into $d_{h}/2$ pairs
and rotates pair $i$ at its own frequency $$\begin{equation}
\theta_i = \beta^{-2i/d_{h}},\qquad i=0,\ldots,d_{h}/2-1,
\label{eq:modern:ropefreq}
\end{equation}$$ where $\beta$ is the *base* (10000 in the original paper, 1000000 in Qwen3). The
first pairs rotate quickly (one radian per token for $i=0$), the last ones very slowly; together
they act like the hands of a clock with many hands, giving every position a unique, smoothly varying
signature ([@fig:modern:ropeclock]).

![RoPE rotates each 2-D pair of a query/key head by an angle proportional to the token position,
with a different speed per pair. (Angles exaggerated for
visibility.)](figures/ch06_modern_arch-5.svg){#fig:modern:ropeclock}

![Wavelength of each RoPE frequency (analytic). With $\beta=10^4$ the slowest pair completes a full
turn every $\approx54000$ tokens; with $\beta=10^6$ every $\approx5.1e6$ tokens, so many more pairs
rotate by less than one turn across a long context and can encode coarse
position.](figures/ch06_modern_arch-6.svg){#fig:modern:ropewave}

##### Two equivalent layouts.

The original formulation pairs *adjacent* dimensions $(2i,2i+1)$; the GPT-NeoX/Llama/Qwen code pairs
dimension $i$ with $i+d_{h}/2$ ("rotate half"), which is cheaper to implement with slicing. The two
are the same function up to a fixed permutation of the head dimensions: interleaved slot $2i$ holds
halves-dimension $i$ and slot $2i+1$ holds $i+d_{h}/2$. Since the query and key projections are
learned, permuting the rows of $\mathbf{W}_Q$ and $\mathbf{W}_K$ converts a checkpoint from one
layout to the other, which is exactly what weight converters between frameworks do. Lab 6 verifies
the equivalence numerically. We use the rotate-half layout throughout:

``` {#lst:modern:rope .python caption="RoPE tables and application (rotate-half layout)."}
def compute_rope_params(head_dim, theta_base=10_000.0, context_length=4096):
    i2 = torch.arange(0, head_dim, 2).float()
    inv_freq = 1.0 / theta_base ** (i2 / head_dim)      # theta_i
    pos = torch.arange(context_length).float()
    angles = pos[:, None] * inv_freq[None, :]           # (T, D/2)
    angles = torch.cat([angles, angles], dim=1)         # freq i at i and i+D/2
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin, offset=0):                 # x: (B, H, T, D)
    T, D = x.shape[-2], x.shape[-1]
    x1, x2 = x[..., : D // 2], x[..., D // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)             # (x1, x2) -> (-x2, x1)
    c, s = cos[offset:offset + T], sin[offset:offset + T]
    return (x.float() * c + rotated.float() * s).to(x.dtype)
```

For each pair $(x_1,x_2)$ the last line computes
$(x_1\cos\phi - x_2\sin\phi,\; x_2\cos\phi+x_1\sin\phi)$, a rotation by $\phi$. The `offset`
argument is the absolute position of the first token in `x`; during cached decoding
([@sec:modern:kvcache]) we process one token at a time and must rotate it by its true
position, not by zero.

::: systemslens
[RoPE in a serving engine]{.box-title}

RoPE has three systems consequences. (i) *No position table in the weights*: the cos/sin tables are
recomputed at load time (we store them as non-persistent buffers so they never appear in the
checkpoint), and the context limit becomes a property of training data rather than of the parameter
file. (ii) *The KV cache stores rotated keys*: keys are rotated before they are written to the
cache, so a cached key never needs to be touched again, and decoding applies RoPE to just one new
query and key per layer. (iii) *Relative positions make left padding free*: in a batch whose rows
are left-padded by different amounts ([@ch:serving]), every row is shifted by a constant,
which cancels in [@eq:modern:roperel]; with GPT-2's absolute embeddings the same padding
changes the outputs. Production kernels fuse RoPE into the QKV projection's epilogue or into the
attention kernel's load of $q$ and $k$, so the rotation costs no extra memory traffic.
:::

::: deepdive
[Stretching the context: PI, NTK-aware scaling, YaRN]{.box-title}

A RoPE model sees angles $m\theta_i$ only for $m$ up to its training length. Beyond it, the fast
pairs have wrapped around many times (fine) but the slow pairs reach angles never seen in training
(not fine). *Position interpolation* (Chen et al., 2023) divides all positions by a factor $s$,
squeezing a context of $sT$ into the trained range at the cost of resolution between neighbouring
tokens; it works with a short fine-tune. *NTK-aware* scaling instead enlarges the base to
$\beta' = \beta\, s^{d_{h}/(d_{h}-2)}$, which leaves the fastest frequency unchanged and slows the
slowest by exactly $s$---high-frequency (local) information is preserved, low-frequency (global)
information is interpolated. *YaRN* (Peng et al., 2023) interpolates per frequency band and rescales
attention temperature; it is what Qwen3 recommends for running beyond its native 32768 tokens (up to
$4\times$). Static YaRN scaling applies to every input regardless of length and can slightly hurt
short prompts, so it should be enabled only when long inputs are actually expected. Lab 6 implements
PI and the NTK base formula.
:::

## Grouped-query attention {#sec:modern:gqa}

In multi-head attention every query head has its own key and value head. During autoregressive
decoding, keys and values of all past tokens are cached ([@sec:modern:kvcache]), and
reading that cache is the dominant memory traffic of long-context inference. *Multi-query attention*
(MQA; Shazeer, 2019) shares a single key/value head among all query heads, shrinking the cache by
$H\times$ but costing some quality. *Grouped-query attention* (GQA; Ainslie et al., 2023)
interpolates: the $H$ query heads are divided into $G$ groups and each group shares one key/value
head ([@fig:modern:gqa]). With $G=H$ it is MHA, with $G=1$ it is MQA; Qwen3 uses $G=8$ for
every model size.

![Multi-head vs. grouped-query attention with 8 query heads. Each orange square is a K/V head that
must be stored in the KV cache; GQA keeps all query heads but cuts the cache by
$H/G$.](figures/ch06_modern_arch-7.svg){#fig:modern:gqa}

``` {#lst:modern:gqa .python caption="Grouped-query attention with QK-norm, RoPE and an optional KV cache."}
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim,
                 qk_norm=True, dtype=None):
        super().__init__()
        self.H, self.G, self.D = num_heads, num_kv_groups, head_dim
        self.group_size = num_heads // num_kv_groups
        kw = dict(bias=False, dtype=dtype)
        self.W_query = nn.Linear(d_in, self.H * self.D, **kw)
        self.W_key = nn.Linear(d_in, self.G * self.D, **kw)     # G, not H, heads
        self.W_value = nn.Linear(d_in, self.G * self.D, **kw)
        self.out_proj = nn.Linear(self.H * self.D, d_in, **kw)
        self.q_norm = RMSNorm(self.D) if qk_norm else None
        self.k_norm = RMSNorm(self.D) if qk_norm else None

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape
        q = self.W_query(x).view(B, T, self.H, self.D).transpose(1, 2)
        k = self.W_key(x).view(B, T, self.G, self.D).transpose(1, 2)
        v = self.W_value(x).view(B, T, self.G, self.D).transpose(1, 2)
        if self.q_norm is not None:                   # QK-norm, per head
            q, k = self.q_norm(q), self.k_norm(k)
        off = cache.seq_len(self.layer_idx) if cache is not None else 0
        q, k = apply_rope(q, cos, sin, off), apply_rope(k, cos, sin, off)
        if cache is not None:                         # cache holds G heads only
            k, v = cache.update(self.layer_idx, k, v)
        k = k.repeat_interleave(self.group_size, dim=1)   # expand to H heads
        v = v.repeat_interleave(self.group_size, dim=1)
        mask = causal_mask(T, k.shape[2], x.device)       # handles the offset
        scores = (q @ k.transpose(-2, -1)) / self.D ** 0.5
        scores = scores.masked_fill(mask, float("-inf"))
        w = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        ctx = (w @ v).transpose(1, 2).reshape(B, T, self.H * self.D)
        return self.out_proj(ctx)
```

Compared to the MHA of [@ch:attention], four lines changed: the key and value projections are
$G/H$ as wide; QK-norm normalizes each query and key head; RoPE rotates $q$ and $k$ with the cache
offset; and `repeat_interleave` copies each K/V head to the $H/G$ query heads of its group *after*
the cache update, so the cache itself stays small. Note also that Qwen3-0.6B's query projection maps
$d_{\text{model}}=1024$ to $Hd_{h}=16\times128=2048$: the attention width is decoupled from the
residual width.

### QK-norm

Qwen3 additionally applies an RMSNorm (with its own learned gain, shared across heads) to every
query and key head vector before RoPE. Attention logits are dot products
$\boldsymbol{q}\cdot\boldsymbol{k}$; if the norms of $\boldsymbol{q}$ and $\boldsymbol{k}$ drift
upwards during training, the softmax saturates into near one-hot distributions, gradients vanish and
the loss can spike. Normalizing $q$ and $k$ bounds the logits (Dehghani et al., 2023 used it to
stabilize 22B-parameter vision transformers) and lets large models train at higher learning rates.
The cost is two tiny RMSNorms over $d_{h}=128$ elements per head.

::: napkinmath
[KV-cache bytes per token]{.box-title}

Each cached token stores one key and one value vector per KV head per layer:
$$\text{KV bytes/token} = 2\times L\times G\times d_{h}\times \text{bytes/element}.$$ For Qwen3-0.6B
in bf16: $2\times28\times8\times128\times2=114688$ B $=112\,\mathrm{\kibi\byte}$. With full MHA (16
K/V heads) it would be 224 . For Qwen3-32B the saving is $64/8=8\times$: 256  instead of 2  per
token. A single 32768-token conversation with the 32B model therefore needs 8.6  of cache with GQA
and would need 69  with MHA---more than the model's own weights. (For comparison, GPT-2 small needs
only 36  per token; Qwen3-0.6B's cache is three times larger because it is deeper and its attention
is twice as wide as its residual stream.)
:::

::: systemslens
[GQA raises the arithmetic intensity of decoding]{.box-title}

During a decode step each sequence contributes one query per head. Attention over a context of $T$
tokens then costs $4T Hd_{h}$ FLOPs and must read $2\cdot2TGd_{h}$ bytes of bf16 cache, an
arithmetic intensity of $H/G$ FLOP/byte: 1 for MHA, the group size for GQA (2 for Qwen3-0.6B, 4 for
8B, 8 for 32B). That is still far below any GPU's ridge point---decode attention is memory bound no
matter what---but it means the time spent reading the cache, which dominates long-context decoding,
drops by the group size. A well-written kernel therefore does *not* materialize the
`repeat_interleave`'d tensors as our reference code does; it loads each K/V tile once into shared
memory and lets all $H/G$ query heads of the group consume it (FlashAttention and FlashInfer do
this; PyTorch's SDPA exposes it via `enable_gqa=True`). Our explicit expansion multiplies the bytes
read back up by $H/G$ and throws the benefit away---a good example of an optimization that exists
only if the kernel knows about it.
:::

## The Qwen3 transformer block {#sec:modern:block}

The block is the GPT-2 block with the new parts slotted in ([@fig:modern:block]): RMSNorm
before attention, GQA with QK-norm and RoPE, residual add, RMSNorm before the SwiGLU MLP, residual
add. There is no dropout and no bias anywhere. The RoPE tables are owned by the model and passed
down, so all 28 layers share one copy.

![One Qwen3-0.6B transformer block (repeated 28 times), showing where the KV cache plugs
in.](figures/ch06_modern_arch-8.svg){#fig:modern:block}

``` {#lst:modern:block .python caption="The Qwen3 transformer block."}
class Qwen3Block(nn.Module):
    def __init__(self, cfg, dtype=None):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"], num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"], head_dim=cfg["head_dim"],
            qk_norm=cfg["qk_norm"], dtype=dtype)
        self.ff = SwiGLUFeedForward(cfg["emb_dim"], cfg["hidden_dim"], dtype=dtype)
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])

    def forward(self, x, cos, sin, cache=None, pad_mask=None):
        x = x + self.att(self.norm1(x), cos, sin, cache, pad_mask)
        x = x + self.ff(self.norm2(x))
        return x
```

## The main model {#sec:modern:model}

``` {#lst:modern:model .python caption="Qwen3 configuration and model (cf. `llmsys/qwen3.py`)."}
QWEN3_CONFIG_06B = dict(
    vocab_size=151_936, context_length=40_960, emb_dim=1024,
    n_heads=16, n_kv_groups=8, head_dim=128, qk_norm=True,
    n_layers=28, hidden_dim=3072, rope_base=1_000_000.0, tie_embeddings=True)

class Qwen3Model(nn.Module):
    def __init__(self, cfg, dtype=None):
        super().__init__()
        self.cfg = cfg
        V, d = cfg["vocab_size"], cfg["emb_dim"]
        self.tok_emb = nn.Embedding(V, d, dtype=dtype)
        self.trf_blocks = nn.ModuleList(
            Qwen3Block(cfg, dtype) for _ in range(cfg["n_layers"]))
        for i, blk in enumerate(self.trf_blocks):
            blk.att.layer_idx = i                 # each layer knows its cache slot
        self.final_norm = RMSNorm(d)
        self.out_head = nn.Linear(d, V, bias=False, dtype=dtype)
        if cfg["tie_embeddings"]:
            self.out_head.weight = self.tok_emb.weight
        cos, sin = compute_rope_params(cfg["head_dim"], cfg["rope_base"],
                                       cfg["context_length"])
        self.register_buffer("cos", cos, persistent=False)  # not in checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, in_idx, cache=None, pad_mask=None):
        x = self.tok_emb(in_idx)                  # no position embedding!
        for blk in self.trf_blocks:
            x = blk(x, self.cos, self.sin, cache=cache, pad_mask=pad_mask)
        return self.out_head(self.final_norm(x))
```

A few configuration values deserve comment. The vocabulary of 151936 rows is padded beyond the
roughly 151.7k tokens the tokenizer actually defines---a multiple of 128 for the GEMM reasons of
[@sec:gpt:arch]. `context_length` is 40960 because the RoPE tables must cover the longest
sequence we ever run, and Qwen configures that much headroom, although the model was trained on
sequences of up to 32768 tokens. Finally, the checkpoint is stored in bfloat16; we instantiate in
bf16 directly (`dtype=torch.bfloat16`) so the 0.6B model needs about 1.2  rather than 2.4 , while
the RoPE tables stay in fp32 for accuracy at large positions.

::: pitfall
[Casting the whole model to bf16]{.box-title}

`model.to(torch.bfloat16)` converts *all* floating-point buffers, including our cos/sin tables. In
bf16, positions above a few hundred are already represented with errors of several units in the last
place of the angle, and the rotation becomes visibly wrong at long range. Construct the parameters
in bf16 (or cast only `model.parameters()`), and keep the RoPE tables in fp32. Lab 6's `run.py` does
the latter.
:::

::: napkinmath
[Where Qwen3-0.6B's parameters are]{.box-title}

Using the Lab 6 formula: the tied embedding is $151936\times1024=155.6\,\mathrm{M}$ parameters (26%
of the model); each block has
$1024\cdot2048\cdot2 + 1024\cdot1024\cdot2 + 3\cdot1024\cdot3072 \approx
15.7\,\mathrm{M}$ (attention 6.3M, MLP 9.4M); 28 blocks give 440 ; total 596 . In bf16 the weights
take 1.19 . A decode step at batch 1 reads all of them: on an H100 at 3.35  the lower bound is
$1.19\,\mathrm{\giga\byte}/3.35\,\mathrm{\tera\byte\per\second}
\approx0.36\,\mathrm{\milli\second}$, i.e. at most $\approx2800$ tokens/s for a single
stream---before any kernel launch overhead, which in practice dominates a model this small
([@ch:inference]).
:::

## KV cache {#sec:modern:kvcache}

Autoregressive generation re-reads the whole prefix at every step ([@sec:gpt:generate]).
Since a past token's keys and values in every layer depend only on the tokens before it, they never
change and can be stored. With a KV cache, the first forward pass processes the prompt (*prefill*)
and stores its keys and values; each later step processes *only* the newest token (*decode*),
appends its key and value to the cache, and attends over the cached history. [@ch:inference]
studies the cache in depth; here we need only its interface, which our attention module already
uses:

``` {#lst:modern:kv .python caption="The two cache flavours (from `llmsys/kvcache.py`, abridged)."}
class KVCache:                     # grows by concatenation
    def __init__(self, n_layers):
        self.k, self.v = [None] * n_layers, [None] * n_layers
    def seq_len(self, layer_idx=0):
        return 0 if self.k[layer_idx] is None else self.k[layer_idx].shape[2]
    def update(self, layer_idx, k_new, v_new):
        if self.k[layer_idx] is None:
            self.k[layer_idx], self.v[layer_idx] = k_new, v_new
        else:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k_new], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v_new], dim=2)
        return self.k[layer_idx], self.v[layer_idx]

class StaticKVCache:               # preallocated (B, G, max_len, D) buffers
    def update(self, layer_idx, k_new, v_new):
        s = self.lengths[layer_idx]; e = s + k_new.shape[2]
        self.k[layer_idx][:, :, s:e] = k_new
        self.v[layer_idx][:, :, s:e] = v_new
        self.lengths[layer_idx] = e
        return self.k[layer_idx][:, :, :e], self.v[layer_idx][:, :, :e]
```

The only subtle point is the causal mask. With $T_q$ new queries and $T_k=T_\text{past}+T_q$ keys,
query $i$ sits at absolute position $T_\text{past}+i$ and may attend to keys
$0,\ldots,T_\text{past}+i$. The mask is therefore a rectangular $(T_q\times T_k)$ matrix whose
forbidden region starts at diagonal $T_\text{past}+1$ ([@fig:modern:cachemask]); in the
decode case $T_q=1$ and the mask is empty.

![Causal mask for a chunk of 4 new tokens attending over a cache of 6 earlier tokens. The mask is
`triu(ones(4,10), diagonal=6+1)`; for single-token decode it disappears
entirely.](figures/ch06_modern_arch-9.svg){#fig:modern:cachemask}

::: systemslens
[Concatenation vs. preallocation]{.box-title}

`torch.cat` allocates a new tensor and copies the whole history at every step, so generating $n$
tokens moves $O(n^2)$ bytes per layer and fragments the allocator. The static cache allocates
$(B,G,T_\text{max},d_{h})$ once and writes one slice per step: $O(n)$ bytes, fixed addresses, and
fixed shapes---which is what CUDA graphs and `torch.compile(mode="reduce-overhead")` need
([@ch:inference]). Its weakness is the opposite: it reserves memory for $T_\text{max}$ tokens
per sequence even if most sequences are short, which is exactly the waste PagedAttention eliminates
([@ch:serving]).
:::

## The tokenizer {#sec:modern:tokenizer}

Qwen3 uses a byte-level BPE tokenizer of the kind we built in [@ch:text], with roughly 151.6k
learned tokens covering many languages and code, plus a few dozen *special tokens* that structure
conversations and reasoning. The ones we need are:

::: center
| Token                        | Role                                                             |
|:-----------------------------|:-----------------------------------------------------------------|
| `<|endoftext|>`              | document separator in pretraining; used as padding               |
| `<|im_start|>`, `<|im_end|>` | begin/end of a chat turn (ChatML); `<|im_end|>` stops generation |
| `<think>`, `</think>`        | delimit the model's reasoning trace                              |
:::

A chat-tuned model expects its input in the format it was trained on. For Qwen3 that is ChatML:

``` {#lst:modern:chatml caption="The Qwen3 chat format for one user turn, thinking disabled."}
<|im_start|>user
What is 7 * 8?<|im_end|>
<|im_start|>assistant
<think>

</think>
```

Qwen3's post-trained checkpoints are *hybrid* reasoning models: by default the assistant first
writes a reasoning trace inside `<think>...</think>` and then the answer. Prefilling an *empty*
think block, as above, switches thinking off; leaving it out lets the model think. Our tokenizer
wrapper (`llmsys.qwen3.Qwen3Tokenizer`) delegates the BPE itself to the fast Rust implementation in
the `tokenizers` library and adds the chat template:

``` {#lst:modern:tok .python caption="Applying the chat template (from `llmsys/qwen3.py`)."}
def apply_chat_template(self, prompt_or_messages, enable_thinking=False):
    msgs = prompt_or_messages
    if isinstance(msgs, str):
        msgs = [{"role": "user", "content": msgs}]
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
                for m in msgs)
    s += "<|im_start|>assistant\n"
    if not enable_thinking:
        s += "<think>\n\n</think>\n\n"       # empty trace = answer directly
    return s
```

::: pitfall
[Base vs. chat formatting]{.box-title}

A *base* checkpoint (e.g. `Qwen3-0.6B-Base`) has never been trained on ChatML; feed it plain text,
and stop on `<|endoftext|>`. A post-trained checkpoint should always receive the template, and
generation should stop on `<|im_end|>`. Mixing these up is the most common reason a from-scratch
implementation "works" but produces rambling output. Evaluation numbers are also sensitive to the
exact template; [@ch:eval] returns to this.
:::

## Using the model {#sec:modern:use}

Loading the official weights is a matter of mapping names. The Hugging Face checkpoint is a set of
safetensors files whose tensor names follow the `transformers` implementation;
[@tab:modern:names] gives the mapping to our modules, and `load_qwen3_hf_state_dict`
applies it with a shape check on every tensor.

::: {#tab:modern:names .table-float}
| Hugging Face name                                    | Our module                               |
|:-----------------------------------------------------|:-----------------------------------------|
| `model.embed_tokens.weight`                          | `tok_emb.weight` (also the head if tied) |
| `model.layers.`$i$`.input_layernorm.weight`          | `trf_blocks[`$i$`].norm1.scale`          |
| `model.layers.`$i$`.self_attn.{q,k,v,o}_proj.weight` | `att.W_query/W_key/W_value/out_proj`     |
| `model.layers.`$i$`.self_attn.{q,k}_norm.weight`     | `att.q_norm.scale`, `att.k_norm.scale`   |
| `model.layers.`$i$`.post_attention_layernorm.weight` | `trf_blocks[`$i$`].norm2.scale`          |
| `model.layers.`$i$`.mlp.{gate,up,down}_proj.weight`  | `ff.fc1`, `ff.fc2`, `ff.fc3`             |
| `model.norm.weight`                                  | `final_norm.scale`                       |
| `lm_head.weight` (untied models only)                | `out_head.weight`                        |

: Weight-name mapping between the Hugging Face Qwen3 checkpoint and our implementation.
:::

``` {#lst:modern:use .python caption="Loading Qwen3-0.6B and streaming a reply."}
# one-time, on a machine with internet:
#   huggingface-cli download Qwen/Qwen3-0.6B --local-dir qwen3-0.6b
from llmsys.qwen3 import load_qwen3_from_dir, Qwen3Tokenizer
from llmsys.generate import generate_stream

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_qwen3_from_dir("qwen3-0.6b", size="0.6B",
                            dtype=torch.bfloat16, device=device)
tok = Qwen3Tokenizer("qwen3-0.6b/tokenizer.json")

prompt = "Explain RoPE in two sentences."
ids = torch.tensor([tok.encode(prompt, chat=True)], device=device)
for t in generate_stream(model, ids, 200, eos_id=tok.eos_token_id):
    print(tok.decode(t[0].tolist()), end="", flush=True)
```

Before trusting a from-scratch port, compare its logits with a reference implementation on a few
prompts. Expect agreement to within bf16 noise (maximum absolute logit differences of order
$10^{-2}$), not bit-exactness: two correct implementations can sum in different orders, use
different attention kernels, or upcast at different points. Large discrepancies almost always come
from one of four bugs: a wrong RoPE layout, RoPE tables in bf16, forgetting QK-norm, or a missing
chat template.

## Scaling up: larger Qwen3 models {#sec:modern:larger}

The 0.6B model is ideal for learning: it runs on a laptop and every experiment is fast. But the code
in this chapter is size-agnostic, and running a larger model is mostly a matter of a different
configuration and a different checkpoint.

### Larger dense configurations

[@tab:modern:family] lists the dense Qwen3 family. All members share the vocabulary, head
dimension 128, eight KV heads, QK-norm and the RoPE base; they differ in width, depth, number of
query heads and MLP width. The smaller three tie their embeddings; from 8B on, the output head is a
separate matrix.

::: {#tab:modern:family .table-float}
| Model | $d_{\text{model}}$ | $L$ | $H$ | $G$ | $h$ (MLP) | Tied | Params | KV/token (bf16) |
|:------|-------------------:|----:|----:|----:|----------:|-----:|-------:|----------------:|
| 0.6B  |               1024 |  28 |  16 |   8 |      3072 |  yes |  0.60B |            112  |
| 1.7B  |               2048 |  28 |  16 |   8 |      6144 |  yes |  1.72B |            112  |
| 4B    |               2560 |  36 |  32 |   8 |      9728 |  yes |  4.02B |            144  |
| 8B    |               4096 |  36 |  32 |   8 |     12288 |   no |  8.19B |            144  |
| 14B   |               5120 |  40 |  40 |   8 |     17408 |   no | 14.77B |            160  |
| 32B   |               5120 |  64 |  64 |   8 |     25600 |   no | 32.76B |            256  |

: The dense Qwen3 family. Parameter and KV-cache sizes computed from the configurations with the
Lab 6 code. Note how the KV cache per token grows much more slowly than the parameter count.
:::

### Downloading larger checkpoints

Checkpoints above a few gigabytes are split into *shards* (`model-00001-of-00005.safetensors`, ...)
plus a JSON index, `model.safetensors.index.json`, that maps every tensor name to the shard
containing it. The simplest robust approach is to download the whole repository snapshot
(`huggingface-cli download Qwen/Qwen3-8B --local-dir qwen3-8b`) and let the loader read every
`*.safetensors` file in the directory, as `load_qwen3_from_dir` does. Keep each model in its own
directory; mixing shards of different models in one folder is a classic way to load garbage
silently, which is why our loader checks every tensor's shape.

### Loading a larger base model

Naive loading of an 8B model needs memory for *two* copies of the weights: the randomly initialized
model and the state dict read from disk. Three standard tricks avoid that (Lab 6's `run.py` uses the
first and third):

``` {#lst:modern:meta .python caption="Memory-frugal loading: meta-device construction and per-shard copy."}
from safetensors import safe_open

with torch.device("meta"):             # 1) allocate no storage at all
    model = Qwen3Model(QWEN3_CONFIGS["8B"], dtype=torch.bfloat16)
model = model.to_empty(device="cuda")  # 2) uninitialized storage on the GPU
for shard in sorted(glob.glob("qwen3-8b/*.safetensors")):
    # 3) memory-map the shard and copy one tensor at a time
    with safe_open(shard, framework="pt", device="cuda") as f:
        for name in f.keys():
            copy_into(model, name, f.get_tensor(name))  # name map: Table 6.2
# buffers created on the meta device (RoPE tables) must be recomputed
```

Meta-device construction skips the random initialization (which for 8B parameters is also tens of
seconds of CPU time), `to_empty` materializes uninitialized storage directly on the GPU, and
`safe_open` memory-maps each shard so only one tensor at a time is resident in host memory. Peak
memory is then the model plus one tensor, rather than twice the model.

### Loading a larger reasoning variant

Each Qwen3 size is published as a `-Base` checkpoint (pretrained only) and as a post-trained
hybrid-thinking checkpoint without suffix. Switching between them changes only the weights directory
and the tokenizer settings: the reasoning variant needs the chat template and stops on `<|im_end|>`,
and thinking is controlled by the empty-think-block trick of [@sec:modern:tokenizer].
Everything else---model class, generation loop, KV cache---is identical. In
[@ch:eval; @ch:scaling] we will use the base 0.6B model as the student we try to improve, and
the reasoning variants as reference points and as distillation teachers ([@ch:distill]).

### Practical recommendations and memory planning

Which size should you use? The honest systems answer is: the largest one whose weights *plus the KV
cache you need* fit in memory with headroom. [@tab:modern:plan] applies the formulas of
this chapter to two common devices, assuming bf16 weights and cache and 10% of memory reserved for
activations, workspace and fragmentation.

::: {#tab:modern:plan .table-float}
+------------------+-----------------------------------------+-------------------------------------+
|                  | 24 GB GPU (e.g. RTX 4090, L4)           | 80 GB GPU (A100/H100)               |
+:=================+================:+======================:+============:+======================:+
| 2-3(lr)4-5 Model | weights         | max. 8k-token         | weights     | max. 8k-token         |
|                  |                 | sequences             |             | sequences             |
+------------------+-----------------+-----------------------+-------------+-----------------------+
| 0.6B             | 1.2 GB          | 21                    | 1.2 GB      | 75                    |
+------------------+-----------------+-----------------------+-------------+-----------------------+
| 1.7B             | 3.4 GB          | 19                    | 3.4 GB      | 72                    |
+------------------+-----------------+-----------------------+-------------+-----------------------+
| 4B               | 8.0 GB          | 11                    | 8.0 GB      | 52                    |
+------------------+-----------------+-----------------------+-------------+-----------------------+
| 8B               | 16.4 GB         | 4                     | 16.4 GB     | 46                    |
+------------------+-----------------+-----------------------+-------------+-----------------------+
| 14B              | 29.5 GB         | --                    | 29.5 GB     | 31                    |
+------------------+-----------------+-----------------------+-------------+-----------------------+
| 32B              | 65.5 GB         | --                    | 65.5 GB     | 3                     |
+------------------+-----------------+-----------------------+-------------+-----------------------+

: How many concurrent 8192-token sequences fit next to the weights (bf16, 10% reserve), from
`max_concurrent_sequences` in Lab 6. This number bounds the batch size, and hence the throughput, of
a serving engine.
:::

A few rules of thumb follow. For the experiments in this book, 0.6B on any GPU (or even CPU) is
enough. If you have a 24 GB card, the 4B model is the sweet spot for RL and evaluation: it fits with
room for a useful batch. The 8B model fits on 24 GB but leaves room for only a handful of long
sequences, which makes batched sampling ([@ch:scaling; @ch:grpo]) slow; quantization
([@ch:efficient]) or a larger GPU helps. The 32B model needs either an 80 GB GPU with 8-bit
weights, or tensor parallelism across two or more GPUs ([@ch:distributed; @ch:efficient]).

## Beyond dense models: mixture of experts {#sec:modern:moe}

Every token in a dense model uses every parameter. A *mixture-of-experts* (MoE) layer replaces the
single MLP with $E$ smaller MLPs ("experts") and a learned *router* that sends each token to only
$k\ll E$ of them (Shazeer et al., 2017; Fedus et al., 2022). The parameter count grows with $E$, but
the FLOPs per token grow only with $k$. Qwen3 itself ships two MoE models next to the dense ones;
Qwen3-30B-A3B, for example, has 48 layers each with $E=128$ experts of hidden width 768, of which
$k=8$ are active per token. Its total parameter count is about 30.5B, but only about 3.3B are used
for any given token (both numbers from our `moe_param_counts` applied to its configuration), so its
per-token compute is close to a 3B dense model while its capacity is closer to a 30B one.

![The MoE data path as implemented in `llmsys/moe.py`: route, permute token copies into per-expert
contiguous groups, run each expert as one batched GEMM, then un-permute and combine with the gate
weights. In expert-parallel training the permute and un-permute steps become all-to-all exchanges
between GPUs.](figures/ch06_modern_arch-10.svg){#fig:modern:moe}

##### Routing.

For a token $\boldsymbol{x}$ the router computes probabilities
$\boldsymbol{p}=\operatorname{softmax}(\mathbf{W}_r\boldsymbol{x})$ over the $E$ experts, keeps the
$k$ largest, renormalizes them to sum to one (the Qwen-MoE and Mixtral convention), and outputs
$\boldsymbol{y}=\sum_{e\in\text{top-}k} p_e\,\mathrm{Expert}_e(\boldsymbol{x})$. The reference
implementation in [@lst:modern:moe] makes the systems structure explicit: rather than
looping over tokens, it sorts the $(\text{token},\text{slot})$ assignments by expert so that each
expert processes one contiguous slice---the single-GPU analogue of the permutation that
expert-parallel systems perform with an all-to-all.

``` {#lst:modern:moe .python caption="Sort-based MoE dispatch (from `llmsys/moe.py`, abridged)."}
def forward(self, x):
    x2d = x.reshape(-1, x.shape[-1]); N = x2d.shape[0]
    probs = torch.softmax(self.gate(x2d).float(), dim=-1)       # (N, E)
    w, idx = torch.topk(probs, self.top_k, dim=-1)              # (N, k)
    w = (w / w.sum(-1, keepdim=True)).to(x.dtype)
    expert = idx.flatten()                                      # (N*k,)
    token = torch.arange(N).repeat_interleave(self.top_k)
    order = torch.argsort(expert, stable=True)                  # permute
    expert, weight, token = expert[order], w.flatten()[order], token[order]
    counts = torch.bincount(expert, minlength=self.n_experts).tolist()
    out, start = torch.zeros_like(x2d), 0
    for e, n in enumerate(counts):
        if n:
            t, g = token[start:start + n], weight[start:start + n, None]
            out.index_add_(0, t, g * self.experts[e](x2d[t]))   # combine
            start += n
    return out.reshape(x.shape)
```

##### Load balancing.

Left alone, the router tends to collapse: a few experts win early, receive more gradient, get
better, and win more. The standard remedy is an auxiliary loss that is minimized by uniform routing.
With $f_e$ the fraction of token slots routed to expert $e$ (a count, not differentiable) and $P_e$
the mean router probability of expert $e$ (differentiable), $$\begin{equation}
\mathcal L_\text{aux} = E\sum_{e=1}^{E} f_e\,P_e ,
\label{eq:modern:aux}
\end{equation}$$ which equals 1 under perfect balance and grows towards $E$ under collapse (Fedus et
al., 2022). It is added to the language-modeling loss with a small coefficient. Some recent models
replace it with bias-based "auxiliary-loss-free" balancing (DeepSeek-V3), or compute balance
statistics over a global batch rather than per micro-batch (as the Qwen team reports doing).

::: systemslens
[Why MoE is a systems problem]{.box-title}

MoE trades FLOPs for memory and communication. (i) *Memory*: all $E$ experts must be resident
somewhere, so a 30B-total MoE needs the memory of a 30B dense model even though it computes like a
3B one; it is cheap to *run* only once you can afford to *hold* it. (ii) *Expert parallelism*:
experts are spread across GPUs, and every MoE layer performs two all-to-all exchanges (dispatch
tokens to their experts, return results). Per token and layer that is $2kd_{\text{model}}$ bf16
values on the wire---for Qwen3-30B-A3B, $2\cdot8\cdot2048\cdot2=64\,\mathrm{\kibi\byte}$ per token
per layer, which at 48 layers and thousands of tokens per batch quickly approaches NVLink bandwidth
([@ch:distributed]). (iii) *Load imbalance*: the slowest expert determines the layer's
latency, so systems either impose a *capacity* (each expert processes at most $c\cdot Nk/E$ tokens;
overflow tokens are dropped and pass through the residual connection) or use dropless grouped-GEMM
kernels (MegaBlocks; Gale et al., 2023) that handle ragged per-expert batches efficiently. (iv)
*Small GEMMs*: each expert sees only about $Nk/E$ tokens, so expert GEMMs are much narrower than a
dense MLP's and are often memory bound---during decode especially, where an MoE model reads far more
distinct weights per step than a dense model of equal active size.
:::

::: napkinmath
[Total vs. active parameters]{.box-title}

Per MoE layer with SwiGLU experts of hidden width $h$: each expert has $3dh$ weights; the router has
$dE$. For Qwen3-30B-A3B ($d=2048$, $h=768$, $E=128$, $k=8$) one expert is 4.72  parameters, so a
layer holds 604  expert parameters but activates 37.7 . Over 48 layers plus attention (18.9  per
layer with 32 query and 4 KV heads) and untied embeddings (622 ), the total is
$\approx30.5\,\mathrm{B}$ and the active count $\approx3.35\,\mathrm{B}$. Training FLOPs scale with
the active count ($6\times3.35$B per token), but serving memory scales with the total. At batch 1 a
decode step reads $\approx6.7\,\mathrm{\giga\byte}$ of active bf16 weights; at large batch, nearly
all experts are hit and the step reads close to all 61 .
:::

::: deepdive
[Other directions in attention design]{.box-title}

GQA is not the only way to shrink the KV cache. *Multi-head latent attention* (MLA, DeepSeek-V2,
2024) caches a single low-rank latent vector per token from which per-head keys and values are
reconstructed, compressing the cache further than GQA while keeping per-head expressivity.
*Sliding-window* attention (Mistral, Gemma) limits some or all layers to the last $w$ tokens,
bounding their cache at $w$ entries. *Hybrid* architectures interleave a minority of full-attention
layers with linear-attention or state-space layers whose state is constant-size, so most layers need
no growing cache at all (Qwen3-Next is one such design). All of these are, at heart, answers to the
same napkin-math question as GQA: how many bytes per token does the model need to remember, and how
fast can the hardware read them?
:::

## Summary

- Qwen3 keeps GPT-2's pre-norm decoder skeleton but replaces LayerNorm with RMSNorm, the GELU MLP
  with SwiGLU, learned positions with RoPE, and MHA with GQA plus QK-norm; it drops biases and
  dropout.

- RMSNorm needs one row statistic instead of two and fuses cleanly with residual adds; compute it in
  fp32.

- SwiGLU at hidden width $\tfrac83 d$ matches a GELU MLP's budget; production code fuses gate and up
  projections into one GEMM.

- RoPE rotates query/key pairs at geometrically spaced frequencies so that scores depend only on
  relative position; it needs no weights, makes the cache store rotated keys, and makes left padding
  harmless. Keep its tables in fp32.

- GQA shares each K/V head among $H/G$ query heads, cutting the KV cache (and decode attention
  traffic) by $H/G$; kernels must avoid materializing the expanded heads to realize the gain.

- KV bytes per token $=2LGd_{h}\cdot$bytes; Qwen3-0.6B needs 112 , Qwen3-32B 256 . Together with the
  weights, this determines how many sequences fit on a device.

- Larger dense Qwen3 models use the same code; load them with meta-device construction and per-shard
  copying to avoid holding two copies.

- Mixture-of-experts layers decouple parameters from per-token FLOPs at the cost of memory,
  all-to-all communication and load-balancing complexity.

## Exercises

::: exercise
[RMSNorm vs. LayerNorm]{.box-title}

Show that if every token vector has zero mean, RMSNorm and LayerNorm (with $\beta=0$) produce
identical outputs. Then measure, with `saved_tensors_hooks` from [@ch:gpt], how many bytes
each of the two explicit implementations saves for the backward pass on a $(4,256,1024)$ input.
:::

::: exercise
[SwiGLU parity]{.box-title}

A GELU MLP with $d_{\text{ff}}=4d$ and a SwiGLU MLP with hidden width $h$ are to have equal
parameter counts. Derive $h$. Llama 2-7B uses $d=4096$ and $h=11008$; by what percentage does it
deviate from parity, and why might a model designer round $h$ to a multiple of 256?
:::

::: exercise
[RoPE relativity]{.box-title}

Prove [@eq:modern:roperel] for the $d_{h}$-dimensional case in the rotate-half layout by
writing the rotation as a block-diagonal matrix in a permuted basis. Then verify numerically that
$\langle\mathrm{RoPE}_m(q),\mathrm{RoPE}_n(k)\rangle$ is unchanged when both $m$ and $n$ are shifted
by the same amount, and explain why that makes left padding harmless.
:::

::: exercise
[Layout conversion]{.box-title}

Write a function that converts the $\mathbf{W}_Q$ and $\mathbf{W}_K$ weights of a checkpoint trained
with interleaved RoPE into the rotate-half layout, and test it end to end with a tiny GQA module.
Does $\mathbf{W}_V$ need to be permuted? Why or why not?
:::

::: exercise
[KV budget]{.box-title}

For Qwen3-8B, compute the KV-cache size of one 131072-token context in bf16 and in fp8. Could one
such context fit on an 80 GB GPU next to the bf16 weights? What if you also quantize the weights to
8 bits?
:::

::: exercise
[GQA intensity]{.box-title}

Derive the arithmetic intensity of single-token decode attention as a function of $H$, $G$ and the
bytes per cache element, and evaluate it for Qwen3-0.6B, 8B and 32B in bf16 and fp8. How does the
materialized `repeat_interleave` in our reference code change the bytes actually read?
:::

::: exercise
[RoPE precision]{.box-title}

Compute the RoPE tables for Qwen3 in fp32 and in bf16 and plot the absolute angle error
$|\phi_\text{bf16}-\phi_\text{fp32}|$ for the fastest frequency at positions 0 to 32768. At roughly
what position does the error exceed $0.1$ radians?
:::

::: exercise
[Context extension]{.box-title}

Using Lab 6's `ntk_aware_base`, compute the base Qwen3 would need to stretch its trained context by
$4\times$ with NTK-aware scaling. Which pair index $i$ has its wavelength changed by the largest
factor, and which by the smallest?
:::

::: exercise
[MoE capacity]{.box-title}

With $N=4096$ tokens, $E=64$, $k=2$ and capacity factor $c=1.25$, how many token slots can each
expert accept? If routing were uniformly random, use a normal approximation to estimate the expected
fraction of dropped token slots. How does the answer change with $c=1.0$ and $c=2.0$?
:::

::: exercise
[Expert-parallel communication]{.box-title}

Qwen3-30B-A3B runs with its 128 experts split evenly over 8 GPUs. For a batch of 16384 tokens,
estimate the all-to-all bytes per MoE layer per GPU (dispatch plus combine, bf16), and the time on
NVLink at 450  per direction. Compare with the time of the experts' GEMMs on one H100.
:::

::: exercise
[Memory-frugal loading]{.box-title}

Instrument [@lst:modern:meta] to record peak memory (CUDA's `max_memory_allocated`, or the
resident set size on CPU) and compare it with naive loading via `load_state_dict` for the 1.7B
model.
:::

## Lab 6: Qwen3 from scratch, and planning its memory {#sec:modern:lab}

::: labbox
[Qwen3 from scratch, and planning its memory]{.box-title}

**Code:** `code/labs/lab06_qwen3`

**Goals**

Re-implement every Qwen3 component so that your model is logit-identical to `llmsys.qwen3` (and
therefore loads the official weights), prove RoPE's properties numerically, implement MoE dispatch
the way systems do, and turn the chapter's memory formulas into a planning tool.

**Setup**

``` bash
cd code
pytest -q labs/lab06_qwen3                      # tests your starter.py
python labs/lab06_qwen3/run.py --impl starter --plan --gpu-gb 24 --seq 8192
```

**Tasks**

1.  **RMSNorm.** Implement `rms_norm` with fp32 accumulation.

2.  **RoPE.** Implement `compute_rope`, `apply_rope` (rotate-half), `apply_rope_interleaved`
    (complex arithmetic), and `halves_to_interleaved_perm`. The tests check the relative-position
    property and the layout equivalence. Implement `position_interpolation_rope` and
    `ntk_aware_base`.

3.  **SwiGLU.** Implement `SwiGLU.forward`.

4.  **GQA.** Implement `GroupedQueryAttention.forward` with QK-norm, RoPE with cache offset, cache
    update storing only $G$ heads, and masked attention. The test checks prefill-then-decode against
    the library.

5.  **Model.** Implement `Qwen3Block.forward` and `Qwen3Model.forward`.

6.  **Memory planning.** Implement `kv_cache_bytes`, `count_params` (exact, tied embeddings counted
    once) and `max_concurrent_sequences`.

7.  **MoE.** Implement `route_topk`, `load_balancing_loss` and the sort-based
    `moe_dispatch_forward`, checked against `llmsys.moe.MoEFeedForward.forward_reference`.

**Measurements**

- With internet access, download `Qwen/Qwen3-0.6B` and run
  `run.py --impl starter --weights-dir qwen3-0.6b`, once with and once without `--thinking`. Record
  tokens/s on your device and compare with the napkin-math bound of [@sec:modern:model].

- Plot decode time per token versus context length (512 to 16k) with the materialized
  `repeat_interleave` and with `F.scaled_dot_product_attention(..., enable_gqa=True)` on a GPU.
  Expect the gap to widen with context.

- Time `MoEFeedForward.forward` vs. `forward_reference` for $E\in\{8,32,128\}$; record the
  tokens-per-expert histogram and the auxiliary loss for a random router.

**Deliverables**

Passing tests; your memory-planning table for the GPU you actually have; the decode-time plot with a
paragraph relating it to [@sec:modern:gqa]'s intensity argument.

**Stretch goals**

- Load `Qwen/Qwen3-4B` or `-8B` with the meta-device loader and report peak memory.

- Add a capacity factor to your dispatch and measure the fraction of dropped tokens for a trained
  vs. a random router.

- Fuse gate and up projections into a single $d\times2h$ `nn.Linear`, convert the checkpoint weights
  accordingly, and measure the speedup.
:::
