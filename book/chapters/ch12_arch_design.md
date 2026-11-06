# The Architecture Design Space: Tensor Programs, Choices, and Ablations {#ch:archdesign}

[@ch:gpt] built GPT-2 and [@ch:modern] built Qwen3. Between those two models sit a few
dozen design decisions: where the normalisation goes, which nonlinearity the MLP uses, how position
is encoded, how wide the MLP is relative to the residual stream, whether linear layers carry biases,
how weights are initialised, and which small tricks keep a large run from blowing up. None of them
changes what a transformer *is*. Together they decide whether a 7B-parameter run trains smoothly at
a high learning rate or wastes a week of cluster time on loss spikes.

This chapter maps that design space. For each decision it gives the options, the mechanism that
makes one option better, the evidence the field has accumulated, and the cost in FLOPs, bytes and
kernels. It also shows how to *run an ablation yourself*: a configurable model in
`llmsys/ablate.py`, a sweep driver, and a methodology for separating real effects from noise on a
small budget. Systems readers should watch for a recurring pattern: several "modelling" choices,
such as dropping biases, RMSNorm, gated MLP widths that are multiples of 64, and head dimensions of
64--128, are really systems choices made to suit the hardware.

![Roadmap of this chapter. The top row covers the components of a block, the bottom row the global
choices and the methodology for testing any of
them.](figures/ch12_arch_design-1.svg){#fig:archdesign:roadmap}

## Tensor programs with einsum and einops {#sec:archdesign:einsum}

Almost every operation in a transformer is either a *contraction* (a sum of products over some
index, i.e. a matrix multiply in disguise), a *rearrangement* of axes, or an elementwise function.
Index bookkeeping is where most implementation bugs hide: `x @ W.T` silently does the wrong thing if
`x` has its axes in the wrong order, and `transpose(-2, -1)` means nothing to a reader without the
shapes in mind. Einstein-summation notation fixes this by naming every axis.

##### The rule.

An einsum string lists the axes of each input and of the output. Axes that appear in an input but
not in the output are summed over; axes that appear in the output are kept; `...` stands for "any
leading axes, carried through unchanged". A linear layer applied to a batch of sequences is
$$\begin{equation}
y_{b,t,o} \;=\; \sum_{i} x_{b,t,i}\, W_{o,i}
\qquad\Longleftrightarrow\qquad
\texttt{einsum("...i,oi->...o", x, W)}.
\label{eq:archdesign:linear}
\end{equation}$$ The same notation expresses the attention score matrix, which contracts over the
head dimension and keeps batch, head, query and key positions:
$$S_{b,h,q,k} = \sum_d Q_{b,h,q,d}\,K_{b,h,k,d}
\quad\Longleftrightarrow\quad
\texttt{einsum("bhqd,bhkd->bhqk", Q, K)} .$$

![Reading `"bhqd,bhkd->bhqk"`. Axes shared by both inputs and kept in the output (b, h) become batch
dimensions of a batched GEMM; the axis that disappears (d) is the GEMM's inner dimension; q and k
become its rows and columns.](figures/ch12_arch_design-2.svg){#fig:archdesign:einsum}

##### einops.

The `einops` library adds two companions with the same naming discipline: `rearrange` for reshapes
and transposes, and `reduce` for reductions. The pattern `"b t (h d) -> b h t d"` splits the last
axis into heads and moves them forward, which replaces a `view` followed by a `transpose` whose
correctness depends on the reader remembering the layout.

``` {#lst:archdesign:mha-einops .python caption="Multi-head attention written with named axes (compare [@lst:archdesign:mha-torch])."}
from einops import rearrange, einsum

def mha(x, Wq, Wk, Wv, Wo, n_heads):
    # x: (b, t, d_model); W*: (d_out, d_in)
    q = einsum(x, Wq, "b t i, o i -> b t o")
    k = einsum(x, Wk, "b t i, o i -> b t o")
    v = einsum(x, Wv, "b t i, o i -> b t o")
    q, k, v = (rearrange(z, "b t (h d) -> b h t d", h=n_heads) for z in (q, k, v))
    s = einsum(q, k, "b h q d, b h k d -> b h q k") / q.shape[-1] ** 0.5
    s = s.masked_fill(causal_mask(s.shape[-1]), float("-inf"))
    o = einsum(s.softmax(-1), v, "b h q k, b h k d -> b h q d")
    o = rearrange(o, "b h t d -> b t (h d)")
    return einsum(o, Wo, "b t i, o i -> b t o")
```

``` {#lst:archdesign:mha-torch .python caption="The same computation with positional axes, as in `llmsys/layers.py`."}
def mha_torch(x, Wq, Wk, Wv, Wo, n_heads):
    B, T, D = x.shape
    split = lambda z: z.view(B, T, n_heads, D // n_heads).transpose(1, 2)
    q, k, v = split(x @ Wq.T), split(x @ Wk.T), split(x @ Wv.T)
    s = (q @ k.transpose(-2, -1)) / q.shape[-1] ** 0.5
    s = s.masked_fill(causal_mask(T), float("-inf"))
    o = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D)
    return o @ Wo.T
```

##### Memory order.

PyTorch tensors are row-major with explicit strides ([@ch:pytorch]). `transpose` and
`rearrange` patterns that only permute axes return *views* with permuted strides and copy nothing;
the copy happens later, when an operation needs contiguous memory (`reshape` on a non-contiguous
view, or a kernel that requires it). That is why [@lst:archdesign:mha-torch] uses
`reshape` rather than `view` after the transpose. Weights are stored as
$(d_\text{out}, d_\text{in})$, the convention of `nn.Linear`, so a forward pass computes
$\mathbf{x}\mathbf{W}^\top$: each output feature is a contiguous row of $\mathbf{W}$, which is also
the layout that tensor-parallel column splits shard along ([@ch:distributed]).

::: systemslens
[Einsum is a front end, not a kernel]{.box-title}

`torch.einsum` parses the string and lowers it to permutes, reshapes and a batched GEMM (`bmm`), so
it runs at the same speed as the hand-written version. Two things can still go wrong. First, a
permute that makes an operand non-contiguous can trigger a hidden copy; profile ([@ch:perf])
rather than guess. Second, with three or more operands the *contraction order* matters:
$(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ costs $O(T^2 d)$ while
$\mathbf{Q}(\mathbf{K}^\top\mathbf{V})$ costs $O(T d^2)$. PyTorch picks an order with `opt_einsum`
when it is installed, but for attention the softmax sits between the two products and prevents
reordering. Removing that softmax to unlock the cheap order is exactly the idea behind linear
attention ([@ch:attnalt]).
:::

## Layers from scratch: Linear, Embedding, and initialisation {#sec:archdesign:init}

[@lst:archdesign:linear] shows the two parameterised building blocks used by the ablation
model in `llmsys/ablate.py`, written without `nn.Linear` or `nn.Embedding` so that the storage
layout and the initialisation are explicit.

``` {#lst:archdesign:linear .python caption="Linear and Embedding layers from scratch (`llmsys/ablate.py`)."}
class Linear(nn.Module):
    def __init__(self, d_in, d_out, bias=False, std=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        std = std if std is not None else math.sqrt(2.0 / (d_in + d_out))
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None

    def forward(self, x):                       # any leading axes
        y = torch.einsum("...i,oi->...o", x, self.weight)
        return y if self.bias is None else y + self.bias

class Embedding(nn.Module):
    def __init__(self, num, d, std=0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num, d))
        nn.init.trunc_normal_(self.weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, ids):                     # a gather, not a matmul
        return self.weight[ids]
```

##### Embeddings are gathers.

Mathematically an embedding is a one-hot vector times a matrix, but nobody computes it that way:
indexing reads $d$ numbers per token, while a one-hot matmul would read the whole $V\times d$ table.
The backward pass is a scatter-add into the rows that were used, so the gradient is *row-sparse*.
Optimisers still update every row, because AdamW's moments decay everywhere, which is one reason
large-vocabulary embeddings are a noticeable share of optimiser time.

##### Why initialisation matters.

Consider $y=\mathbf{W}x$ with $x$ having i.i.d. entries of variance $\sigma_x^2$ and $\mathbf{W}$
having i.i.d. entries of variance $\sigma_w^2$. Each output is a sum of $d_\text{in}$ independent
products, so $$\begin{equation}
\operatorname{Var}(y_o) = d_\text{in}\,\sigma_w^2\,\sigma_x^2 .
\label{eq:archdesign:var}
\end{equation}$$ Preserving the forward variance needs $\sigma_w^2 = 1/d_\text{in}$ (LeCun);
preserving the backward variance, where the roles swap, needs $1/d_\text{out}$. Glorot and Bengio
(2010) compromise with $\sigma_w^2 = 2/(d_\text{in}+d_\text{out})$, which is the default in
[@lst:archdesign:linear]. ReLU-family activations zero half the signal, which motivates He
initialisation, $2/d_\text{in}$. Drawing from a normal distribution *truncated* at $\pm3\sigma$
removes the rare large weights that would otherwise create outlier features at initialisation; the
truncation shrinks the realised standard deviation by about 1.3%, which is harmless.

##### Residual streams add variance.

A residual network adds each block's output to the stream, $x_{\ell+1}=x_\ell+f_\ell(x_\ell)$. If
each branch contributes variance comparable to the stream, the stream's variance grows with depth.
Pre-norm blocks normalise the branch *input*, so each branch adds a roughly constant amount and the
stream's RMS grows like $\sqrt{\ell}$. GPT-2 counteracts this by scaling the initial weights of each
branch's output projection by $1/\sqrt{2L}$ (two branches per block), which keeps the stream at unit
scale at initialisation. Without any normalisation the multiplicative coupling compounds and the
stream grows exponentially. [@fig:archdesign:depth] shows all three regimes, measured on a
random 48-block residual MLP stack.

![Growth of the residual stream with depth at initialisation, for random inputs passed through 48
residual blocks $x\leftarrow x+\mathbf{W}_2\,\mathrm{ReLU}(\mathbf{W}_1 N(x))$ with $d=256$
(measured on CPU in this environment). With no norm the growth is exponential; pre-norm makes it
$\propto\sqrt{\ell}$; depth-scaled output projections keep it
flat.](figures/ch12_arch_design-3.svg){#fig:archdesign:depth}

::: napkinmath
[The loss at initialisation]{.box-title}

A correctly initialised language model should start out predicting a nearly uniform distribution, so
its cross-entropy at step 0 is close to $\ln V$: 10.82 for GPT-2's $V=50{,}257$, 11.93 for Qwen3's
$V=151{,}936$, and 5.55 for a byte-level model with $V=256$. If the logits have a small random
spread with standard deviation $\sigma_z$, the expected loss is about $\ln V+\sigma_z^2/2$. With the
final norm producing unit-RMS features of width $d$ and a head initialised with standard deviation
$s$, $\sigma_z\approx s\sqrt d$. For the $d=64$, $V=256$ ablation model with fan-average init,
$s=\sqrt{2/320}=0.079$, so $\sigma_z\approx0.63$ and the initial loss is about $5.55+0.2=5.75$,
which is what the model reports. An initial loss of 30 means the init is broken, not that the model
is "hard". Checking this number is the first step of every debugging session
([@sec:archdesign:ablate]).
:::

::: deepdive
[Width-aware parameterisations]{.box-title}

Fan-in initialisation makes activations the right size at step 0, but it does not make the *updates*
the right size as width grows: with Adam, a weight update has a roughly width-independent per-entry
magnitude, so its effect on activations grows with width. The maximal update parameterisation
($\mu$P; Yang et al., 2021) rescales initialisation, learning rates and output multipliers per layer
type so that the optimal learning rate transfers from a narrow proxy to a wide model.
[@ch:scalinglaws] uses this to tune hyperparameters cheaply.
:::

## Normalisation: what to normalise, and where {#sec:archdesign:norm}

[@ch:gpt] implemented LayerNorm and [@sec:modern:rmsnorm] implemented RMSNorm. Here
we look at the two decisions that matter more than the formula: *where* the norm sits in the block,
and whether there is one at all.

##### Placement.

[@fig:archdesign:normplace] shows the four placements in use.

- **Post-norm** (the original transformer, BERT): $x\leftarrow N(x+f(x))$. Every residual path
  passes through a norm, so the identity path is not an identity. Xiong et al. (2020) showed that
  this makes gradients near the output large relative to those of early layers at initialisation,
  which is why post-norm models need a learning-rate warmup and are fragile at depth.

- **Pre-norm** (GPT-2 onwards, nearly every current LLM): $x\leftarrow x+f(N(x))$. The residual path
  is a clean identity from input to output, gradients flow to every layer unattenuated, and training
  tolerates larger learning rates. The price is the $\sqrt{L}$ stream growth of
  [@fig:archdesign:depth] and the need for a final norm before the LM head.

- **Sandwich** ("double norm" or peri-norm): $x\leftarrow x+N_b(f(N_a(x)))$. A second norm on the
  branch output bounds what each block can add to the stream. Gemma 2 normalises both before and
  after each sublayer, and OLMo 2 moved its norms onto the branch outputs; both are reported to
  improve stability. The norm still stays *off* the residual path itself.

- **Parallel**: $x\leftarrow x+\mathrm{Attn}(N(x))+\mathrm{MLP}(N(x))$ (GPT-J, PaLM). One norm
  serves both branches, and the first projections of attention and MLP can be fused into one GEMM;
  [@sec:archdesign:block] looks at the trade-off.

![Where the norm goes, for one sublayer $f$ (attention or MLP); time flows upwards. Only post-norm
puts a normalisation *on* the residual
path.](figures/ch12_arch_design-4.svg){#fig:archdesign:normplace}

##### No norm at all.

Removing normalisation (Lab 7, "no norm") leaves the model correct but unstable: without the
per-token rescaling, the stream grows as in [@fig:archdesign:depth], the attention logits
grow with it, and the optimiser has to use a much smaller learning rate. Careful initialisation
schemes (Fixup, T-Fixup) and per-branch scalars can make norm-free transformers train, but nobody
trains large LLMs that way.

##### LayerNorm, RMSNorm, and dropping biases.

RMSNorm skips the mean subtraction and the bias. In quality it matches LayerNorm in essentially
every published comparison. The common justification is "fewer FLOPs", which is true but misleading,
because norms are a vanishing fraction of a transformer's FLOPs. The real gain is bytes moved:
normalisation and other elementwise operators are memory-bound, so their share of runtime is far
larger than their share of FLOPs (Ivanov et al., 2021). The same logic, plus some evidence of better
stability, explains why most modern models also drop the biases from linear layers.

::: napkinmath
[A norm is cheap in FLOPs and expensive in bytes]{.box-title}

Take one RMSNorm over a bf16 activation of $8192$ tokens $\times\,4096$ features on an H100. It
reads and writes $2\times 8192\times4096\times2\,\text{B}\approx134$ MB and performs about 4 FLOPs
per element, roughly $1.3\times10^8$ FLOPs. At 3.35 TB/s the memory traffic takes about 40 $\mu$s;
at 989 TFLOP/s the arithmetic would take 0.14 $\mu$s. The arithmetic intensity is about 1 FLOP/byte,
nearly 300$\times$ below the H100's ridge point. For comparison, the $4096\times4096$ projection
that follows does $2\times8192\times4096^2\approx2.7\times10^{11}$ FLOPs in about 280 $\mu$s at
peak. So an unfused norm adds roughly 15% to the time of the matmul next to it, despite contributing
0.05% of the FLOPs. Fusing the norm into the neighbouring kernel ([@ch:perf],
[@app:triton]) removes most of that cost.
:::

## MLPs, activations, and gating {#sec:archdesign:mlp}

The position-wise MLP holds about two thirds of a dense transformer's parameters, so its form
matters. The classic form is $\mathrm{FFN}(x)=\mathbf{W}_2\,\sigma(\mathbf{W}_1x)$ with
$d_\text{ff}=4d$.

##### Activations.

ReLU, $\max(0,z)$, was the original choice. GELU, $z\,\Phi(z)$, is a smooth version that weights
each input by the probability that it is positive under a standard normal; GPT-2 uses its tanh
approximation ([@ch:gpt]). SiLU (Swish), $z\,\mathrm{sigmoid}(z)$, has the same shape with a
slightly deeper negative dip. Squared ReLU, $\max(0,z)^2$, appears in a few models (Primer;
Nemotron-4 340B is reported to use it). [@fig:archdesign:acts] plots them.

![The common MLP nonlinearities. GELU and SiLU are smooth and non-monotonic near zero; the gated
variants of [@eq:archdesign:glu] multiply one of these curves by a second linear
projection.](figures/ch12_arch_design-5.svg){#fig:archdesign:acts}

##### Gated linear units.

A GLU MLP replaces $\sigma(\mathbf{W}_1x)$ with the elementwise product of an activated projection
and a linear one: $$\begin{equation}
\mathrm{FFN}_\text{GLU}(x) = \mathbf{W}_2\big(\sigma(\mathbf{W}_g x)\odot \mathbf{W}_1 x\big),
\label{eq:archdesign:glu}
\end{equation}$$ which gives ReGLU ($\sigma=$ ReLU), GeGLU (GELU) and SwiGLU (SiLU). The extra
matrix costs parameters, so GLU MLPs use $d_\text{ff}\approx\frac{8}{3}d$, which makes
$3\cdot\frac83 d^2 = 8d^2$ equal to the $2\cdot4d^2$ of a plain MLP. Shazeer (2020) found that GLU
variants consistently improved perplexity at matched parameters, and Narang et al. (2021) found GLUs
among the few modifications whose gains transferred across settings. Almost every LLM released since
2023 uses SwiGLU or GeGLU.

``` {#lst:archdesign:mlp .python caption="A parameter-matched MLP family (`llmsys/ablate.py`)."}
GATED = {"swiglu", "geglu", "reglu"}

def ffn_hidden_dim(d_model, kind, multiple_of=64):
    raw = 8 * d_model / 3 if kind in GATED else 4 * d_model
    return int(multiple_of * math.ceil(raw / multiple_of))   # tensor-core friendly

class MLP(nn.Module):
    def __init__(self, cfg, out_std):
        super().__init__()
        d, h = cfg.d_model, cfg.ff_dim()
        self.gated = cfg.ffn in GATED
        self.act = {"swiglu": F.silu, "geglu": F.gelu, "reglu": F.relu}.get(cfg.ffn) \
            or ACTS[cfg.ffn]
        self.w_in = Linear(d, h)
        self.w_gate = Linear(d, h) if self.gated else None
        self.w_out = Linear(h, d, std=out_std)        # depth-scaled init

    def forward(self, x):
        if self.gated:
            return self.w_out(self.act(self.w_gate(x)) * self.w_in(x))
        return self.w_out(self.act(self.w_in(x)))
```

::: systemslens
[What gating costs on the GPU]{.box-title}

A SwiGLU MLP at matched parameters does the same matmul FLOPs as a 4$d$ MLP, but in *three* GEMMs of
width $\frac83 d$ instead of two of width $4d$. Production implementations concatenate
$\mathbf{W}_g$ and $\mathbf{W}_1$ into one $(2d_\text{ff})\times d$ weight (the "`gate_up_proj`" in
many inference engines), so the hardware sees one wide GEMM, then a fused `silu(a) * b` elementwise
kernel. For the backward pass, a SwiGLU MLP saves three hidden-width tensors per token (gate
pre-activation, up projection, and their product), while a GELU MLP saves two (pre-activation and
activation). Measured in $d_\text{ff}$ that sounds worse, but at matched parameters
$3\cdot\frac83d=8d=2\cdot4d$, so the activation memory is the same; an unfused implementation,
however, also materialises $\mathrm{SiLU}(\boldsymbol{a})$ separately, which is exactly what fusion
removes ([@ch:perf] shows how to measure this). Finally, $\frac83 d$ is rarely a nice number.
Rounding it up to a multiple of 64 or 128 keeps every GEMM tile full; Qwen3-0.6B's
$d_\text{ff}=3072=3d$ is one such rounding choice.
:::

## Positional information {#sec:archdesign:pos}

Self-attention is permutation-equivariant: without extra information, swapping two input tokens just
swaps the corresponding outputs. Something has to tell the model where each token sits. Five
families are in use.

##### No positional encoding (NoPE).

A *causal* decoder is not actually permutation-invariant, because the mask itself breaks the
symmetry. A token at position $i$ attends over exactly $i+1$ keys, so quantities such as the
attention entropy, or the average of a constant value vector, already depend on $i$. Theory
(Kazemnejad et al., 2023) and experiments show that causal transformers trained without any explicit
encoding can learn both absolute and relative positional information. NoPE is also used deliberately
in modern hybrids, for example in the global-attention layers of some models that use RoPE only in
their local layers.

##### Absolute embeddings.

GPT-2 adds a learned vector $\boldsymbol{u}_i$ per position ([@ch:text]); the original
transformer added fixed sinusoids,
$$\mathrm{PE}_{i,2j}=\sin(i/10000^{2j/d}),\qquad \mathrm{PE}_{i,2j+1}=\cos(i/10000^{2j/d}).$$
Sinusoids have a relative structure: $\langle\mathrm{PE}_i,\mathrm{PE}_{i+\Delta}\rangle$ depends
only on $\Delta$. But once they are *added* to content embeddings, the query--key product
$\langle\boldsymbol{x}+\mathrm{PE}_i,\boldsymbol{y}+\mathrm{PE}_j\rangle$ contains cross terms such
as $\langle\boldsymbol{x},\mathrm{PE}_j\rangle$ that depend on absolute positions. Learned absolute
embeddings cannot extrapolate past the trained length at all, because those rows were never trained.

##### Relative biases and ALiBi.

T5 adds a learned scalar bias to each attention logit, indexed by a bucketed distance $i-j$. ALiBi
(Press et al., 2022) uses a fixed bias that grows linearly with distance,
$$s_{ij} \leftarrow s_{ij} - m_h\,(i-j),\qquad m_h = 2^{-8h/H},$$ with a geometric set of slopes so
that some heads are local and others nearly global. It adds no parameters and degrades gracefully
beyond the training length.

##### RoPE.

Rotary embeddings ([@sec:modern:rope]) satisfy the property we actually want: a scoring
function of the form
$\langle f(\boldsymbol{x},i), f(\boldsymbol{y},j)\rangle = g(\boldsymbol{x},\boldsymbol{y},i-j)$.
Rotating queries and keys by position-dependent angles achieves this exactly, because rotations
preserve inner products, and there are no additive cross terms. RoPE is the default for essentially
all current open LLMs; partial RoPE (rotating only some dimensions) and per-layer variations are
common refinements.

![Left: sinusoidal encodings use a geometric range of frequencies, so low-index channels change
quickly with position and high-index channels slowly. Right: ALiBi subtracts $m(i-j)$ from each
logit; darker cells are closer keys, which receive less penalty. Cells above the diagonal are
masked.](figures/ch12_arch_design-6.svg){#fig:archdesign:pos}

::: systemslens
[Positional schemes and the inference stack]{.box-title}

The choice of positional scheme reaches into the inference engine. Absolute embeddings are added
once at the input and never touch attention. RoPE is applied to $\boldsymbol{q}$ and
$\boldsymbol{k}$ *before* keys are cached, so the cache stores already-rotated keys: prefix caching
([@ch:serving]) stays valid only if a shared prefix sits at the same positions, and positions
shifted by left padding are harmless because only relative angles matter ([@ch:inference]).
ALiBi and T5-style biases must be added *inside* the attention kernel, which FlashAttention
supports, but each such bias is one more variant that kernels must specialise. NoPE costs nothing
anywhere, which is part of its appeal for long-context global layers.
:::

## Hyperparameters that (mostly) do not need tuning {#sec:archdesign:hparams}

A transformer's shape is fixed by a handful of numbers: $d$, $L$, $H$, $d_{h}$, $d_\text{ff}$ and
$V$. Across many published models these have settled into narrow ranges
([@tab:archdesign:shapes]).

::: {#tab:archdesign:shapes .table-float}
| Model       |   $d$ | $L$ | $H$ (KV) | $d_{h}$ | $d_\text{ff}$ | $d_\text{ff}/d$ | $d/L$ |
|:------------|------:|----:|---------:|--------:|--------------:|----------------:|------:|
| GPT-2 small |   768 |  12 |       12 |      64 |          3072 |             4.0 |    64 |
| GPT-2 XL    |  1600 |  48 |       25 |      64 |          6400 |             4.0 |    33 |
| GPT-3 175B  | 12288 |  96 |       96 |     128 |         49152 |             4.0 |   128 |
| Llama 2 70B |  8192 |  80 |   64 (8) |     128 |         28672 |             3.5 |   102 |
| Qwen3-0.6B  |  1024 |  28 |   16 (8) |     128 |          3072 |             3.0 |    37 |
| Qwen3-8B    |  4096 |  36 |   32 (8) |     128 |         12288 |             3.0 |   114 |
| Qwen3-32B   |  5120 |  64 |   64 (8) |     128 |         25600 |             5.0 |    80 |

: Shapes of some well-documented dense models. Qwen3 and Llama use gated MLPs, so the
parameter-matched ratio is $\frac83\approx2.67$; GPT-2 and GPT-3 use plain 4$d$ MLPs. In Qwen3-0.6B
and Qwen3-32B, $H\cdot d_{h}\neq d$: the attention width is decoupled from the residual width.
:::

##### MLP ratio.

$d_\text{ff}=4d$ for plain MLPs and about $\frac83 d$ to $3.5d$ for GLUs. Kaplan et al. (2020) found
loss to be flat across a wide range of ratios, and T5's 11B model used $d_\text{ff}=64d$ without
disaster. The default is a safe point in a broad basin, not a sharp optimum.

##### Heads and head dimension.

The convention $H\cdot d_{h}=d$ keeps attention's projections square, but nothing requires it, as
two of the Qwen3 configurations show. What matters for systems is $d_{h}$: the FlashAttention tile
budget in [@sec:attention:variants] makes 64--128 the efficient range on current GPUs, and
kernels are specialised per head dimension. The number of KV heads is an inference decision
([@sec:modern:gqa]).

##### Aspect ratio.

Wide-and-shallow and narrow-and-deep models of the same parameter count reach similar loss over a
broad range of $d/L$, roughly 50 to 200 (Kaplan et al., 2020). Systems considerations usually decide
the value: depth adds sequential latency per decoded token and more pipeline stages, while width
makes each GEMM larger and easier to split with tensor parallelism.

##### Vocabulary size.

Monolingual English models use 32k--50k tokens; multilingual and production models use 100k--260k
([@ch:text]). Larger vocabularies shorten sequences but enlarge the embedding table and the
output softmax, which dominate the parameter count of small models: in Qwen3-0.6B the tied
$151{,}936\times1024$ embedding is 156M of its roughly 600M parameters.

##### Regularisation.

Modern pretraining runs are single-epoch, so overfitting is not the concern. Most recent models use
no dropout and weight decay of about 0.1. Andriushchenko et al. (2023) argue that weight decay in
LLM training acts mainly on optimisation dynamics, through its interaction with the learning-rate
schedule, rather than as a regulariser. [@ch:optim] implements decoupled weight decay. In
code, "weight decay on matrices only" means two parameter groups. Norm gains, biases and (often)
embeddings are excluded, because shrinking a norm gain toward zero or a bias toward zero is not a
sensible prior:

``` {#lst:archdesign:wd .python caption="Parameter groups for decoupled weight decay."}
decay    = [p for n, p in model.named_parameters() if p.dim() >= 2]   # weight matrices
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]    # gains, biases
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                         {"params": no_decay, "weight_decay": 0.0}],
                        lr=3e-4, betas=(0.9, 0.95))
```

::: napkinmath
[Pad your shapes]{.box-title}

Tensor-core GEMMs work in tiles; on an H100 bf16 kernels are happiest when the matrix dimensions are
multiples of 64 or 128. GPT-2's $V=50{,}257$ is prime-ish, so the LM-head GEMM ($T\times d\times V$)
has a ragged last tile. Padding the vocabulary to $50{,}304=64\times786$ adds 47 unused rows (0.09%
more parameters) and makes every tile full. In small-model training, where the LM head is a large
share of the FLOPs, practitioners report noticeable speedups from this one change. The same
reasoning applies to $d_\text{ff}$ (round $\frac83d$ up to a multiple of 64) and to batch $\times$
sequence products.
:::

## Stability tricks {#sec:archdesign:stability}

Large runs fail in a characteristic way: the loss curve develops spikes, some of which recover and
some of which do not. Most of the fixes target the two softmaxes in the model, because exponentials
turn moderate growth in logits into overflow and saturated gradients.

##### Output softmax: the z-loss.

The cross-entropy depends only on logit *differences*, so nothing in the loss stops all logits from
drifting upward together. The log-partition function
$\log Z=\operatorname{logsumexp}(\boldsymbol{z})$ can wander far from zero, which in bf16 costs
precision and can eventually overflow. PaLM added an auxiliary term $$\begin{equation}
\mathcal{L} = \mathcal{L}_\text{CE} + \alpha\,(\log Z)^2,\qquad \alpha\approx10^{-4},
\label{eq:archdesign:zloss}
\end{equation}$$ whose gradient, $2\alpha\log Z\cdot\operatorname{softmax}(\boldsymbol{z})$, gently
pulls $\log Z$ back toward 0 without changing the predicted distribution. Several later open models
(Baichuan 2, DCLM, OLMo 2) also use it.

##### Attention softmax: QK-norm.

As training proceeds, attention logits $\boldsymbol{q}^\top\boldsymbol{k}/\sqrt{d_{h}}$ can grow
until the softmax becomes one-hot. Gradients through it then vanish, which is a known cause of
divergence (Dehghani et al., 2023). Normalising $\boldsymbol{q}$ and $\boldsymbol{k}$ per head
before the dot product bounds the logits by the learned norm scales. Qwen3 adopted QK-norm
([@sec:modern:gqa]), as did OLMo 2 and Gemma 3, among others.

##### Logit soft-capping.

Gemma 2 instead bounds logits smoothly with $c\tanh(z/c)$ ([@fig:archdesign:softcap]):
nearly the identity for $|z|\ll c$, saturating at $\pm c$. It caps attention logits at 50 and final
logits at 30. The tanh does not fit the standard fused attention kernels without modification, which
is one reason QK-norm has been the more popular route.

![Logit soft-capping. Ordinary logits pass through almost unchanged; runaway logits saturate
smoothly instead of overflowing, and gradients never become exactly
zero.](figures/ch12_arch_design-7.svg){#fig:archdesign:softcap}

``` {#lst:archdesign:zloss .python caption="Cross-entropy with z-loss, computed in fp32 (`llmsys/ablate.py`)."}
def lm_loss(logits, targets, z_loss=0.0):
    lf = logits.float().flatten(0, -2)                 # (N, V) in fp32
    lse = torch.logsumexp(lf, dim=-1)                  # log Z per position
    ce = (lse - lf.gather(-1, targets.reshape(-1, 1)).squeeze(-1)).mean()
    total = ce + z_loss * lse.pow(2).mean() if z_loss > 0 else ce
    return total, ce.detach(), lse.mean().detach()     # log the size of log Z too
```

##### Everything else.

The remaining items on the stability checklist are less glamorous:

- compute softmaxes, norms and the loss in fp32 even when matmuls run in bf16
  ([@ch:pretrain]);

- use depth-scaled initialisation ([@fig:archdesign:depth]) and no biases;

- use warmup, gradient clipping and $\beta_2=0.95$ rather than 0.999 for AdamW ([@ch:optim]);

- in MoE models, run the router in fp32 and add a router z-loss ([@ch:attnalt]);

- log the gradient norm, the mean $\log Z$, the maximum attention logit and the per-layer activation
  RMS. Each of these usually starts drifting several hundred steps before a spike, which makes them
  early warnings.

::: systemslens
[A spike costs more than the steps it wastes]{.box-title}

At scale a divergence is a systems event. The usual recovery is to roll back to a checkpoint from
before the spike, skip or reshuffle the data batches that triggered it, and sometimes lower the
learning rate. That requires frequent asynchronous checkpoints ([@ch:pretrain]),
deterministic data loading that can skip specific batches, and monitoring that alerts within
minutes. Several published training logs (OPT, BLOOM) describe dozens of manual restarts. Stability
tricks are cheap insurance against an expensive operational failure mode.
:::

## Block-level variants {#sec:archdesign:block}

##### Serial versus parallel blocks.

A parallel block computes attention and MLP from the same normalised input and adds both to the
stream. Because both branches start with a matmul of the same input, their weights can be
concatenated into one $(3d + 2d_\text{ff})\times d$ GEMM, which improves utilisation at small batch
sizes and lets tensor parallelism overlap one all-reduce for both branches. GPT-J, GPT-NeoX and PaLM
used parallel blocks, and PaLM reported about 15% faster training at large scale with a small
quality cost at smaller sizes. Most recent models have returned to serial blocks.

##### One block, every option.

[@lst:archdesign:block] shows how the ablation model implements all four placements and the
parallel form in one class. Keeping every variant behind a configuration field, rather than in a
separate file per experiment, is what makes a sweep a one-line change and keeps the variants from
silently diverging in unrelated details such as initialisation.

``` {#lst:archdesign:block .python caption="The configurable block of `llmsys/ablate.py`."}
class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        out_std = init_std(cfg.d_model, cfg.d_model, cfg.init)
        if cfg.depth_scaled_init:
            out_std /= math.sqrt(2 * cfg.n_layers)       # GPT-2 style residual scaling
        self.cfg = cfg
        self.attn, self.mlp = Attention(cfg, out_std), MLP(cfg, out_std)
        mk = lambda: make_norm(cfg.norm, cfg.d_model, cfg.bias)   # rms | layer | none
        self.n1, self.n2 = mk(), mk()
        if cfg.norm_position == "sandwich":
            self.n1b, self.n2b = mk(), mk()

    def forward(self, x, rope=None, alibi=None):
        c = self.cfg
        if c.parallel_block:                              # x + attn(N(x)) + mlp(N(x))
            h = self.n1(x)
            return x + self.attn(h, rope, alibi) + self.mlp(h)
        if c.norm_position == "pre":                      # x + f(N(x))
            x = x + self.attn(self.n1(x), rope, alibi)
            return x + self.mlp(self.n2(x))
        if c.norm_position == "post":                     # N(x + f(x))
            x = self.n1(x + self.attn(x, rope, alibi))
            return self.n2(x + self.mlp(x))
        x = x + self.n1b(self.attn(self.n1(x), rope, alibi))   # sandwich
        return x + self.n2b(self.mlp(self.n2(x)))
```

##### Biases.

Dropping all bias vectors removes parameters that the optimiser must still track in fp32 (master
weights plus two Adam moments), removes elementwise kernels or epilogues, and has been reported to
help stability. The cost in quality has not been measurable at LLM scale, with one common exception:
Qwen2 kept a bias on the QKV projections.

##### Tied embeddings.

Sharing the input embedding and the output head saves $Vd$ parameters, which is a large fraction for
small models (GPT-2, Qwen3 up to 4B) and negligible for large ones (Qwen3-8B and larger untie them).
Tying couples two roles with different gradient statistics, so large models prefer the flexibility.

## What recent open models chose {#sec:archdesign:survey}

[@tab:archdesign:survey] summarises the architectural choices of several widely studied
models, as reported in their papers and released configurations. The convergence is striking:
pre-norm (often with extra branch norms), RMSNorm, gated MLPs, RoPE and no biases make up the
"Llama-like" default, and new models mostly vary the attention pattern (GQA, local/global
interleaving, hybrids; [@ch:attnalt]) and the stability tricks.

::: {#tab:archdesign:survey .table-float}
| Model | Norm | MLP | Position | Notable extras |
|:---|:---|:---|:---|:---|
| GPT-2 | pre-LN, LayerNorm | GELU, 4$d$ | learned absolute | biases; tied embeddings |
| GPT-3 | pre-LN, LayerNorm | GELU, 4$d$ | learned absolute | alternating dense and locally banded sparse attention |
| PaLM | pre-norm, LayerNorm without bias | SwiGLU | RoPE | parallel blocks; MQA; z-loss; no biases |
| Llama 2 | pre-RMSNorm | SwiGLU | RoPE | GQA in the 70B model; no biases |
| Gemma 2 | RMSNorm before and after each sublayer | GeGLU | RoPE | logit soft-capping; alternating local/global attention |
| OLMo 2 | RMSNorm on branch outputs | SwiGLU | RoPE | QK-norm; z-loss |
| Qwen3 (dense) | pre-RMSNorm | SwiGLU | RoPE, $\theta=10^6$ | QK-norm; GQA; no biases |

: Architectural choices of some well-documented models, as reported by their authors. Details vary
across sizes within a family; always check the released config of the specific checkpoint.
:::

## Running ablations on a budget {#sec:archdesign:ablate}

Everything above is a claim made by somebody else. The only way to build real intuition, and to
settle questions for your own setting, is to run controlled experiments. Small-scale ablations are
cheap, but they are full of traps. This section covers the setup, the methodology and the debugging
habits, and then reports a sweep you can reproduce in a few CPU-minutes.

### The setup: one config, one model, one loop

The ablation model in `llmsys/ablate.py` exposes every choice in this chapter as a field of a
dataclass, so a variant is a one-line override and the full configuration is logged with the result:

``` {#lst:archdesign:config .python caption="Configuring and running ablations."}
from dataclasses import replace
from llmsys.ablate import ArchConfig, train_lm, make_corpus

base = ArchConfig(d_model=64, n_layers=4, n_heads=4, context_length=64,
                  norm="rms", norm_position="pre", pos="rope", ffn="swiglu")
ids = make_corpus(300_000)            # or: bytes of a real text file
variants = {
    "baseline": {},
    "post-norm": {"norm_position": "post"},
    "no norm": {"norm": "none"},
    "NoPE": {"pos": "nope"},
    "SiLU MLP (4d)": {"ffn": "silu"},        # parameter-matched to SwiGLU
}
results = [train_lm(replace(base, **ov), ids, steps=250, lr=3e-3, seed=s, name=n)
           for n, ov in variants.items() for s in (0, 1)]
```

`train_lm` is a deliberately complete loop: AdamW with $\beta_2=0.95$ and decoupled weight decay on
matrices only, linear warmup and cosine decay, gradient clipping, periodic validation on a held-out
slice, gradient-norm and $\log Z$ logging, and divergence detection that records a NaN as a *result*
rather than crashing the sweep. `labs/lab12_ablations/run.py` wraps it with a command-line
interface, saves JSON after every run so an interrupted sweep loses at most one run, and plots the
curves. On a GPU, point `--text` at a real corpus (for example a TinyStories dump) and scale the
shape up.

### Methodology

1.  **Change one thing.** Each variant differs from the baseline in exactly one field, and all
    variants share the data order (same seed for the data sampler) so that differences are not
    caused by which batches happened to be drawn.

2.  **Match what should be matched.** Compare at equal parameters (hence $4d$ for non-gated MLPs)
    and equal tokens. When a change alters FLOPs per token, also compare at equal FLOPs or equal
    wall-clock time.

3.  **Tune the learning rate per variant**, at least coarsely. Many "X is better than Y" results are
    really "X tolerates the learning rate that was tuned for X". Post-norm and no-norm models in
    particular need lower rates, so a single-LR comparison mostly measures LR sensitivity. Report
    the LR sweep too.

4.  **Know your noise floor.** Run the baseline with two or three seeds. Differences smaller than
    the seed-to-seed spread are not results.

5.  **Look at curves, not just endpoints.** A variant that starts slowly may cross over later, and
    early curves can mislead in both directions.

6.  **Be humble about transfer.** Narang et al. (2021) re-tested dozens of published transformer
    modifications and found that most did not transfer across codebases and scales. A small-scale
    win is a hypothesis to test at the next scale up, not a conclusion.

::: pitfall
[Debugging before ablating]{.box-title}

Ablations on a broken model measure nothing. Before any sweep, check, in order: (1) the loss at step
0 is close to $\ln V$ ([@sec:archdesign:init]); (2) the model can *overfit a single batch*
to near-zero loss within a few hundred steps, since if it cannot, there is a bug in the model, the
loss or the optimiser; (3) tensor shapes match your expectations at every boundary (assert them, or
name them with einops); (4) the norms of activations, weights and gradients are finite and roughly
constant across layers. Step through the forward pass in a debugger rather than adding prints. Most
"my architecture change hurt" results at small scale turn out to be a mask, a transpose or an init
bug.
:::

### From a CPU sweep to a GPU sweep

The same code scales to real ablations; only the data, the shape and four flags change.

1.  **Data.** Tokenise the corpus once, offline, into a flat `uint16` file ([@ch:text]); a
    random-window sampler over a `np.memmap` costs nothing per step. Hold out validation
    *documents*, not random windows, so that validation text is never seen in training. The
    byte-level path in `run.py --text` skips tokenisation entirely, which is convenient for a first
    GPU sweep.

2.  **Run layout.** Give every run a directory named after its config hash and seed, write the full
    config and the git commit hash there before training starts, append metrics as JSON lines, and
    checkpoint periodically ([@ch:optim] builds this infrastructure). A sweep is then a list
    of configs plus a launcher that skips directories that already contain a finished marker, so
    re-running the sweep after a crash resumes it.

3.  **Speed.** Pass `--sdpa` for the fused attention kernel, `--amp` for bf16 autocast and
    `--compile` for `torch.compile`; the loss, softmax and norms stay in fp32 inside the model
    regardless ([@lst:archdesign:zloss]).

4.  **Budget.** Decide the sweep size from a napkin estimate before launching, as below.

``` bash
python labs/lab12_ablations/run.py --text tinystories_train.txt --device cuda \
    --sdpa --amp --compile --d-model 512 --layers 8 --heads 8 --ctx 256 \
    --batch 128 --steps 10000 --lr 1e-3 --seeds 0 1 --out runs/abl_ts.json
```

::: napkinmath
[Budgeting an ablation sweep]{.box-title}

A $d=512$, $L=8$ model has about $12Ld^2\approx25$M non-embedding parameters. With batch 128 and
context 256, one step is 32,768 tokens and needs about
$6\times25\times10^6\times32{,}768\approx4.9\times10^{12}$ FLOPs. Models this small rarely exceed
20--30% MFU on an H100 (kernel launches and the LM head dominate), so plan on about 200 TFLOP/s, or
25 ms per step: 10,000 steps (330M tokens) take roughly 4 minutes. A sweep of 10 variants $\times$ 3
seeds $\times$ 4 learning rates is 120 runs, about 8 GPU-hours. Cutting the LR sweep to the three
variants that are most LR-sensitive (no norm, post-norm, and the baseline) brings it under 4 hours.
Writing this estimate down *before* launching is what stops a sweep from quietly consuming a week.
:::

### A worked sweep

We ran the standard variants on the synthetic corpus of `make_corpus` with a 4-layer, $d=64$ model
(about 234k parameters), context 64, batch 16, 250 steps at learning rate $3\times10^{-3}$, with
three seeds for the baseline and the two clearest losers and one seed elsewhere. Everything ran on a
shared CPU in this environment, so the absolute numbers are tiny-scale and only the comparisons
matter. [@fig:archdesign:curves] shows validation curves,
[@tab:archdesign:results] the final losses and [@fig:archdesign:lrsweep] a
learning-rate sweep for three norm settings.

![Validation loss during training for six architecture variants (seed 0; 4 layers, $d=64$,
byte-level synthetic corpus; measured on CPU in this environment). The positional variants separate
early and stay separated; norm placement makes no difference at this depth; removing the norm costs
about 0.23 nats.](figures/ch12_arch_design-8.svg){#fig:archdesign:curves}

::: {#tab:archdesign:results .table-float}
| Variant                             | seeds | final val. loss | $\Delta$ vs. baseline | parameters |
|:------------------------------------|------:|----------------:|----------------------:|-----------:|
| baseline: pre-RMSNorm, RoPE, SwiGLU |     3 |           0.922 |                   --- |    234,048 |
| post-norm                           |     1 |           0.927 |                -0.001 |    233,984 |
| sandwich norm                       |     1 |           0.929 |                +0.001 |    234,560 |
| no norm                             |     3 |           1.146 |                +0.223 |    233,472 |
| learned absolute positions          |     1 |           1.028 |                +0.100 |    238,144 |
| ALiBi                               |     1 |           0.960 |                +0.031 |    234,048 |
| NoPE                                |     3 |           1.333 |                +0.410 |    234,048 |
| SiLU MLP ($4d$)                     |     1 |           0.951 |                +0.023 |    229,952 |
| GELU MLP ($4d$)                     |     1 |           0.944 |                +0.016 |    229,952 |
| QK-norm + z-loss ($10^{-4}$)        |     1 |           0.933 |                +0.005 |    234,176 |

: Final validation loss after 250 steps at learning rate $3\times10^{-3}$ (mean over seeds; $\Delta$
is the paired difference to the baseline run with the same seed, i.e. the same initialisation and
data order). The baseline's seed-to-seed standard deviation is 0.005, so differences below about
0.01 are noise.
:::

![Learning-rate sweeps (seed 0, 250 steps). The "no norm" model is not hopeless; it needs a
different learning rate and still ends worse at the best one. A single-LR comparison at
$3\times10^{-3}$ would overstate its gap.](figures/ch12_arch_design-9.svg){#fig:archdesign:lrsweep}

What the sweep supports, with mechanisms:

- **Position matters most here.** NoPE is about 0.4 nats worse and learned absolute embeddings about
  0.1 worse than RoPE. The corpus contains counting runs and copy lines, which need to know *how far
  back* something is, which is exactly the relative information RoPE supplies directly and NoPE must
  infer from the causal mask ([@sec:archdesign:pos]). ALiBi, with its recency bias, sits in
  between.

- **Gating helps a little, beyond noise.** The parameter-matched SiLU and GELU MLPs end about 0.02
  worse than SwiGLU, consistent with the direction (though not necessarily the size) of the
  published results.

- **Norm placement does not matter at depth 4.** Post-norm and sandwich match pre-norm within noise.
  The post-norm pathologies of [@sec:archdesign:norm] grow with depth and learning rate.
  Our 4-layer model has neither, which is a reminder that small-scale ablations can miss effects
  that only appear at scale (exercise: rerun with $L=24$).

- **No norm is mainly an LR-sensitivity story.** At $3\times10^{-3}$ it is about 0.22 worse; at its
  own best learning rate the gap shrinks to about 0.04. Report sweeps, not single points.

- **Stability tricks are neutral when nothing is unstable.** QK-norm plus z-loss changes the loss by
  less than the noise. Their value appears at high learning rates and long runs, which is a
  different experiment.

## Summary {#sec:archdesign:summary}

- Einsum and einops name every axis; they lower to the same batched GEMMs as hand-written code, but
  remove whole classes of transpose bugs. Weights are stored $(d_\text{out},d_\text{in})$.

- Initialisation is variance bookkeeping: fan-average truncated normals for matrices, small
  embeddings, and $1/\sqrt{2L}$ scaling of residual output projections. The loss at step 0 should be
  about $\ln V$.

- Pre-norm keeps the residual path an identity and is why deep transformers train at high learning
  rates; extra norms on branch outputs are a common refinement. RMSNorm and bias-free layers win
  mainly by moving fewer bytes.

- Gated MLPs (SwiGLU, GeGLU) at $d_\text{ff}\approx\frac83 d$ are the consensus; implement them as
  one fused gate-up GEMM.

- RoPE dominates positional encoding. NoPE works in causal models, and ALiBi and relative biases
  live inside the attention kernel.

- Shapes sit in broad basins ($d_\text{ff}/d\approx 3$--4, $d/L\approx 50$--200, $d_{h}=64$--128);
  systems constraints usually decide the exact values.

- Stability tricks target softmaxes: z-loss for the output and QK-norm or soft-capping for
  attention, plus fp32 reductions and monitoring.

- Ablations need one change at a time, matched parameters and tokens, per-variant learning rates, a
  noise floor, and a debugged baseline.

## Exercises {#sec:archdesign:exercises}

::: exercise
[einsum fluency]{.box-title}

Write einsum strings for: (a) the GQA score computation where queries have shape $(b,g,r,q,d)$ with
$g$ KV groups and $r$ queries per group, and keys $(b,g,k,d)$; (b) a batched outer product
$(b,i),(b,j)\to(b,i,j)$; (c) the trace of each matrix in a batch $(b,n,n)$; (d) the sum over heads
of per-head output projections, given $\mathbf{O}\in\mathbb{R}^{b\times h\times t\times d_h}$ and
$\mathbf{W}\in\mathbb{R}^{h\times d_h\times D}$.
:::

::: exercise
[Contraction order]{.box-title}

For $\mathbf{A}\in\mathbb{R}^{n\times d}$, $\mathbf{B}\in\mathbb{R}^{d\times n}$,
$\mathbf{C}\in\mathbb{R}^{n\times d}$, count the FLOPs of $(\mathbf{A}\mathbf{B})\mathbf{C}$ and
$\mathbf{A}(\mathbf{B}\mathbf{C})$. For $d=128$, at what $n$ does the second order become cheaper,
and by how much at $n=32{,}768$? Which nonlinearity in attention prevents using the cheaper order?
:::

::: exercise
[Variance propagation]{.box-title}

Derive the variance of the output of a 2-layer ReLU MLP
$y=\mathbf{W}_2\,\mathrm{ReLU}(\mathbf{W}_1x)$ under fan-average initialisation, for
$x\sim\mathcal{N}(0,\mathbf{I}_d)$ and $d_\text{ff}=4d$. What scale on $\mathbf{W}_2$ keeps
$\operatorname{Var}(y)=\operatorname{Var}(x)$?
:::

::: exercise
[Loss at init]{.box-title}

Predict the initial loss of (a) Qwen3-0.6B with a head initialised with standard deviation 0.02 and
unit-RMS final features; (b) the same model with the head initialised at standard deviation 1.
Verify (a) with the core `Qwen3Model` on random tokens (no download needed).
:::

::: exercise
[Parameter matching]{.box-title}

Show that a GLU MLP with $d_\text{ff}=\frac83d$ has the same parameter count and the same matmul
FLOPs per token as a plain MLP with $4d$. How many bytes of activations per token does each store
for the backward pass, in bf16, if only GEMM inputs and activation inputs are saved?
:::

::: exercise
[NoPE and the causal mask]{.box-title}

Consider a single causal attention layer with all keys equal (so scores are uniform). Show that the
output at position $i$ is the mean of $v_0,\ldots,v_i$. Construct value vectors that let the layer
output a quantity strictly decreasing in $i$, demonstrating how position can be inferred without any
encoding.
:::

::: exercise
[z-loss gradient]{.box-title}

Compute $\partial(\log Z)^2/\partial z_j$. Decompose this gradient into a component along the
all-ones vector (which shifts every logit equally, changing $\log Z$ but not the predicted
distribution) and an orthogonal component. Show that the cross-entropy gradient has no all-ones
component, and explain why, for small $\alpha$, the z-loss controls $\log Z$ while barely perturbing
the predictions.
:::

::: exercise
[Norm fusion]{.box-title}

Using the napkin model of [@sec:archdesign:norm], estimate for Qwen3-8B on an H100 the
fraction of a transformer block's forward time spent in unfused RMSNorms at 8192 tokens. Repeat for
a fused implementation that reads the input once and writes once inside the next GEMM's prologue.
:::

::: exercise
[Aspect ratio and latency]{.box-title}

Two 1.3B-parameter models have $d/L=32$ and $d/L=128$. Ignoring embeddings, find $d$ and $L$ for
each (with $d_\text{ff}=4d$). Under a memory-bound decode where each layer adds a fixed 15 $\mu$s of
kernel launch and synchronisation overhead plus bandwidth time, which model decodes faster on one
GPU, and which one splits more evenly over 4-way tensor parallelism?
:::

::: exercise
[Design your own ablation]{.box-title}

Pick one modification not in `STANDARD_ABLATIONS`, such as sandwich norm, parallel blocks, ALiBi,
QK-norm without z-loss or tied embeddings. Write down a hypothesis and the evidence that would
refute it, run it with two seeds and a 3-point LR sweep, and report a result with error bars.
:::

## Lab 8: An architecture ablation workbench {#sec:archdesign:lab}

::: labbox
[An architecture ablation workbench]{.box-title}

**Code:** `code/labs/lab12_ablations`

**Goals**

Implement the small components whose details decide whether a transformer trains well, verify each
one against a reference, and then use the configurable model in `llmsys/ablate.py` to run and
analyse a small ablation study.

**Setup**

``` bash
cd code
LAB_IMPL=starter  python -m pytest -q labs/lab12_ablations   # your work
LAB_IMPL=solution python -m pytest -q labs/lab12_ablations   # reference
pip install einops        # optional: to try the einops versions in the chapter
```

**Tasks**

1.  **Einsum plumbing.** `einsum_linear`, `split_heads`, `merge_heads`: write them with
    `torch.einsum`, then again with einops, and check both against `F.linear` and against a
    view-and-transpose implementation.

2.  **Initialisation.** `fan_avg_std` and `trunc_normal`, implemented by *resampling* out-of-range
    draws rather than calling `nn.init.trunc_normal_`. Verify that the realised standard deviation
    is about 98.7% of the target.

3.  **MLP width.** `ffn_hidden_dim` for gated and plain MLPs with rounding to a multiple.

4.  **Positions.** `sinusoidal_positions` (check the relative-dot-product property in the test) and
    `alibi_bias`.

5.  **Norm placement.** `block_forward` for pre, post, sandwich and parallel blocks, given arbitrary
    callables.

6.  **Stability.** `z_loss_cross_entropy` (numerically stable for logits of $\pm10^4$) and
    `softcap`.

7.  **Accounting.** `parameter_count(cfg)`: derive by hand the exact parameter count of
    `ArchLM(cfg)` for every option combination. The test compares it against the module.

8.  **Reporting.** `summarize_ablation`: average final losses across seeds, count divergences, and
    compute deltas against the baseline.

**Measurements**

- Run `run.py --steps 400 --seeds 0 1` on CPU (a few minutes per variant) and plot the curves. Is
  any difference larger than your seed spread?

- For pre-norm, post-norm and no-norm, sweep the learning rate over
  $\{10^{-3},3\cdot10^{-3},10^{-2},
    3\cdot10^{-2}\}$ and plot final loss against learning rate. Expect the normalised models to
  tolerate higher rates.

- On a GPU, rerun the sweep on a real corpus with $d=512$, $L=8$ for about 5k steps. Which
  conclusions from the CPU sweep survive?

**Deliverables**

Passing tests; a learning-curve figure and an LR-sweep figure; a short table from
`summarize_ablation`; and three sentences per variant explaining the result with the mechanisms in
this chapter.

**Stretch goals**

Add a `"peri"` placement that normalises only branch outputs (OLMo 2 style); add per-layer logging
of the maximum attention logit and show that QK-norm bounds it; reproduce
[@fig:archdesign:depth] for the actual `ArchLM` at initialisation for each norm placement.
:::
