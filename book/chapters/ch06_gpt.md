# Implementing a GPT Model {#ch:gpt}

In [@ch:text] we turned raw text into tensors of token IDs and embeddings, and in
[@ch:attention] we built the causal multi-head attention module that lets every token look
back at its predecessors. Those are the two hardest ingredients. This chapter assembles them,
together with a handful of small but essential components (layer normalization, a position-wise
feed-forward network, residual connections, dropout), into a complete GPT-2-style language model
that maps a batch of token IDs to next-token logits, and then uses that model to generate text.

For a systems person this chapter has a second, equally important goal: *accounting*. Once the model
exists as code we can say *exactly* how many parameters it has, how many floating-point operations a
forward pass costs, how many bytes of activations the backward pass will need, and which kernels
dominate the runtime. Every later systems decision in this book---how big a batch fits on a GPU,
when decoding becomes memory bound, how to shard a model across devices, how to size a KV
cache---rests on the arithmetic we derive here. We will check that arithmetic against PyTorch
itself, using `FlopCounterMode` and autograd's saved-tensor hooks, so that nothing is taken on
faith.

![Roadmap of this chapter. We first fix the overall skeleton and configuration, build the three
missing components, compose them into a transformer block and then a full model, generate text with
it, and finally account for every parameter, FLOP and
byte.](figures/ch06_gpt-1.svg){#fig:gpt:roadmap}

## Coding an LLM architecture {#sec:gpt:arch}

A GPT (*generative pretrained transformer*) is a decoder-only transformer: a stack of identical
blocks, each combining causal self-attention with a small per-token neural network, sandwiched
between an embedding layer at the bottom and a linear "unembedding" layer at the top. The original
GPT-2 release (Radford et al., 2019) came in four sizes. We will build the smallest, usually called
the 124M model, but every line of code is size-agnostic: the four variants differ only in three
numbers.

::: {#tab:gpt:sizes .table-float}
| Variant      | $d_{\text{model}}$ | Layers $L$ | Heads $H$ | Params (tied) | Params (untied, no QKV bias) |
|:-------------|-------------------:|-----------:|----------:|--------------:|-----------------------------:|
| GPT-2 small  |                768 |         12 |        12 |     124439808 |                    163009536 |
| GPT-2 medium |               1024 |         24 |        16 |     354823168 |                    406212608 |
| GPT-2 large  |               1280 |         36 |        20 |     774030080 |                    838220800 |
| GPT-2 XL     |               1600 |         48 |        25 |    1557611200 |                   1637792000 |

: The GPT-2 family. All share a vocabulary of 50257 tokens and a context of 1024 positions. "Tied"
counts the released checkpoints, which share the token-embedding and output matrices and use biases
on the query/key/value projections; the last column is our from-scratch configuration with a
separate output head. Counts computed by the Lab 5 code.
:::

We describe a model by a plain dictionary. Keeping the configuration as data (rather than
hard-coding numbers) is what lets the same class instantiate a 64-dimensional toy model for unit
tests and the real 1.5B-parameter model for inference.

``` {#lst:gpt:config .python caption="The configuration that defines GPT-2 small in our code."}
GPT_CONFIG_124M = dict(
    vocab_size=50257,     # BPE vocabulary of the GPT-2 tokenizer (Ch. 3)
    context_length=1024,  # number of learned position embeddings
    emb_dim=768,          # d_model: width of the residual stream
    n_heads=12,           # attention heads per layer, head_dim = 768/12 = 64
    n_layers=12,          # number of transformer blocks
    drop_rate=0.1,        # dropout probability (0.0 is common for fine-tuning)
    qkv_bias=False,       # Q/K/V biases (True for the OpenAI weights)
)
```

##### Shapes are the contract.

Before writing any layer it pays to fix the shape of the data at every interface, because the whole
model is just a sequence of shape-preserving transformations on a tensor of width
$d_{\text{model}}$. [@fig:gpt:topdown] shows the top-down view. A batch of $B$ sequences of
$T$ token IDs, shape $(B,T)$, is embedded into $(B,T,d_{\text{model}})$ by adding a token embedding
and a learned position embedding. That tensor---the *residual stream*---flows through $L$
transformer blocks, each of which reads it and writes an update of the same shape. A final
normalization and a linear map to the vocabulary produce logits of shape $(B,T,V)$: one score per
vocabulary entry, for every position.

![Top-down view of a GPT model with the tensor shape at each interface. Everything between the
embedding and the output head preserves the shape
$(B,T,d_{\text{model}})$.](figures/ch06_gpt-2.svg){#fig:gpt:topdown}

##### Stub-first development.

A good way to build any multi-component system is to write the outermost structure first with
placeholder components that already honor the interfaces, run it end to end, and then replace the
placeholders one by one. In PyTorch `nn.Identity` is a perfect placeholder for a block or a
normalization layer: it accepts a tensor of any shape and returns it unchanged. The skeleton in
[@lst:gpt:skeleton] runs, produces correctly shaped logits, and gives us a harness to test
each real component as it arrives.

``` {#lst:gpt:skeleton .python caption="A runnable skeleton: real embeddings and head, placeholder blocks and norm."}
import torch, torch.nn as nn

class SkeletonGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # placeholders that respect the (B, T, d) -> (B, T, d) contract
        stubs = [nn.Identity() for _ in range(cfg["n_layers"])]
        self.trf_blocks = nn.Sequential(*stubs)
        self.final_norm = nn.Identity()
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, idx):                       # idx: (B, T)
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.trf_blocks(self.drop_emb(x))      # (B, T, d)
        return self.out_head(self.final_norm(x))   # (B, T, V)

logits = SkeletonGPT(GPT_CONFIG_124M)(torch.randint(0, 50257, (2, 4)))
print(logits.shape)   # torch.Size([2, 4, 50257])
```

Each of the $2\times4$ output rows is a 50,257-dimensional vector. Position $t$ of row $b$ holds the
model's (currently random) scores for the token that should follow the first $t+1$ tokens of
sequence $b$. During training *every* position contributes a prediction, which is why a single
sequence of length $T$ yields $T$ training examples.

::: systemslens
[The vocabulary is a systems parameter]{.box-title}

Two things about $V=50257$ matter for performance. First, the output head is the single largest
matrix in GPT-2 small: $Vd_{\text{model}}=38597376$ parameters, 24% of the untied model. Second,
50257 is an awkward GEMM dimension: it is odd, so tiles of 64 or 128 columns on a tensor-core GPU
always leave a ragged remainder. Padding the vocabulary to 50304 ($=393\times128$) adds only 36096
never-used parameters but lets every tile be full; nanoGPT-style training scripts do exactly this
and observe a measurable speedup. The general rule: choose $d_{\text{model}}$, $d_{\text{ff}}$, head
dimension and vocabulary as multiples of 64 (better 128) so the matmul kernels never run partially
empty tiles.
:::

::: napkinmath
[The logits tensor is bigger than the model]{.box-title}

For a training batch of $B=8$ sequences of $T=1024$ tokens, the logits have
$8\times1024\times50257\approx4.1e8$ entries. In fp32 that is $1.65\,\mathrm{\giga\byte}$---more
than twice the 0.62  of fp32 weights of the untied 163M model---and the softmax inside the
cross-entropy loss materializes another tensor of the same size for the backward pass. This is why
large-vocabulary models (Qwen3 has $V\approx152000$) compute the loss in *chunks* of the sequence,
or with fused "linear + cross-entropy" kernels that never store the full logits
([@ch:pretrain]).
:::

## Normalizing activations with layer normalization {#sec:gpt:ln}

Deep networks are hard to train when the scale of the activations drifts from layer to layer:
gradients either explode or shrink exponentially with depth, and the optimizer spends its effort
chasing scale rather than learning features. *Layer normalization* (Ba et al., 2016) fixes the scale
locally. For each token vector $\boldsymbol{x}\in\mathbb{R}^{d_{\text{model}}}$ independently, it
subtracts the mean of the features and divides by their standard deviation, then applies a learned
per-feature affine map: $$\begin{equation}
\operatorname{LN}(\boldsymbol{x}) = \boldsymbol{\gamma}\odot\frac{\boldsymbol{x}-\mu}{\sqrt{\sigma^2+\epsilon}}+\boldsymbol{\beta},
\qquad \mu=\frac{1}{d}\sum_{i=1}^{d}x_i,\quad \sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2 .
\label{eq:gpt:ln}
\end{equation}$$ The trainable *scale* $\boldsymbol{\gamma}$ (initialized to ones) and *shift*
$\boldsymbol{\beta}$ (initialized to zeros) let the network undo the normalization where that helps;
the small $\epsilon=10^{-5}$ prevents division by zero for a constant input.

##### Along which axis?

The statistics are computed over the *feature* dimension, which in a $(B,T,d_{\text{model}})$ tensor
is the last one, so the code uses `dim=-1` with `keepdim=True` to keep the result broadcastable.
[@fig:gpt:normaxes] contrasts this with batch normalization, which reduces over the batch
(and sequence) axes instead. The difference is decisive for language models: layer norm treats each
token in isolation, so its output does not depend on what else happens to be in the batch, on the
padding, or on how many GPUs the batch is split across.

![Which entries share normalization statistics. Layer norm (left) computes one mean and variance per
token across its $d_{\text{model}}$ features; batch norm (right) computes one per feature across all
tokens in the batch, which couples examples together.](figures/ch06_gpt-3.svg){#fig:gpt:normaxes}

``` {#lst:gpt:ln .python caption="Layer normalization written out explicitly (as in `llmsys/layers.py`)."}
class LayerNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))    # gamma
        self.shift = nn.Parameter(torch.zeros(emb_dim))   # beta

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # divide by d, not d-1
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift
```

Two details deserve comment. We use the *biased* variance estimator (dividing by $d$ rather than
$d-1$). Bessel's correction is meaningless here because we are not estimating a population variance
from a sample, and for $d=768$ the difference is 0.13% anyway; the real reason is compatibility,
since GPT-2 was trained in TensorFlow, whose default is the biased estimator, and we will load its
weights in [@ch:pretrain]. Second, a quick test confirms the contract: after normalization
(with $\gamma=1,\beta=0$) every token has mean $\approx 0$ (on the order of $10^{-8}$ in fp32, i.e.,
rounding noise) and variance $\approx 1$.

``` {#lst:gpt:lncheck .python caption="Checking the normalization contract."}
torch.manual_seed(0)
x = torch.randn(2, 5, 16) * 7 + 3          # badly scaled activations
y = LayerNorm(16)(x)
print(y.mean(-1).abs().max())              # ~1e-7
print(y.var(-1, unbiased=False).mean())    # ~1.0
```

::: systemslens
[LayerNorm is a bandwidth problem, not a FLOP problem]{.box-title}

[@eq:gpt:ln] costs roughly $8$ FLOPs per element but must read every element of
$\boldsymbol{x}$ and write every element of the output. Its arithmetic intensity is therefore on the
order of 1--2 FLOP/byte, two orders of magnitude below the ridge point of any modern GPU
([@ch:pytorch]): the kernel runs at memory bandwidth, and the only way to make it faster is
to move fewer bytes. Our explicit implementation is about as bad as it can be---`mean`, `var`, the
subtraction, the division, the scale and the shift each launch a separate kernel that makes a full
round trip through HBM. A fused LayerNorm kernel reads $\boldsymbol{x}$ once (computing mean and
variance in a single pass with Welford's algorithm or the $E[x^2]-E[x]^2$ identity in fp32), keeps
the row in registers or shared memory, and writes once. Even better is to fuse the norm into the
*neighbouring* matmul or residual add so the normalized tensor never touches HBM at all; this is one
of the main wins of `torch.compile` ([@ch:pretrain]) and of the Triton kernels in
[@app:triton].
:::

::: napkinmath
[How long should one LayerNorm take on an H100?]{.box-title}

For a $B\times T\times d_{\text{model}}= 8\times1024\times768$ bf16 activation, a fused kernel reads
$12.6\,\mathrm{\mega\byte}$ and writes $12.6\,\mathrm{\mega\byte}$. At 3.35  that is
$25.2\,\mathrm{\mega\byte}/3.35\,\mathrm{\tera\byte\per\second}\approx7.5\,\mathrm{\micro\second}$.
The unfused version with six kernels moves roughly six times as many bytes and pays six kernel
launches of a few microseconds each: easily 40 μs--60 μs. Multiply by $2L+1=25$ norms per forward
pass and the difference is a millisecond per step---comparable to the time of the attention matmuls
at this size.
:::

::: deepdive
[Why not batch normalization?]{.box-title}

Batch norm normalizes each feature using statistics pooled over the batch. For language models this
is a poor fit for three systems reasons: (i) at inference time the batch is arbitrary (often a
single request), so batch norm must fall back to running averages collected during training, and
train/test behaviour diverges; (ii) sequences have variable length, and padded positions would
pollute the statistics; (iii) under data parallelism, the statistics either differ per GPU or
require an extra all-reduce in every layer ("SyncBatchNorm"). Layer norm has none of these problems
because it never looks across tokens. Its cousin RMSNorm, which drops the mean subtraction and the
shift, is the default in modern LLMs ([@sec:modern:rmsnorm]).
:::

## A feed-forward network with GELU activations {#sec:gpt:ffn}

Attention mixes information *across* positions; the feed-forward network (FFN, also called the MLP)
transforms each position *independently*. It is a two-layer perceptron that first expands the
$d_{\text{model}}$-dimensional token vector to $d_{\text{ff}}=4d_{\text{model}}$ dimensions, applies
a nonlinearity, and projects back: $$\begin{equation}
\mathrm{FFN}(\boldsymbol{x}) = \mathbf{W}_2\,\operatorname{GELU}(\mathbf{W}_1\boldsymbol{x}+\boldsymbol{b}_1)+\boldsymbol{b}_2,\qquad
\mathbf{W}_1\in\mathbb{R}^{4d\times d},\ \mathbf{W}_2\in\mathbb{R}^{d\times 4d}.
\label{eq:gpt:ffn}
\end{equation}$$ The expansion gives the network a wide hidden space in which features can be
computed and gated; the contraction brings the result back to the width of the residual stream so
blocks can be stacked. Mechanistic-interpretability work often describes the FFN as a key--value
memory: rows of $\mathbf{W}_1$ act as pattern detectors ("keys") and the corresponding columns of
$\mathbf{W}_2$ as the vectors written into the residual stream when the pattern fires ("values").

##### GELU.

GPT-2 uses the Gaussian error linear unit (Hendrycks and Gimpel, 2016) instead of ReLU.
$\operatorname{GELU}(x)=x\,\Phi(x)$, where $\Phi$ is the standard normal CDF: the input is weighted
by the probability that a standard Gaussian falls below it. GPT-2 was trained with the cheaper tanh
approximation $$\begin{equation}
\operatorname{GELU}(x)\approx \tfrac12 x\left(1+\tanh\!\left[\sqrt{2/\pi}\,\big(x+0.044715\,x^3\big)\right]\right),
\label{eq:gpt:gelu}
\end{equation}$$ and since we will load its weights we must use the same formula.
[@fig:gpt:gelu] compares the shapes. Unlike ReLU, GELU is smooth at the origin and lets
small negative inputs pass a small negative output (its minimum is about $-0.17$ near
$x\approx-0.75$), so neurons with negative pre-activations still receive gradient. Those properties
tend to make optimization of deep transformers slightly easier; the SiLU/Swish curve, used in the
gated FFNs of modern models ([@sec:modern:swiglu]), is almost indistinguishable.

![ReLU, GELU and SiLU. GELU and SiLU are smooth, non-monotonic, and give small negative outputs for
moderately negative inputs. (Plotted analytically.)](figures/ch06_gpt-4.svg){#fig:gpt:gelu}

``` {#lst:gpt:ffn .python caption="GELU and the expand--contract feed-forward network."}
import math

class GELU(nn.Module):
    def forward(self, x):
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x.pow(3))))

class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),   # expand:   (B,T,d)  -> (B,T,4d)
            GELU(),
            nn.Linear(4 * emb_dim, emb_dim),   # contract: (B,T,4d) -> (B,T,d)
        )

    def forward(self, x):
        return self.layers(x)
```

![The position-wise FFN expands each token vector fourfold, applies GELU, and contracts it back, so
it can be stacked like any other shape-preserving layer.](figures/ch06_gpt-5.svg){#fig:gpt:ffn}

::: systemslens
[Where the FLOPs live: the FFN's GEMMs]{.box-title}

Per block, the two FFN matrices hold $8d_{\text{model}}^2$ weights versus $4d_{\text{model}}^2$ in
attention's four projections, so the FFN owns two thirds of every block's matmul FLOPs. Applied to
$N=BT$ tokens at once, the first layer is a single GEMM $(N\times d)\cdot(d\times 4d)$ costing
$8Nd^2$ FLOPs and moving (in bf16) $2Nd+8d^2+8Nd$ bytes (read activations, read weights, write
output). Its arithmetic intensity $$I(N)=\frac{8Nd^2}{10Nd+8d^2}\ \xrightarrow{N\to\infty}\ 0.8\,d$$
grows with the number of tokens in the GEMM and saturates at $0.8d$ FLOP/byte. Whether this kernel
is compute- or memory-bound is therefore decided by *how many tokens you batch together*, a fact
that will reappear as the central tension of LLM inference ([@ch:inference]): during training
or prefill $N$ is in the thousands and the FFN is compute bound; during single-sequence decoding
$N=1$, the intensity collapses to about 1 FLOP/byte, and the GEMM becomes a pure weight-streaming
operation. [@fig:gpt:intensity] plots $I(N)$ against the H100 ridge point.
:::

![Arithmetic intensity of the first FFN GEMM (bf16) as a function of how many tokens it processes at
once (analytic). Below the dashed ridge line the kernel is memory bound. The curve for
$d_{\text{model}}=768$ crosses the ridge at about 570 tokens and for $d_{\text{model}}=4096$ at
about 320.](figures/ch06_gpt-6.svg){#fig:gpt:intensity}

::: napkinmath
[When does the FFN become compute bound?]{.box-title}

Setting $I(N)$ equal to the H100's ridge point $989/3.35\approx295$ FLOP/byte and solving
$8Nd^2=295\,(10Nd+8d^2)$ for $N$ gives $N=2360d/(8d-2950)$: about **567 tokens** for $d=768$ and
**324 tokens** for $d=4096$. A training micro-batch of $8\times1024$ tokens is far beyond both; a
decode step with batch 32 is far below. Wider models reach the ridge with fewer tokens because the
weights are re-used more per byte loaded. Note also that GPT-2 small can *never* exceed
$0.8\times768\approx614$ FLOP/byte in this GEMM, only twice the ridge point: small models are never
very far from the memory wall.
:::

## Adding shortcut connections {#sec:gpt:shortcut}

A deep stack of layers computes a composition $f_L\circ\cdots\circ f_1$, and by the chain rule the
gradient reaching layer 1 is a product of $L$ Jacobians. If those Jacobians typically shrink vectors
(saturating sigmoids are the textbook case), the product decays exponentially and the early layers
stop learning: the *vanishing gradient* problem. Shortcut (residual, skip) connections, introduced
for very deep image networks by He et al. (2016), replace each layer $x\mapsto f(x)$ by
$x\mapsto x+f(x)$. The Jacobian of a residual layer is $\mathbf{I}+\partial f/\partial x$, so the
backward pass always contains an identity path from the loss straight to every layer.

We can watch this happen. [@lst:gpt:vanish] builds a 16-layer MLP of width 32 with sigmoid
activations, once plain and once with residual connections, runs one backward pass on random data,
and prints the mean absolute gradient of each layer's weights.

``` {#lst:gpt:vanish .python caption="A vanishing-gradient experiment (our own setup)."}
class DeepMLP(nn.Module):
    def __init__(self, depth=16, width=32, residual=False):
        super().__init__()
        self.residual = residual
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(width, width), nn.Sigmoid())
            for _ in range(depth))
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x) if self.residual else layer(x)
        return self.head(x)

x, y = torch.randn(64, 32), torch.randn(64, 1)
for residual in (False, True):
    torch.manual_seed(1)
    net = DeepMLP(residual=residual)
    nn.functional.mse_loss(net(x), y).backward()
    grads = [l[0].weight.grad.abs().mean().item() for l in net.layers]
    print(residual, [f"{g:.1e}" for g in grads[::4]])
```

![Gradient magnitude per layer in a 16-layer sigmoid MLP, measured on CPU in this environment with
[@lst:gpt:vanish]. Without shortcuts the gradient shrinks by about 12 orders of magnitude
between the last and first layer; with them it stays within one order of
magnitude.](figures/ch06_gpt-7.svg){#fig:gpt:vanish}

[@fig:gpt:vanish] shows the result: in the plain network the first layer receives gradients
around $10^{-16}$, which in fp32 arithmetic is effectively zero, while in the residual network every
layer gets a usable signal.

##### The residual stream.

In a transformer, residual connections do more than fix gradients; they define the architecture's
central data structure. It is helpful to picture a single vector per token---the residual
stream---flowing from the embedding to the output head, with every attention and FFN sublayer
*reading* from it (through a LayerNorm) and *adding* a write back into it
([@fig:gpt:residualstream]). Because sublayers only add, information placed in the stream
early (e.g., the token identity) remains available to all later layers unless something actively
cancels it.

![The residual-stream view of a transformer: two blocks shown, each with an attention and an FFN
sublayer reading the stream and adding their output
back.](figures/ch06_gpt-8.svg){#fig:gpt:residualstream}

::: deepdive
[Pre-LN, post-LN, and residual scaling]{.box-title}

The original transformer (Vaswani et al., 2017) placed the normalization *after* the addition,
$x\leftarrow\operatorname{LN}(x+f(x))$ ("post-LN"). GPT-2 moved it *before* the sublayer,
$x\leftarrow x+f(\operatorname{LN}(x))$ ("pre-LN"). Xiong et al. (2020) showed that post-LN produces
large gradients near the output at initialization and needs a careful learning-rate warmup, whereas
pre-LN keeps the identity path clean (nothing ever normalizes the stream itself) and trains stably
at depth. The price is that the stream's magnitude grows with depth: each of the $2L$ sublayers adds
a roughly independent update, so the variance of the stream grows linearly in $L$. GPT-2 therefore
scales the initial weights of each sublayer's *output* projection by $1/\sqrt{2L}$, and adds a final
LayerNorm before the head so the logits see a normalized vector. Both conventions survive in today's
models.
:::

## Connecting attention and linear layers in a transformer block {#sec:gpt:block}

We now have every ingredient of the GPT-2 block ([@fig:gpt:block]): masked multi-head
attention from [@ch:attention], the FFN, two LayerNorms, two residual connections, and
dropout applied to each sublayer's output before it is added back (sometimes called *residual
dropout*). The block is shape-preserving: $(B,T,d_{\text{model}})$ in, $(B,T,d_{\text{model}})$ out.
That property is not incidental---it is what allows the block to be stacked any number of times and
what makes the model's depth a single configuration number.

![The pre-LN GPT-2 transformer block. Green arrows are the two shortcut
connections.](figures/ch06_gpt-9.svg){#fig:gpt:block}

``` {#lst:gpt:block .python caption="The transformer block (cf. `llmsys/gpt.py`)."}
from llmsys.layers import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"], num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"], qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg["emb_dim"])
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.drop_shortcut(self.att(self.norm1(x)))   # attention
        x = x + self.drop_shortcut(self.ff(self.norm2(x)))    # FFN sublayer
        return x
```

The code mirrors the figure line by line. Note that we never write `shortcut = x` and add it later:
`x = x + f(x)` creates a *new* tensor, so the old `x` is still referenced by the autograd graph and
the in-place semantics are safe.

::: systemslens
[Anatomy of one block: what is compute bound and what is not]{.box-title}

[@tab:gpt:blockops] lists the operations in one GPT-2 small block for a training
micro-batch of $N=BT=8192$ tokens in bf16, with their FLOPs and the minimum bytes they must move.
Four GEMM groups (QKV, attention scores and weighted sum, output projection, two FFN layers) are
comfortably compute bound; everything else---two LayerNorms, softmax, GELU, two dropouts, two
residual adds---is elementwise or a row reduction with intensity of a few FLOP/byte. In an eager
PyTorch run, those cheap operations are a long tail of small memory-bound kernels, and on a GPU they
can take 30--50% of the step time despite contributing under 1% of the FLOPs. This is the motivation
for fusion: FlashAttention fuses mask+softmax+dropout into the attention matmuls
([@ch:attention]), GEMM epilogues fuse bias+GELU, and compilers fuse residual-add+LayerNorm.
:::

::: {#tab:gpt:blockops .table-float}
| Operation (one block) | FLOPs | Min. bytes moved | Bound |
|:---|---:|---:|:---|
| Q, K, V projections ($3\times d\!\times\! d$) | $6Nd^2$ | $6Nd+6d^2+\ldots$ | compute |
| $QK^\top$ and $AV$ (per sequence $T\times T$) | $4NTd$ | $\propto BHT^2$ if unfused | compute |
| mask $+$ softmax $+$ dropout | $\approx 5BHT^2$ | $\approx 4BHT^2\cdot 2$ | memory |
| output projection | $2Nd^2$ | $4Nd+2d^2$ | compute |
| FFN up $+$ down | $16Nd^2$ | $20Nd+16d^2$ | compute |
| GELU (tanh) | $\approx 15\cdot 4Nd$ | $16Nd$ | memory |
| 2$\times$ LayerNorm, 2$\times$ residual add, 2$\times$ dropout | $\approx 30Nd$ | $\approx 24Nd$ | memory |

: Operation inventory for one pre-LN block on $N=BT$ tokens ($d=d_{\text{model}}$, bf16, 2 bytes per
element). Counts are approximate for the elementwise rows.
:::

## Coding the GPT model {#sec:gpt:model}

With the block in hand, the full model is short. [@lst:gpt:model] replaces the placeholders
of the skeleton with real blocks and a real final LayerNorm. (The library version in `llmsys/gpt.py`
additionally threads an optional KV cache and padding mask through the blocks; we will need those
from [@ch:inference] onwards.)

``` {#lst:gpt:model .python caption="The complete GPT model."}
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, idx):                        # (B, T) token ids
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop_emb(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.trf_blocks:
            x = blk(x)
        return self.out_head(self.final_norm(x))   # (B, T, V) logits
```

##### Counting parameters.

Instantiating the model with the 124M configuration and summing `p.numel()` over
`model.parameters()` gives 163009536, not 124 million. The difference is *weight tying*: the
released GPT-2 reuses the token-embedding matrix ($V\times d_{\text{model}}$) as the output
projection, while our model has a separate `out_head` of the same shape. Removing it subtracts
$50257\times768=38597376$, leaving 124412160; the remaining 27648 difference to the official
124439808 comes from the QKV biases the released checkpoints use ($3\times768\times12$). Every
parameter can be accounted for in closed form. Per block, $$\begin{align}
P_\text{attn} &= \underbrace{3d^2}_{W_Q,W_K,W_V} + \underbrace{3d}_{\text{QKV bias, optional}} + \underbrace{d^2+d}_{\text{out proj}}, \\
P_\text{FFN}  &= \underbrace{4d^2+4d}_{W_1,b_1} + \underbrace{4d^2+d}_{W_2,b_2},\qquad
P_\text{LN} = 2\times 2d,
\end{align}$$ and for the whole model $$\begin{equation}
P = \underbrace{Vd}_{\text{tok emb}} + \underbrace{T_\text{max} d}_{\text{pos emb}}
 + L\,(P_\text{attn}+P_\text{FFN}+P_\text{LN}) + \underbrace{2d}_{\text{final LN}}
 + \underbrace{Vd}_{\text{head, if untied}} \;\approx\; 12Ld^2 + 2Vd .
\label{eq:gpt:params}
\end{equation}$$ Lab 5 asks you to implement this formula and asserts that it matches the module
exactly for several configurations. [@fig:gpt:parambreakdown] shows how the budget splits
across the four GPT-2 sizes: the embedding and head dominate the small model, while for GPT-2 XL the
transformer blocks account for 90% of the parameters and the FFN alone for 60%.

![Parameter breakdown of the four GPT-2 sizes in our untied configuration (LayerNorm parameters are
too small to see). Computed with `count_params_analytic` from
Lab 5.](figures/ch06_gpt-10.svg){#fig:gpt:parambreakdown}

::: napkinmath
[Memory footprint of the weights]{.box-title}

At 4 bytes per fp32 parameter, the untied 163M model needs
$163009536\times4\,\text{B}\approx622\,\mathrm{\mebi\byte}$; the tied 124M checkpoint in bf16 needs
$249\,\mathrm{\mega\byte}$. GPT-2 XL in fp32 is 6.1 . These numbers are only the *weights*: training
with AdamW in mixed precision needs about 16 bytes per parameter for weights, gradients and
optimizer state ([@ch:pretrain]), so GPT-2 XL needs $\approx26\,\mathrm{\giga\byte}$ before a
single activation is stored---already more than a 24 GB consumer GPU.
:::

::: systemslens
[Weight tying as a systems trade-off]{.box-title}

Tying saves $Vd_{\text{model}}$ parameters (24% of GPT-2 small, 5% of GPT-2 XL) and the
corresponding gradient and optimizer memory. It does *not* save FLOPs: the output projection is
still a $d\times V$ GEMM on every position. It also couples two very different gradient signals
(sparse row updates from the embedding lookup, dense updates from the head) into one tensor, and in
pipeline-parallel training the first and last stages must synchronize that shared tensor's gradient
with an extra all-reduce. Small models (GPT-2, Qwen3-0.6B to 4B) tend to tie; large ones (Qwen3-8B
and up, Llama 3) usually do not.
:::

### Initialization

A freshly constructed `nn.Linear` uses PyTorch's default Kaiming-uniform initialization, which was
designed for ReLU convolutional networks. GPT-2 instead draws every weight matrix from
$\mathcal N(0,0.02^2)$, sets biases to zero, and (as discussed in [@sec:gpt:shortcut])
shrinks the standard deviation of the two *output* projections in each block (attention's `out_proj`
and the FFN's second linear) by $1/\sqrt{2L}$, so that the $2L$ residual updates sum to a stream of
roughly constant scale at initialization. [@lst:gpt:init] applies this recipe. It matters
only when training from scratch---loaded checkpoints overwrite every value---but a bad
initialization is one of the most common reasons a from-scratch pretraining run diverges in its
first few hundred steps.

``` {#lst:gpt:init .python caption="GPT-2 style initialization."}
import math

def init_gpt2_weights(model, std=0.02):
    L = model.cfg["n_layers"]
    for name, p in model.named_parameters():
        if p.dim() == 2:                          # weight matrices and embeddings
            s = std / math.sqrt(2 * L) if name.endswith(
                ("out_proj.weight", "ff.layers.2.weight")) else std
            torch.nn.init.normal_(p, mean=0.0, std=s)
        elif name.endswith(("bias", "shift")):
            torch.nn.init.zeros_(p)
        # LayerNorm scales keep their default value of 1
```

::: systemslens
[Initialization at scale]{.box-title}

For a 124M-parameter model initialization takes a fraction of a second; for a 70B-parameter model,
drawing 70 billion Gaussian samples on one CPU core takes minutes and needs 280 GB of host memory in
fp32. Large-scale training frameworks therefore construct the model on PyTorch's `meta` device
(shapes only, no storage), shard it across GPUs first (FSDP, [@ch:distributed]), and only
then run the initializer on each GPU's shard. Two subtleties follow. First, every data-parallel
replica must start from *identical* weights, so either one rank initializes and broadcasts, or all
ranks seed their random generators identically. Second, the initializer must be written
per-parameter-name (as in [@lst:gpt:init]) rather than per-module-constructor, because
under sharding the module that "owns" a weight may never see its full shape.
:::

::: deepdive
[What dropout costs]{.box-title}

Dropout looks free, but in an eager implementation each dropout call generates a random tensor of
the same shape as its input, stores a boolean mask for the backward pass (1 byte per element, or one
bit if packed), and launches two extra kernels. GPT-2 has $2L+1$ residual/embedding dropouts plus
$L$ dropouts on the $BHT^2$ attention probabilities, the largest activation tensor in the model.
Fused kernels regenerate the mask in the backward pass from the random generator's seed and offset
(Philox counters) instead of storing it, which is how FlashAttention supports attention dropout
without materializing anything. For pretraining modern LLMs on trillions of tokens, the usual answer
is simpler: set the dropout rate to zero ([@ch:modern]).
:::

## Generating text {#sec:gpt:generate}

The model outputs logits for every position, but to generate text we only need the last one: it
scores every candidate for the token that follows the whole input. Generation is a loop
([@fig:gpt:genloop]): run the model, take the logits at position $-1$, pick a token, append
it to the input, repeat. Picking the token with the highest score is *greedy decoding*. Because
softmax is monotonic, the argmax of the logits equals the argmax of the probabilities, so the
softmax can be skipped; we will need the probabilities once we sample
([@ch:pretrain; @ch:scaling]).

![The autoregressive generation loop. Each iteration re-runs the model on the (cropped) sequence,
keeps only the last position's logits, and appends the chosen
token.](figures/ch06_gpt-11.svg){#fig:gpt:genloop}

``` {#lst:gpt:generate .python caption="Greedy generation without a cache."}
@torch.no_grad()
def generate_greedy(model, idx, max_new_tokens, context_size):
    model.eval()                               # disable dropout
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]      # never exceed the position table
        logits = model(idx_cond)[:, -1, :]     # (B, V): only the last position
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx
```

Three details are easy to get wrong. (i) `model.eval()` switches dropout off; leaving it on makes
generation nondeterministic and slightly worse. (ii) The crop `idx[:, -context_size:]` is mandatory
for a model with learned absolute positions: GPT-2 has exactly 1024 position embeddings and indexing
the 1025th raises an error. (iii) Since the weights are random, the untrained model produces
fluent-looking nonsense: a valid sequence of token IDs that decodes to unrelated word pieces. Giving
the model useful weights is the subject of [@ch:pretrain], where we both train it and load
OpenAI's checkpoint.

::: napkinmath
[The quadratic cost of cache-less generation]{.box-title}

Each step of [@lst:gpt:generate] re-processes the whole prefix. A forward pass over a
sequence of length $t$ costs about $t\cdot(2P_\text{matmul} + 4Ldt)$ FLOPs
([@sec:gpt:accounting]). Generating 1000 tokens from a 10-token prompt with GPT-2 small
therefore costs $\sum_{t=10}^{1009} F(t)\approx139$ TFLOP (computed with the Lab 5 formula), whereas
processing each new token only once---which is what the KV cache of [@ch:inference]
enables---costs about 0.27 TFLOP, roughly **500$\times$** less. The cache-less loop is $O(n^2)$ in
the matmuls and $O(n^3)$ in attention; it is fine for a teaching example and useless in production.
:::

## Accounting for every FLOP and byte {#sec:gpt:accounting}

We close the chapter with the systems analysis that the rest of the book builds on. The goal is
three formulas---FLOPs, weight memory, activation memory---each validated against PyTorch.

### FLOPs of a forward pass

A matrix product $(m\times k)\cdot(k\times n)$ performs $mkn$ multiply--accumulates, conventionally
counted as $2mkn$ FLOPs. Every `nn.Linear` applied to $N$ tokens is such a product with $m=N$, so it
costs $2N$ times its number of weights. Summing over the block gives $2N\cdot12d^2$; the head adds
$2NVd$. The two attention matmuls are different because their "weights" are other activations:
$QK^\top$ for one head of one sequence is $(T\times d_{h})\cdot(d_{h}\times T)$, i.e. $2T^2d_{h}$
FLOPs, and $AV$ costs the same, so per layer and sequence the pair costs $4T^2d_{h}H = 4T^2 d$.
Putting it together for $B$ sequences of length $T$: $$\begin{equation}
F_\text{fwd}(B,T) = BT\Big[\,L\,(24d^2 + 4Td) + 2Vd\,\Big].
\label{eq:gpt:flops}
\end{equation}$$ Per token, the matmul part is twice the number of non-embedding parameters (plus
the head), the classic "$2N$ FLOPs per token" rule; the attention part grows linearly with context
and overtakes the $24d^2$ term once $T>6d$ (4608 tokens for GPT-2 small).

``` {#lst:gpt:flopcounter .python caption="Verifying [@eq:gpt:flops] against PyTorch's own FLOP counter."}
from torch.utils.flop_counter import FlopCounterMode

def forward_flops(cfg, batch, seq_len):
    d, V, L = cfg["emb_dim"], cfg["vocab_size"], cfg["n_layers"]
    per_token = L * (24 * d * d + 4 * seq_len * d) + 2 * d * V
    return batch * seq_len * per_token

model = GPTModel(cfg).eval()
idx = torch.randint(0, cfg["vocab_size"], (2, 16))
with FlopCounterMode(display=False) as fc:
    model(idx)
assert fc.get_total_flops() == forward_flops(cfg, 2, 16)   # exact match
```

The match in [@lst:gpt:flopcounter] is *exact* because `FlopCounterMode` counts only
matmul-like operators with the same $2mkn$ convention, and our explicit attention computes the full
$T\times T$ score matrix even though half of it is masked. (A causal FlashAttention kernel skips the
masked tiles and does half the attention FLOPs; FLOP counters then disagree with "useful" FLOPs, a
recurring source of confusion when people report utilization.) For GPT-2 small at $T=1024$ the
formula gives 0.285 GFLOP per token.

::: keyidea
[The 6N rule]{.box-title}

The backward pass of a matmul computes two products of the same size as the forward one (gradient
with respect to the input and with respect to the weight), so it costs about twice the forward.
Training therefore costs $\approx 3\times F_\text{fwd}$, i.e. about $6P$ FLOPs per token for a model
with $P$ (non-embedding) parameters when attention is a minor term. Training GPT-2 small on 10
billion tokens costs $6\times1.24e8\times10^{10}\approx7.4e18$ FLOPs: about 2.1 hours on one H100 at
100% utilization, or roughly 5 hours at a realistic 40% MFU ([@ch:pretrain]).
:::

### Activation memory

For the backward pass, autograd keeps every tensor that some operation's derivative needs. Walking
through our block on $s$ tokens per sequence, batch $b$, width $d$ and $h$ heads, the saved tensors
are: the input of LN1 ($sbd$ elements); its output, shared by the Q/K/V projections ($sbd$); $Q$ and
$K$ for the score matmul ($2sbd$); the softmax probabilities ($bhs^2$); $V$ ($sbd$); the context fed
to the output projection ($sbd$); the input of LN2 ($sbd$); the FFN input ($sbd$); the pre-GELU
hidden ($4sbd$); and the post-GELU hidden fed to the second linear ($4sbd$). In total
$$\begin{equation}
A_\text{layer} \approx \big(16\,sbd + b h s^2\big)\ \text{elements},
\label{eq:gpt:act}
\end{equation}$$ which with 2-byte activations and 1-byte dropout masks becomes the widely quoted
$sbd\,(34+5hs/d)$ bytes of Korthikanti et al. (2022). Two consequences: activation memory is linear
in batch size and in depth, and the attention term grows *quadratically* in sequence length, which
is precisely what FlashAttention removes by never storing the probabilities.

``` {#lst:gpt:savedhooks .python caption="Measuring saved activations with autograd hooks."}
def measure_saved_activation_bytes(model, idx):
    params = {p.untyped_storage().data_ptr() for p in model.parameters()}
    seen = {}
    def pack(t):                        # called for every tensor saved for backward
        ptr = t.untyped_storage().data_ptr()
        if ptr not in params and ptr not in seen:
            seen[ptr] = t.untyped_storage().nbytes()
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        model(idx)
    return sum(seen.values())
```

Running [@lst:gpt:savedhooks] on the tiny test configuration ($d=64$, 2 layers, 4 heads,
one sequence of 64 tokens, fp32) gives the numbers in [@tab:gpt:actmeasured]. The measured
value is about $2.1\times$ the estimate of [@eq:gpt:act]. The explanation is instructive:
our explicit tanh-GELU is built from half a dozen primitive operations, and autograd saves several
of their intermediates, each $4d$ wide. Replacing it with PyTorch's fused
`nn.GELU(approximate="tanh")`, which saves only its input, removes 28% of all activation memory; the
explicit LayerNorm similarly saves extra intermediates that a fused kernel would not. Fusion saves
memory, not just time.

::: {#tab:gpt:actmeasured .table-float}
| Configuration (tiny GPT, $b=1$, $s=64$, fp32) | Saved activations |
|:----------------------------------------------|------------------:|
| Estimate, [@eq:gpt:act]              |              655  |
| Measured, explicit GELU and LayerNorm         |             1387  |
| Measured, fused `nn.GELU`, explicit LayerNorm |              994  |

: Activation memory: estimate vs. measurement with saved-tensor hooks (measured on CPU in this
environment). The measured value doubles exactly when the batch doubles.
:::

::: napkinmath
[Activations of GPT-2 small at full context]{.box-title}

For $b=8$, $s=1024$, $d=768$, $h=12$, $L=12$ in bf16, [@eq:gpt:act] gives per layer
$16\cdot1024\cdot8\cdot768\cdot2\,\text{B}\approx201\,\mathrm{\mega\byte}$ for the linear terms and
$8\cdot12\cdot1024^2\cdot2\,\text{B}\approx201\,\mathrm{\mega\byte}$ for the attention
probabilities: 0.4  per layer, 4.8  for the whole model, plus the 1.65  fp32 logits. That is roughly
ten times the weights. At $s=8192$ the attention term alone would be 12.9  *per layer*. This is why
long-context training requires both FlashAttention (no stored probabilities) and activation
checkpointing ([@ch:pretrain]).
:::

### Where does the time go?

FLOPs are not time. The Lab 5 `bench.py` script profiles a forward pass with `torch.profiler` and
groups the time by operator. On a GPU you will typically find the large GEMMs (`aten::addmm`,
`aten::mm`) at the top of the table but far from 100% of the time, followed by a long tail of
elementwise kernels (`aten::mul`, `aten::add`, `aten::tanh`, `aten::pow`) from our explicit GELU and
LayerNorm, and copies from `transpose().contiguous()` in attention. Dividing
[@eq:gpt:flops] by the measured time gives the achieved FLOP/s; dividing that by the device
peak gives the *model FLOPs utilization* (MFU), the single most useful efficiency number in LLM
systems and the subject of [@ch:pretrain].

::: systemslens
[A first look at kernel fusion and compilation]{.box-title}

The gap between "FLOPs" and "time" in an eager model has three sources: memory-bound elementwise
kernels that each round-trip through HBM, per-kernel launch overhead (a few microseconds each, which
dominates small models), and extra copies from layout changes. All three are attacked by the same
technique---*fusion*---either by hand (fused LayerNorm, fused bias-GELU, FlashAttention), by vendor
libraries (cuBLASLt epilogues), or automatically by `torch.compile`, whose TorchInductor backend
traces the model, groups pointwise and reduction operations into generated Triton kernels, and can
capture the whole forward pass in a CUDA graph to eliminate launch overhead. We will measure the
effect in [@ch:pretrain; @ch:inference] and write such kernels ourselves in
[@app:triton].
:::

## Summary

- A GPT model is an embedding layer, a stack of $L$ identical shape-preserving transformer blocks, a
  final LayerNorm, and a linear head producing $(B,T,V)$ logits.

- LayerNorm normalizes each token across its features; it is independent of the batch and therefore
  well suited to variable-length, distributed training. It is memory bound and an important fusion
  target.

- The FFN expands to $4d_{\text{model}}$, applies GELU, and contracts back. It holds two thirds of
  each block's weights and FLOPs; its arithmetic intensity depends on how many tokens are processed
  at once.

- Residual connections keep gradients alive in deep networks and define the residual stream; pre-LN
  placement and $1/\sqrt{2L}$ output scaling keep training stable.

- GPT-2 small has 163009536 parameters with a separate head and 124439808 in the released tied
  checkpoint; parameters, FLOPs ($\approx 2P$ per token forward, $6P$ training) and activation
  memory ($\approx 16sbd+bhs^2$ elements per layer) can all be computed exactly and checked against
  PyTorch.

- Greedy generation repeatedly runs the model and appends the argmax token; without a cache it
  re-processes the whole prefix each step, which is quadratically wasteful.

## Exercises

::: exercise
[GPT-2 sizes]{.box-title}

Using [@eq:gpt:params], compute the parameter counts of GPT-2 medium, large and XL in the
untied configuration, and separately the share held by attention and by the FFN. Check your numbers
against [@tab:gpt:sizes] and [@fig:gpt:parambreakdown]. At what $d_{\text{model}}$
does the FFN first hold more parameters than the token embedding plus the head, for $L=12$ and
$V=50257$?
:::

::: exercise
[Separate dropout rates]{.box-title}

Our configuration uses one `drop_rate` for the embedding dropout, the residual dropout, and the
attention-weight dropout. Change the configuration and code so the three rates can be set
independently, and argue which of the three is cheapest to remove from a systems perspective.
:::

::: exercise
[Vocabulary padding]{.box-title}

Pad the vocabulary of GPT-2 small to the next multiple of 128. How many parameters and how many
bytes of fp32 optimizer state (two Adam moments) does that add? What must the generation code do to
make sure the padded rows are never sampled?
:::

::: exercise
[Ridge point]{.box-title}

Derive the crossover $N^\star$ at which the first FFN GEMM becomes compute bound on an A100 (312
TFLOP/s bf16, 2.04 TB/s) for $d=768$ and $d=4096$. Repeat for the second FFN GEMM, whose input is
$4d$ wide. Which of the two GEMMs reaches the ridge first, and why?
:::

::: exercise
[Residual growth]{.box-title}

Initialize GPT-2 small with all weights drawn from $\mathcal N(0,0.02^2)$ and measure the RMS of the
residual stream after each block on random tokens. Then scale the weights of every `out_proj` and
second FFN linear by $1/\sqrt{2L}$ and measure again. Plot both curves and explain the shape of the
first.
:::

::: exercise
[Post-LN]{.box-title}

Implement a post-LN variant of `TransformerBlock`. Train a 24-layer tiny model with both variants
for a few hundred steps on the same data without learning-rate warmup. Which one diverges first as
you raise the learning rate?
:::

::: exercise
[Fused activations save memory]{.box-title}

Using [@lst:gpt:savedhooks], measure saved-activation bytes for (a) our explicit model, (b)
the model with `nn.GELU(approximate="tanh")`, (c) additionally with `nn.LayerNorm`. Explain each
reduction by listing the tensors autograd stops saving.
:::

::: exercise
[Causal FLOPs]{.box-title}

[@eq:gpt:flops] counts the full $T\times T$ attention. Derive the "useful" FLOPs when only
the causal half is computed, and compute the relative difference for GPT-2 small at $T=1024$ and
$T=8192$. Why does it matter when comparing MFU numbers across papers?
:::

::: exercise
[Cost of generation]{.box-title}

Write a function that returns the total FLOPs of generating $n$ tokens from a $p$-token prompt (a)
without a cache, (b) with a cache. Plot the ratio as a function of $n$ for $p=100$. At which $n$
does the cached version save $100\times$?
:::

::: exercise
[Chunked loss]{.box-title}

Design (and implement) a cross-entropy computation that processes the sequence in chunks of 128
positions so that the full $(B,T,V)$ logits tensor is never materialized. What is the peak memory
saving for $B=8$, $T=1024$, $V=50257$ in fp32? What extra cost do you pay in the backward pass?
:::

## Lab 6: Assemble GPT-2 and account for every parameter, FLOP and byte {#sec:gpt:lab}

::: labbox
[Assemble GPT-2 and account for every parameter, FLOP and byte]{.box-title}

**Code:** `code/labs/lab06_gpt`

**Goals**

Build the GPT-2 architecture from its components so that it is bit-compatible with
`llmsys.gpt.GPTModel`, then turn the chapter's accounting into tested code: an exact parameter
formula, an exact forward-FLOP formula, and an activation-memory measurement.

**Setup**

``` bash
cd code && pip install -r requirements.txt
pytest -q labs/lab06_gpt            # runs the tests against your starter.py
```

**Tasks**

1.  **Components.** Implement `LayerNorm.forward`, `gelu_tanh` and `FeedForward.__init__`. Keep the
    attribute names (`scale`, `shift`, `layers`) so state dicts are interchangeable with the
    library.

2.  **Block and model.** Implement `TransformerBlock.forward` (pre-LN, residual dropout) and
    `GPTModel.forward`. The test loads the library model's weights into yours and requires identical
    logits.

3.  **Generation.** Implement `generate_greedy` with context cropping; the test generates past
    `context_length` to make sure you crop.

4.  **Accounting.** Implement `count_params_analytic` (with and without tying) and `forward_flops`.
    The tests compare against `sum(p.numel())` and against `FlopCounterMode` for several shapes, and
    pin the GPT-2 small numbers 163009536 and 124439808.

5.  **Activation memory.** Implement `activation_bytes_estimate` ([@eq:gpt:act]) and
    `measure_saved_activation_bytes` with `saved_tensors_hooks`.

**Measurements**

- Run `python labs/lab06_gpt/bench.py --impl starter --seq 256 --batch 4` on your machine. Record
  forward time, achieved TFLOP/s, and the top 10 operators. Then swap in the fused GELU and
  `nn.LayerNorm` and re-run. Expect the elementwise tail to shrink noticeably on a GPU.

- Plot measured activation bytes against batch size (should be a straight line through the origin)
  and against sequence length (should curve upwards: the $bhs^2$ term).

- If you have the HF `gpt2` `model.safetensors`, pass `--weights` and confirm your model generates
  coherent English with greedy decoding.

**Deliverables**

Passing tests; a short table of operator times before and after fusion; the two activation plots
with one paragraph explaining their shapes.

**Stretch goals**

- Wrap the model in `torch.compile` and compare time and saved-activation bytes.

- Implement the chunked cross-entropy of the last exercise and measure peak memory with
  `torch.cuda.max_memory_allocated`.
:::
