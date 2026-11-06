# Attention Mechanisms, from Dot Products to FlashAttention {#ch:attention}

Attention is the one operation in a transformer that mixes information *across* positions.
Everything else---embeddings, layer norms, MLPs---acts on each token independently. It is also the
one operation whose cost grows quadratically with sequence length, and so the one that dominates
long-context training, shapes the KV cache that dominates serving memory, and motivated the most
influential systems paper of the transformer era, FlashAttention.

This chapter builds attention up in small steps: a weight-free version that exposes the core
"weighted average of the inputs" idea, then trainable queries, keys and values, then causal masking
and dropout, then multi-head attention in two implementations. With the algorithm in hand, we
analyse its FLOPs and memory traffic, derive the online softmax, and implement an IO-aware tiled
attention whose forward and backward passes match the naive version to machine precision without
ever forming the $T\times T$ attention matrix.

![Roadmap: steps 1--4 build the attention used in GPT-2 (blue); steps 5--7 analyse and optimise it
(amber).](figures/ch07_attention-1.svg){#fig:attention:roadmap}

## The problem with modeling long sequences {#sec:attention:long}

Consider translating a sentence. Word-by-word substitution fails immediately: languages reorder
words, split one word into several, and resolve meanings from context far away in the sentence. A
translation model must read the whole source, keep what matters, and consult the relevant parts as
it writes each output word.

Before transformers, the standard architecture was the *encoder--decoder recurrent neural network*
(Sutskever et al., 2014; Cho et al., 2014). The encoder RNN reads the source tokens one at a time,
updating a hidden state $\boldsymbol{h}_t = f(\boldsymbol{h}_{t-1},\boldsymbol{x}_t)$. Its final
hidden state---a single fixed-size vector---is handed to the decoder, which generates the output one
token at a time from that vector and its own running state ([@fig:attention:rnn], left).

The weakness is the bottleneck: the entire meaning of an arbitrarily long input must be squeezed
through one vector of a few hundred numbers, and the decoder has no way to look back at individual
input positions. Empirically, translation quality fell off sharply for long sentences. A second,
systems-flavoured weakness is that the recurrence is *sequential*: $\boldsymbol{h}_t$ cannot be
computed before $\boldsymbol{h}_{t-1}$, so training cannot parallelise across the time dimension of
a sequence, leaving most of a GPU idle.

![Left: the encoder--decoder RNN must compress the whole input into one vector $\boldsymbol{c}$.
Right: with attention, each decoder step forms its own weighted combination of *all* encoder states;
line width shows how strongly each input is used to produce
"Cache".](figures/ch07_attention-2.svg){#fig:attention:rnn}

## Capturing data dependencies with attention {#sec:attention:deps}

Bahdanau et al. (2015) removed the bottleneck by letting the decoder, at every output step, compute
a fresh set of weights over *all* encoder hidden states and use their weighted sum as context
([@fig:attention:rnn], right). The weights are produced by a small learned scoring function
and normalised with a softmax, so the decoder can "attend" to whichever input positions are relevant
to the word it is producing.

Three years later, Vaswani et al. (2017) asked what happens if one drops the recurrence entirely and
builds the whole network out of attention. The resulting *transformer* uses *self-attention*: a
sequence attends to *itself*, so every position's new representation is a learned, data-dependent
mixture of every (allowed) position's old representation. Without recurrence, all positions of a
sequence are processed in parallel by large matrix multiplications, which is precisely what GPUs are
built for. That property, more than any modelling advantage, is why transformers displaced RNNs.

::: keyidea
[Self-attention in one sentence]{.box-title}

For each position $i$, compute a probability distribution $\alpha_{i,\cdot}$ over positions, based
on how well $i$'s "question" matches each position's "label", and output the $\alpha$-weighted
average of the positions' "contents". Queries, keys and values are those three roles.
:::

## Self-attention without trainable weights {#sec:attention:simple}

To isolate the mechanism, we first drop all learned parameters. Take a six-token input whose
3-dimensional embeddings we simply write down (real models use $d$ in the hundreds or thousands and
learn the embeddings, [@ch:text]):

``` {#lst:attention:inputs .python caption="The running example: six tokens with toy 3-d embeddings."}
inputs = torch.tensor(
  [[0.40, 0.10, 0.80],   # fused     (x1)
   [0.60, 0.90, 0.20],   # kernels   (x2)
   [0.50, 0.80, 0.70],   # keep      (x3)
   [0.10, 0.30, 0.90],   # data      (x4)
   [0.90, 0.20, 0.10],   # on        (x5)
   [0.20, 0.70, 0.40]])  # chip      (x6)
```

### One query at a time

Pick the third token, "keep", as the *query*, $\boldsymbol{q}=\boldsymbol{x}^{(3)}$. Self-attention
computes its *context vector* $\boldsymbol{z}^{(3)}$ in three steps.

1.  **Scores.** Measure the similarity of the query to every input with a dot product:
    $\omega_{3j} = \boldsymbol{x}^{(3)}\cdot\boldsymbol{x}^{(j)}$. A dot product is large when two
    vectors point the same way and have large norms, so it serves as an un-normalised relevance
    score. For our numbers: $\boldsymbol{\omega}_3 = (0.84,\ 1.16,\ 1.38,\ 0.92,\ 0.68,\ 0.94)$.

2.  **Weights.** Normalise the scores into a probability distribution with a softmax,
    $\alpha_{3j} = e^{\omega_{3j}}/\sum_k e^{\omega_{3k}}$, giving
    $\boldsymbol{\alpha}_3 = (0.140, 0.193, 0.241, 0.152, 0.120, 0.155)$. Softmax keeps every weight
    positive, makes them sum to one, and exaggerates differences (it is a smooth arg-max).

3.  **Context.** Average the inputs with those weights:
    $\boldsymbol{z}^{(3)} = \sum_j \alpha_{3j}\boldsymbol{x}^{(j)} = (0.446,\ 0.558,\ 0.530)$.

[@fig:attention:onequery] draws the computation. Note that the query attends most strongly
to itself (its dot product with itself is its squared norm), a pattern that trainable projections
will be free to change.

![Weight-free self-attention for the query "keep": dot-product scores against every input, softmax
to weights, weighted sum to the context
vector.](figures/ch07_attention-3.svg){#fig:attention:onequery}

::: pitfall
[Never write softmax as `exp(x)/exp(x).sum()`]{.box-title}

A score of 100 overflows `exp` in fp16 (max $\approx 65504$) and one of 1000 overflows fp32. Since
$\operatorname{softmax}(\boldsymbol{x}) = \operatorname{softmax}(\boldsymbol{x}-c)$ for any constant
$c$, numerically stable implementations subtract the row maximum first. `torch.softmax` does this;
so must every hand-written kernel. The running maximum is also the key to the online softmax of
[@sec:attention:flash].
:::

### All queries at once

Repeating the procedure for every query is just three matrix operations on the input matrix
$\mathbf{X}\in\mathbb{R}^{T\times d}$: $$\begin{equation}
\mathbf{\Omega} = \mathbf{X}\mathbf{X}^\top \in\mathbb{R}^{T\times T},\qquad
\mathbf{A} = \operatorname{softmax}_{\text{row}}(\mathbf{\Omega}),\qquad
\mathbf{Z} = \mathbf{A}\mathbf{X}\in\mathbb{R}^{T\times d}.
\label{eq:attention:simple}
\end{equation}$$ Row $i$ of $\mathbf{A}$ is the distribution of query $i$ over all keys
([@fig:attention:heat]). In PyTorch it is three lines, and replacing Python loops with
these matrix operations is the first and most important performance optimisation in this chapter:
one $T\times d$ by $d\times T$ GEMM instead of $T^2$ tiny dot products.

``` {#lst:attention:simple .python caption="Weight-free self-attention for all tokens ([@lab:attention], Part A)."}
def simple_attention(x):              # x: (T, d)
    scores = x @ x.T                  # (T, T) all pairwise dot products
    weights = torch.softmax(scores, dim=-1)   # each row sums to 1
    return weights @ x, weights       # (T, d) context vectors
```

![Attention weight matrices $\mathbf{A}$ for the running example (rows: queries, columns: keys;
numbers computed with [@lst:attention:simple]). Each row sums to 1. In (b) the hatched
future positions are masked and each row is renormalised over the visible
prefix.](figures/ch07_attention-4.svg){#fig:attention:heat}

## Self-attention with trainable weights {#sec:attention:trainable}

The weight-free version has no parameters to learn: what a token attends to is fixed by raw
embedding similarity, and what it retrieves is the raw embedding itself. *Scaled dot-product
attention* (Vaswani et al., 2017) introduces three learned projection matrices
$\mathbf{W}_q,\mathbf{W}_k\in\mathbb{R}^{d_{\text{in}}\times d_k}$ and
$\mathbf{W}_v\in\mathbb{R}^{d_{\text{in}}\times d_v}$ that give each token three separate roles:

- **query** $\boldsymbol{q}_i=\boldsymbol{x}_i\mathbf{W}_q$ --- what token $i$ is looking for;

- **key** $\boldsymbol{k}_j=\boldsymbol{x}_j\mathbf{W}_k$ --- what token $j$ advertises about
  itself;

- **value** $\boldsymbol{v}_j=\boldsymbol{x}_j\mathbf{W}_v$ --- what token $j$ contributes if
  attended to.

The terminology comes from databases and key--value stores: a query is matched against keys, and the
matching entries' values are returned---except that here matching is soft and every value is
returned with some weight.

### Computing the attention weights step by step

For the full input, $$\begin{equation}
\mathbf{Q}=\mathbf{X}\mathbf{W}_q,\quad \mathbf{K}=\mathbf{X}\mathbf{W}_k,\quad \mathbf{V}=\mathbf{X}\mathbf{W}_v,\qquad
\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \operatorname{softmax}\!\Big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\Big)\mathbf{V}.
\label{eq:attention:sdpa}
\end{equation}$$ [@fig:attention:qkv] traces the shapes for our example with
$d_{\text{in}}=3$ and $d_k=d_v=2$ (deliberately different so the shapes cannot be confused; in GPT
models $d_{\text{in}}=d_{\text{out}}$).

![Shapes in single-head scaled dot-product attention for the running example ($T=6$,
$d_{\text{in}}=3$, $d_k=d_v=2$).](figures/ch07_attention-5.svg){#fig:attention:qkv}

##### Why divide by $\sqrt{d_k}$?

Suppose the entries of $\boldsymbol{q}$ and $\boldsymbol{k}$ are independent with mean 0 and
variance 1. Then $\boldsymbol{q}\cdot\boldsymbol{k}=\sum_{m=1}^{d_k}q_mk_m$ has mean 0 and variance
$d_k$. With $d_k=128$, raw scores have a standard deviation of about 11; a softmax over such scores
is nearly one-hot, and its gradient (which is proportional to $\alpha_i(1-\alpha_i)$ on the
diagonal) nearly vanishes. Dividing by $\sqrt{d_k}$ restores unit variance at initialisation and
keeps the softmax in a regime where gradients flow. (Some newer models instead normalise
$\boldsymbol{q}$ and $\boldsymbol{k}$ directly---the QK-norm of Qwen3, [@ch:modern].)

### A compact self-attention class

Wrapping the computation in an `nn.Module` gives us trainable parameters, device movement and
state-dict serialisation for free:

``` {#lst:attention:sa .python caption="Single-head self-attention with trainable projections ([@lab:attention], Part B)."}
class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):                       # (..., T, d_in)
        q, k, v = self.W_query(x), self.W_key(x), self.W_value(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
        return torch.softmax(scores, dim=-1) @ v   # (..., T, d_out)
```

A first version might store the projections as raw `nn.Parameter(torch.rand(d_in, d_out))` and
compute `x @ W`. Using `nn.Linear` instead changes two things. First, `nn.Linear` stores its weight
as $(d_{\text{out}},d_{\text{in}})$ and computes $\boldsymbol{x}\mathbf{W}^\top$---the two versions
compute the same function if one copies `W.T` into the linear layer, which is a useful test. Second,
`nn.Linear` uses a scaled (Kaiming-uniform) initialisation, whereas `torch.rand` draws from $[0,1)$;
all-positive, unscaled weights make scores grow with $d$ and training less stable.

## Hiding future words with causal attention {#sec:attention:causal}

A language model is trained to predict token $t+1$ from tokens $1..t$. If position $t$ could attend
to position $t+1$ during training, the task would be trivial (copy the answer) and the model would
learn nothing useful for generation, where the future does not yet exist. *Causal* (or *masked*)
attention restricts query $i$ to keys $j\le i$.

### Applying a causal attention mask

There are two equivalent ways to apply the mask ([@fig:attention:mask]).

1.  *Mask after softmax, then renormalise*: compute $\mathbf{A}=\operatorname{softmax}(\mathbf{S})$,
    zero the entries above the diagonal, and divide each row by its new sum.

2.  *Mask before softmax*: set $s_{ij}=-\infty$ for $j>i$, then apply softmax. Since $e^{-\infty}=0$
    the masked entries get exactly zero weight and the remaining entries are already normalised.

They agree exactly: in route 1, row $i$'s surviving entries are $e^{s_{ij}}/Z$ with
$Z=\sum_{k} e^{s_{ik}}$ over *all* $k$; renormalising divides by $\sum_{k\le i}e^{s_{ik}}/Z$, and
the full-row $Z$ cancels. In particular, the future *does not leak*: although route 1's first
softmax looks at future scores, their influence is removed entirely by the renormalisation. Route 2
is what every implementation uses---it is one fused operation, does less work, and is what
FlashAttention exploits to skip masked tiles entirely.

![Causal masking applied before the softmax. Only the lower triangle (including the diagonal)
receives probability mass.](figures/ch07_attention-6.svg){#fig:attention:mask}

``` {#lst:attention:causalroutes .python caption="Two routes to causal weights; they are equal to float precision."}
def causal_softmax(scores):                     # route 2: -inf before softmax
    T = scores.shape[-1]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)

def mask_after_softmax(scores):                 # route 1: zero, renormalise
    w = torch.tril(torch.softmax(scores, dim=-1))
    return w / w.sum(dim=-1, keepdim=True)
```

### Masking additional attention weights with dropout

*Dropout* zeroes a random subset of activations during training (each with probability $p$) and
scales the survivors by $1/(1-p)$ so expected values are unchanged; at evaluation time it is the
identity. In GPT-2, dropout is applied in three places, one of them to the attention weight matrix
after the softmax, so that a token cannot rely too heavily on any single other position
([@fig:attention:dropout]). With $p=0.5$ a weight of 0.28 becomes either 0 or 0.56, and a
row no longer sums exactly to 1---only in expectation.

![Dropout on the attention weights (one random draw, illustrative). Zeros are dropped connections;
surviving weights double so each row's *expected* sum is still
1.](figures/ch07_attention-7.svg){#fig:attention:dropout}

::: systemslens
[Dropout is not free in fused kernels]{.box-title}

In eager PyTorch, attention dropout materialises a $B\times H\times T\times T$ random mask and
stores it for the backward pass---as large as the attention matrix itself. FlashAttention instead
generates dropout bits on the fly from a counter-based RNG (Philox) keyed by a seed and an offset,
and *regenerates* identical bits in the backward pass, storing nothing. This is why fused kernels
take a seed/offset rather than a mask, and why bitwise reproducibility of dropout across different
kernels or tile sizes is not guaranteed. Most modern LLMs pretrain with attention dropout set to 0
anyway; with trillions of tokens and a single epoch, overfitting is not the concern and the extra
RNG work is wasted.
:::

### Implementing a compact causal attention class

Two practical changes turn the above into a reusable layer: it must accept batches of shape
$(B,T,d_{\text{in}})$, and the mask should be created once and moved with the module. Registering
the mask with `register_buffer` does the latter: buffers are moved by `.to(device)` with the
parameters but are not trained. We register it with `persistent=False` so it is not written to
checkpoints (it can always be recomputed).

``` {#lst:attention:causal .python caption="Batched causal attention with dropout ([@lab:attention], Part C)."}
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.0,
                 qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask.bool(), persistent=False)

    def forward(self, x):                                   # (B, T, d_in)
        B, T, _ = x.shape
        q, k, v = self.W_query(x), self.W_key(x), self.W_value(x)
        scores = q @ k.transpose(1, 2) / math.sqrt(self.d_out)  # (B, T, T)
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        w = self.dropout(torch.softmax(scores, dim=-1))
        return w @ v                                         # (B, T, d_out)
```

The slice `self.mask[:T, :T]` lets the layer handle any length up to `context_length`. A good test
of causality ([@lab:attention]) perturbs the inputs at positions $\ge t$ and checks that outputs at positions
$<t$ are bit-for-bit unchanged.

## Extending single-head attention to multi-head attention {#sec:attention:mha}

A single head computes one attention pattern per query. But a token typically needs several kinds of
information at once---the subject it agrees with, the noun a pronoun refers to, the previous token,
the matching bracket. *Multi-head attention* runs $H$ attention heads in parallel, each with its own
projections into a smaller $d_{h}=d_{\text{out}}/H$-dimensional subspace, concatenates their
outputs, and mixes them with an output projection $\mathbf{W}_o$: $$\begin{equation}
\operatorname{MHA}(\mathbf{X}) = \big[\,\operatorname{head}_1\,\Vert\,\cdots\,\Vert\,\operatorname{head}_H\big]\mathbf{W}_o,\qquad
\operatorname{head}_h = \operatorname{Attention}(\mathbf{X}\mathbf{W}_q^{(h)},\mathbf{X}\mathbf{W}_k^{(h)},\mathbf{X}\mathbf{W}_v^{(h)}).
\end{equation}$$ The total parameter count and FLOPs match one big head of width $d_{\text{out}}$;
what changes is that the model gets $H$ independent attention distributions per token.

### Stacking multiple single-head attention layers

The literal implementation keeps a list of `CausalAttention` modules and concatenates their outputs
along the feature dimension. With $d_{\text{out}}=2$ per head and $H=2$ heads, the output is
$(B,T,4)$.

``` {#lst:attention:wrapper .python caption="Multi-head attention as a list of heads."}
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads,
                 qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, H*d_out)
```

### Implementing multi-head attention with weight splits

The wrapper runs the heads one after another: $3H$ small projection GEMMs, $H$ score GEMMs, $H$
softmaxes and $H$ value GEMMs, each a separate kernel launch on a GPU. The standard implementation
instead uses *one* $d_{\text{in}}\times d_{\text{out}}$ projection each for
$\mathbf{Q},\mathbf{K},\mathbf{V}$ and splits the result into heads by reshaping
([@fig:attention:split]):
$(B,T,d_{\text{out}})\xrightarrow{\text{view}}(B,T,H,d_{h})\xrightarrow{\text{transpose}}(B,H,T,d_{h})$.
Batched matrix multiplication then treats $(B,H)$ as batch dimensions, so all heads are computed by
single kernels.

![Weight-split multi-head attention: one projection, then reshape into heads, batched attention over
$(B,H)$, merge, output projection. Same shapes for $\mathbf{K}$ and
$\mathbf{V}$.](figures/ch07_attention-8.svg){#fig:attention:split}

``` {#lst:attention:mha .python caption="Multi-head attention with weight splits ([@lab:attention], Part D; same parameter names as `llmsys.layers.MultiHeadAttention`)."}
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads,
                 qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out, self.num_heads = d_out, num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)      # mixes the heads
        self.dropout = nn.Dropout(dropout)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        self.register_buffer("mask", mask.bool(), persistent=False)

    def forward(self, x):
        B, T, _ = x.shape
        H, Dh = self.num_heads, self.head_dim
        split = lambda t: t.view(B, T, H, Dh).transpose(1, 2)    # (B, H, T, Dh)
        q, k, v = (split(f(x)) for f in (self.W_query, self.W_key, self.W_value))
        scores = q @ k.transpose(2, 3) / math.sqrt(Dh)          # (B, H, T, T)
        scores = scores.masked_fill(self.mask[:T, :T], float("-inf"))
        w = self.dropout(torch.softmax(scores, dim=-1))
        ctx = (w @ v).transpose(1, 2).contiguous().view(B, T, self.d_out)
        return self.out_proj(ctx)
```

Note the `.contiguous()` before the final `view`: after `transpose` the tensor's memory layout no
longer matches its logical shape, and `view` requires compatible strides. `.contiguous()` performs a
physical copy into the merged-head layout (alternatively, `.reshape` copies only when needed). For
GPT-2 small, $H=12$ heads of $d_{h}=64$ give $d_{\text{out}}=768$; GPT-2 XL uses 25 heads of 64 for
$d=1600$.

::: systemslens
[Why weight splitting is faster: kernel granularity]{.box-title}

For GPT-2 small ($d=768$, $H=12$) the wrapper performs 36 projections of size
$(BT\times768)\cdot(768\times64)$; the weight-split version performs 3 of size
$(BT\times768)\cdot(768\times768)$---the same FLOPs. But a $768\times64$ weight gives each GEMM far
fewer output tiles to spread over a GPU's 100+ streaming multiprocessors, so many SMs idle, and each
call pays a few microseconds of launch overhead. GPT-2's original code goes one step further and
fuses Q, K and V into a single $768\times2304$ projection (`c_attn`), which is why the weight loader
in `llmsys/gpt.py` splits it in three. The general lesson---*batch small operations into large
ones*---reappears at every level of this book, from heads to requests ([@ch:serving]).
:::

## The cost of attention {#sec:attention:cost}

We now treat attention as a workload. Let $B$ be the batch size, $T$ the sequence length, $H$ the
number of heads, $d_{h}$ the head dimension and $d=Hd_{h}$ the model width.

##### FLOPs.

Counting a multiply--add as 2 FLOPs, one forward pass of one attention layer costs $$\begin{align}
\text{projections } (\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{W}_o) &: \; 4\cdot 2BTd^2 = 8BTd^2,\\
\mathbf{Q}\mathbf{K}^\top \text{ and } \mathbf{A}\mathbf{V} &: \; 2\cdot 2BHT^2d_{h}= 4BT^2d .
\end{align}$$ The quadratic term overtakes the projections when $4BT^2d > 8BTd^2$, i.e. when $T>2d$:
around 1.5k tokens for GPT-2 small, 8k for a $d=4096$ model. Causal masking halves the useful work
of the quadratic term (only the lower triangle matters), which a good kernel exploits.

[@tab:attention:gpt2] instantiates these formulas for one GPT-2-small attention layer. Even
at GPT-2's modest context of 1024, the score and value products are a third of the layer's FLOPs,
and the materialised attention matrix is the largest activation in the layer by a wide margin.

::: {#tab:attention:gpt2 .table-float}
| quantity (one layer, $B=8$, $T=1024$, $d=768$, $H=12$) | formula | value | share |
|:---|---:|---:|---:|
| Q, K, V, O projections (fwd FLOPs) | $8BTd^2$ | 38.7 GFLOP | 67% |
| $\mathbf{Q}\mathbf{K}^\top$ and $\mathbf{A}\mathbf{V}$ (fwd FLOPs, dense) | $4BT^2d$ | 25.8 GFLOP | 33% |
| input / output activations (fp32) | $BTd\cdot4$ B | 25.2 MB each |  |
| $\mathbf{Q},\mathbf{K},\mathbf{V}$ (fp32) | $3BTd\cdot4$ B | 75.5 MB |  |
| attention matrix $\mathbf{A}$ (fp32), kept for backward | $BHT^2\cdot4$ B | 403 MB |  |

: FLOPs and activation memory of one GPT-2-small attention layer (analytic). The $T^2$ terms are
already significant at $T=1024$; they dominate beyond $T\approx2d$.
:::

##### Memory.

The naive implementation materialises the scores $\mathbf{S}$ and probabilities $\mathbf{A}$, each
$B\cdot H\cdot T^2$ elements, and keeps $\mathbf{A}$ (and a dropout mask) for the backward pass.
Everything else is $O(BTd)$.

::: napkinmath
[The $T^2$ wall]{.box-title}

Take one layer of a model with $H=32$ heads at $T=8192$, $B=1$, bf16 (2 bytes):
$32\cdot 8192^2\cdot 2 = 4.3$ GB for one copy of the attention matrix. Training must keep it for
backward in *every* layer: 32 layers would need $\approx 137$ GB---more than an 80 GB H100 holds,
for a single sequence. At $T=32{,}768$ one layer alone needs $\approx 69$ GB. Long-context training
with materialised attention is simply impossible.
:::

##### Arithmetic intensity.

Is naive attention compute-bound or memory-bound? Per head, $\mathbf{Q}\mathbf{K}^\top$ performs
$2T^2d_{h}$ FLOPs and must at minimum write $T^2$ scores. The mask, scaling, softmax and dropout are
elementwise passes over $T^2$ values with only a handful of FLOPs each. With $d_{h}=128$ the score
GEMM itself is fine, but the elementwise passes have an arithmetic intensity of order 1 FLOP/byte,
far below the hundreds of FLOPs/byte an H100 needs to be compute-bound ([@ch:pytorch]).

::: napkinmath
[Naive attention on an H100: where the time goes]{.box-title}

One head, $T=8192$, $d_{h}=128$, bf16. Matmul FLOPs: $4T^2d_{h}= 34.4$ GFLOP,
i.e. $\approx 35\,\mu$s at 989 TFLOP/s. If the unfused implementation makes about four round trips
of the $T\times T$ matrix through HBM (write $\mathbf{S}$; read $\mathbf{S}$, write $\mathbf{A}$;
read $\mathbf{A}$ for $\mathbf{A}\mathbf{V}$), traffic is
$4\cdot T^2\cdot 2\,\text{B}\approx 537$ MB, i.e. $\approx 160\,\mu$s at 3.35 TB/s. The kernel
spends about $4.6\times$ longer moving the attention matrix than computing with it. Avoiding those
round trips---not reducing FLOPs---is the goal.
:::

![Memory needed per attention layer for the backward pass (analytic). The naive implementation
stores the $T\times T$ probabilities; IO-aware attention stores only inputs, output and one scalar
per row.](figures/ch07_attention-9.svg){#fig:attention:mem}

## IO-aware attention: the online softmax and FlashAttention {#sec:attention:flash}

FlashAttention (Dao et al., 2022) computes *exact* attention---the same numbers as
[@eq:attention:sdpa] up to floating-point reassociation---without writing $\mathbf{S}$ or
$\mathbf{A}$ to HBM. It rests on two ideas: *tiling*, so that blocks of $\mathbf{Q}$, $\mathbf{K}$,
$\mathbf{V}$ fit in on-chip SRAM (shared memory and registers), and the *online softmax*, which
makes it possible to normalise rows whose entries are seen one block at a time.

### The online softmax

The softmax of a row $\boldsymbol{x}$ needs its maximum $m=\max_j x_j$ (for stability) and
normaliser $\ell=\sum_j e^{x_j-m}$---global quantities over the whole row. The trick, due to Milakov
and Gimelshein (2018), is that both can be maintained incrementally. Suppose we have processed a
prefix of the row and hold $(m,\ell)$ for it. A new block $\boldsymbol{x}_B$ arrives:
$$\begin{equation}
m' = \max\big(m,\ \max_j x_{B,j}\big),\qquad
\ell' = \ell\,e^{m-m'} + \sum_j e^{x_{B,j}-m'} .
\label{eq:attention:online}
\end{equation}$$ The factor $e^{m-m'}\le 1$ *rescales* the old normaliser from the old reference
maximum to the new one. After the last block, $(m,\ell)$ are exactly the global values.

``` {#lst:attention:onlinesm .python caption="Online softmax over blocks ([@lab:attention], Part E)."}
def online_softmax(x, block):
    m = torch.full(x.shape[:-1], float("-inf"))     # running max
    l = torch.zeros(x.shape[:-1])                   # running normaliser
    for s in range(0, x.shape[-1], block):
        xb = x[..., s:s + block]
        m_new = torch.maximum(m, xb.max(dim=-1).values)
        l = l * torch.exp(m - m_new) \
            + torch.exp(xb - m_new[..., None]).sum(-1)
        m = m_new
    return torch.exp(x - m[..., None]) / l[..., None]   # second pass
```

Listing [-@lst:attention:onlinesm] still makes a second pass to emit probabilities. Attention
does not need the probabilities themselves, only $\mathbf{A}\mathbf{V}$, and the same rescaling
applies to a running *output accumulator*: if $\boldsymbol{o}$ holds
$\sum_j e^{s_j-m}\boldsymbol{v}_j$ for the prefix, then on a new block
$\boldsymbol{o}' = e^{m-m'}\boldsymbol{o} + \sum_{j\in B}e^{s_j-m'}\boldsymbol{v}_j$, and the final
output is $\boldsymbol{o}/\ell$. A single pass over $\mathbf{K}$ and $\mathbf{V}$ suffices.

![FlashAttention's data movement for one query tile $\mathbf{Q}_i$ (one thread block).
$\mathbf{Q}_i$ is loaded once, key/value tiles stream through SRAM, scores and probabilities live
only on chip, and the output tile plus one log-sum-exp per row are written
once.](figures/ch07_attention-10.svg){#fig:attention:tiling}

### The tiled forward pass

Putting the pieces together gives Algorithm 4.1 (boxed below), implemented in plain PyTorch in
`llmsys/flash.py` ([@lst:attention:flashfwd]). On a GPU, the outer loop over query tiles is
the grid of thread blocks (one block per $(b,h,i)$), and the inner loop runs inside each block,
keeping $\mathbf{Q}_i$, the current $\mathbf{K}_j,\mathbf{V}_j$, the score tile and the accumulators
in shared memory and registers. Appendix [-@app:triton] writes this kernel in Triton.

``` {#lst:attention:flashfwd .python caption="Tiled attention forward with online softmax (`llmsys/flash.py`, simplified)."}
for qs in range(0, Tq, block_q):                       # parallel on a GPU
    qb = q[..., qs:qs + block_q, :]
    m = full(-inf); l = zeros(); acc = zeros(block_q, D)
    k_end = min(Tk, qs + block_q + offset) if causal else Tk  # skip future
    for ks in range(0, k_end, block_k):                # sequential per block
        kb, vb = k[..., ks:ks + block_k, :], v[..., ks:ks + block_k, :]
        s = (qb @ kb.transpose(-2, -1)) * scale        # lives in SRAM only
        if causal:
            s = s.masked_fill(k_pos[None, :] > q_pos[:, None], -inf)
        m_new = torch.maximum(m, s.max(-1).values)
        p = torch.exp(s - m_new[..., None])
        alpha = torch.exp(m - m_new)                   # rescale old state
        l = alpha * l + p.sum(-1)
        acc = alpha[..., None] * acc + p @ vb
        m = m_new
    out[..., qs:qs + block_q, :] = acc / l[..., None]
    lse[..., qs:qs + block_q] = m + torch.log(l)       # saved for backward
```

::: figure
:::

##### IO complexity.

Let $M$ be the SRAM capacity in elements. A query tile of $B_r$ rows needs room for itself, a key
and a value tile, and the accumulator, so $B_r,B_c=\Theta(M/D)$. Each of the $T/B_r$ query tiles
streams all of $\mathbf{K}$ and $\mathbf{V}$ once, for $\Theta(T^2D/B_r)=\Theta(T^2D^2/M)$ element
reads, versus $\Theta(TD+T^2)$ for the naive algorithm. Dao et al. prove this is optimal up to
constants for exact attention. Because $D^2/M$ is a small constant for typical head dimensions
(about $1/6$ for $D=128$ in bf16 with H100-sized shared memory), the asymptotic win is a constant
factor of traffic *plus* the elimination of all $T^2$-sized intermediate tensors and the separate
mask/softmax/dropout kernels; in practice the latter dominate.

### The backward pass: recompute instead of store

Backpropagation through $\mathbf{O}=\mathbf{P}\mathbf{V}$ with
$\mathbf{P}=\operatorname{softmax}(\mathbf{S})$ normally needs $\mathbf{P}$. FlashAttention stores
only $\mathbf{O}$ and the per-row log-sum-exp $\boldsymbol{L}$, and *recomputes* each tile
$\mathbf{P}_{ij}=\exp(\mathbf{S}_{ij}-\boldsymbol{L}_i)$ during the backward pass from
$\mathbf{Q}_i$ and $\mathbf{K}_j$. With $\mathbf{dO}$ the incoming gradient, the tile-local
gradients are $$\begin{gather}
\mathbf{dV}_j \mathrel{+}= \mathbf{P}_{ij}^\top\mathbf{dO}_i,\qquad
\mathbf{dP}_{ij}=\mathbf{dO}_i\mathbf{V}_j^\top,\qquad
\mathbf{dS}_{ij}=\mathbf{P}_{ij}\odot(\mathbf{dP}_{ij}-\boldsymbol{\delta}_i),\\
\mathbf{dQ}_i \mathrel{+}= \mathbf{dS}_{ij}\mathbf{K}_j,\qquad
\mathbf{dK}_j \mathrel{+}= \mathbf{dS}_{ij}^\top\mathbf{Q}_i ,
\end{gather}$$ (each $\mathbf{dS}$ also carries the $1/\sqrt{D}$ scale), where the softmax
Jacobian's row term $\boldsymbol{\delta}_i=\operatorname{rowsum}(\mathbf{P}_i\odot\mathbf{dP}_i)$
would seem to need the whole row. It does not:
$\operatorname{rowsum}(\mathbf{P}\odot\mathbf{dO}\mathbf{V}^\top)=\operatorname{rowsum}(\mathbf{dO}\odot\mathbf{P}\mathbf{V})=\operatorname{rowsum}(\mathbf{dO}\odot\mathbf{O})$,
which is computed once up front from $\mathbf{O}$ and $\mathbf{dO}$. Recomputation costs roughly one
extra $\mathbf{Q}\mathbf{K}^\top$ GEMM, but saves storing and re-reading $O(T^2)$ bytes; on
memory-bound hardware this makes the backward pass *faster*, not just leaner. This is activation
checkpointing ([@ch:pretrain]) applied at the granularity of a single kernel.

::: systemslens
[What FlashAttention-2 and -3 changed]{.box-title}

FlashAttention-2 (Dao, 2023) reorganised the work: it parallelises over the sequence dimension as
well as batch and heads (important when $B\cdot H$ is smaller than the number of SMs, as in
long-context training), swaps the loop order in the backward pass so that
$\mathbf{dK}_j,\mathbf{dV}_j$ are accumulated in registers and written once, partitions work across
warps to avoid shared-memory communication, and minimises non-matmul FLOPs (e.g. deferring the
$1/\ell$ rescale to the end), because on tensor-core GPUs a non-matmul FLOP is an order of magnitude
more expensive than a matmul FLOP. FlashAttention-3 (Shah et al., 2024) targets Hopper: asynchronous
TMA loads and warp-specialised producer/consumer pipelines overlap data movement with WGMMA
tensor-core instructions, softmax of one tile is overlapped with the GEMMs of the next, and FP8
support uses incoherent processing to limit quantisation error. For decoding, where $T_q=1$,
parallelising over queries is useless; *split-KV* kernels (FlashDecoding) parallelise over chunks of
the KV cache and combine partial results with the same log-sum-exp algebra ([@ch:inference]).
:::

::: pitfall
[Exact, but not bit-identical]{.box-title}

Tiling changes the order of floating-point additions, so FlashAttention's output differs from the
naive result in the last bits, and different tile sizes or GPUs give different bits. Our float64
tests in [@lab:attention] agree to $\sim10^{-15}$; in bf16 expect relative differences around $10^{-3}$. This
matters for RL ([@ch:rlsys]), where the rollout engine and the trainer may use different
attention kernels and so assign slightly different probabilities to the same tokens.
:::

### Using it from PyTorch

In practice one calls PyTorch's fused entry point,

``` python
F.scaled_dot_product_attention(q, k, v, is_causal=True)   # "SDPA"
```

and lets PyTorch dispatch to a FlashAttention, memory-efficient (Rabe and Staats, 2021, style),
cuDNN, or plain math backend depending on hardware, dtype, head dimension and masking. The core
`llmsys` attention layers accept `use_sdpa=True` to switch from the explicit implementation to SDPA,
and the core test suite checks that both produce the same logits. One caveat: an arbitrary boolean
`attn_mask` can force a slower backend than `is_causal=True`, because fused kernels generate
structured masks on the fly instead of reading a $T\times T$ mask from memory.

## Attention at inference time: a preview of the KV cache {#sec:attention:decode}

During training, all $T$ queries are processed at once. During generation ([@ch:inference]),
the model produces one token per step, and causal masking guarantees that the keys and values of
earlier positions never change once computed. They can therefore be stored---the *KV cache*---so
that each new step projects only the newest token into
$\boldsymbol{q},\boldsymbol{k},\boldsymbol{v}$, appends $\boldsymbol{k},\boldsymbol{v}$ to the
cache, and attends with a single query over all cached keys ([@fig:attention:decode]). The
core `llmsys` attention modules accept a cache object for exactly this purpose, and the general
causal mask `llmsys.layers.causal_mask(q_len, k_len)` handles $T_q<T_k$ by placing query $i$ at
absolute position $T_k-T_q+i$.

![Prefill processes the prompt in parallel and fills the cache; each decode step adds one key/value
pair (red) and attends with a single query over everything
cached.](figures/ch07_attention-11.svg){#fig:attention:decode}

::: napkinmath
[Decode attention is a memory stream]{.box-title}

For one decode step with context $T_k$, one layer reads the whole cache:
$2\cdot H_{kv}\cdot T_k\cdot d_{h}$ values, and does $\approx 4\cdot H\cdot T_k\cdot d_{h}$ FLOPs
with them. With $H=H_{kv}$ (plain MHA) and bf16 that is about
$4\,\text{FLOPs}/(4\,\text{bytes}) = 1$ FLOP/byte, two to three orders of magnitude below the H100's
balance point of $989/3.35\approx 295$ FLOP/byte. Decode attention runs at memory bandwidth, so its
time is (cache bytes)/(3.35 TB/s). Two levers follow: read fewer bytes (GQA/MQA, cache quantisation;
[@ch:modern; @ch:efficient]), or amortise each byte over more queries---which is what
batching many requests does *not* achieve for attention, since each request has its own cache
([@ch:serving]).
:::

::: deepdive
[What do heads learn?]{.box-title}

Interpretability work has found recurring head types in trained transformers: *previous-token heads*
that attend to position $i-1$; *induction heads* (Olsson et al., 2022) that look for an earlier
occurrence of the current token and attend to the token that followed it, implementing in-context
copying of patterns ("...A B ...A $\to$ B"); and heads whose attention collapses onto the first
token or onto delimiters, acting as a no-op "sink". Such *attention sinks* matter for systems:
StreamingLLM (Xiao et al., 2024) shows that a windowed KV cache must keep the first few tokens to
remain stable, a constraint any cache-eviction policy must respect.
:::

## Beyond dense multi-head attention {#sec:attention:variants}

The attention of this chapter is *dense*: every query sees every allowed key, and every head has its
own keys and values. Most of the attention variants in modern models change one of those two
properties, for systems reasons more than modelling reasons.

##### Sharing keys and values: MQA and GQA.

At inference time the keys and values of all past tokens are cached ([@ch:inference]), and
reading that cache is what makes decoding slow. Multi-query attention (Shazeer, 2019) keeps $H$
query heads but a single key/value head; grouped-query attention (Ainslie et al., 2023) uses $G$
key/value heads, each shared by $H/G$ query heads. The KV cache shrinks by $H/G$. Qwen3-0.6B has 16
query heads and 8 KV heads; [@ch:modern] implements GQA and quantifies the savings.
DeepSeek's multi-head latent attention (MLA) instead caches a low-rank compressed latent from which
keys and values are reconstructed.

##### Restricting the pattern.

Sliding-window attention lets each token attend only to the previous $w$ tokens, making cost $O(Tw)$
and the KV cache bounded; models often interleave windowed and global layers. Block-sparse and
dilated patterns (Child et al., 2019; Beltagy et al., 2020) are other points in the design space.
Document masks for packed training sequences ([@ch:text]) are a block-diagonal restriction
whose purpose is correctness rather than speed. Prefix-LM masks let a prompt attend bidirectionally
while the continuation remains causal. [@fig:attention:patterns] shows these masks side by
side; all are supported efficiently by FlashAttention-style kernels because whole tiles can be
skipped.

![Attention masks as query (row) $\times$ key (column) patterns; shaded cells are allowed. Tiles
that are entirely unshaded can be skipped by a tiled
kernel.](figures/ch07_attention-12.svg){#fig:attention:patterns}

##### Changing the algorithm.

Linear attention (Katharopoulos et al., 2020) replaces $\exp(\boldsymbol{q}\cdot\boldsymbol{k})$ by
a kernel feature map so that $\mathbf{A}\mathbf{V}$ can be computed as
$\phi(\mathbf{Q})(\phi(\mathbf{K})^\top\mathbf{V})$ in $O(Td^2)$, turning attention into a
recurrence with a fixed-size state; state-space models and their hybrids pursue the same goal. These
trade exact retrieval over the whole context for constant per-token cost.

##### Distributing the sequence.

When even $O(T)$ memory per device is too much, the sequence itself is split across GPUs. Ring
attention (Liu et al., 2023) passes key/value blocks around a ring of devices while each device
keeps its query block, using exactly the online-softmax merge of [@eq:attention:online] to
combine partial results. [@ch:distributed] covers this context parallelism, and
[@ch:serving] covers PagedAttention, which changes *where* keys and values live in memory
rather than which ones are attended to.

::: napkinmath
[Choosing a tile size]{.box-title}

An H100 SM has up to 228 KB of shared memory. With $D=128$ in bf16, one $64\times128$ tile is 16 KB.
A forward block holding $\mathbf{Q}_i$ ($B_r=128$ rows: 32 KB), double-buffered $\mathbf{K}_j$ and
$\mathbf{V}_j$ ($B_c=128$: $2\times2\times 32$ KB $=128$ KB) uses 160 KB, leaving headroom; the
$128\times128$ fp32 score tile lives in registers spread across the block's warps. Doubling $D$ to
256 halves the feasible tile sizes, which is one reason head dimensions above 128 are less efficient
and kernels specialise by head dimension.
:::

## Summary {#sec:attention:summary}

- Attention replaces the RNN's single context vector with a per-output, data-dependent weighted
  average over all inputs; self-attention applies this within one sequence and parallelises across
  positions.

- Scaled dot-product attention projects inputs to queries, keys and values, computes
  $\operatorname{softmax}(\mathbf{Q}\mathbf{K}^\top/\sqrt{d_k})\mathbf{V}$, and scales by
  $\sqrt{d_k}$ to keep the softmax out of saturation.

- Causal masking (scores set to $-\infty$ above the diagonal) prevents a position from seeing its
  future; masking after the softmax with renormalisation is equivalent. Dropout on attention weights
  is a training-only regulariser.

- Multi-head attention runs $H$ heads of width $d/H$; the weight-split implementation computes all
  heads with a few large batched GEMMs and an output projection.

- Attention costs $O(T^2d)$ FLOPs and, naively, $O(BHT^2)$ memory; at long context the memory and
  the HBM traffic of the $T\times T$ matrix, not the FLOPs, are the binding constraints.

- FlashAttention tiles $\mathbf{Q},\mathbf{K},\mathbf{V}$, uses the online softmax to normalise
  incrementally, stores one log-sum-exp per row, and recomputes probabilities in the backward pass:
  exact attention with $O(T)$ memory and far less HBM traffic.

- GQA/MQA, windowed and sparse masks, linear attention and ring attention modify dense attention to
  cut KV-cache size, compute, or per-device memory.

## Exercises {#sec:attention:exercises}

::: exercise
[hand computation]{.box-title}

Using the running example, compute by hand (or in five lines of PyTorch) the context vector for the
query "chip" with weight-free attention, and verify it against the last row of
[@fig:attention:heat](a). Then compute its causal context vector.
:::

::: exercise
[scaling]{.box-title}

Draw $\boldsymbol{q},\boldsymbol{k}\in\mathbb{R}^{d_k}$ with i.i.d. standard normal entries for
$d_k\in\{16,64,256,1024\}$ and $T=512$ keys. Plot the average maximum softmax probability with and
without the $1/\sqrt{d_k}$ scale. Relate what you see to the softmax gradient
$\partial\alpha_i/\partial s_j=\alpha_i(\delta_{ij}-\alpha_j)$.
:::

::: exercise
[Parameter vs. Linear]{.box-title}

Implement self-attention with raw `nn.Parameter` matrices used as `x @ W`. Show how to copy weights
between it and [@lst:attention:sa] so that both produce identical outputs.
:::

::: exercise
[no leakage]{.box-title}

Prove algebraically that masking after the softmax followed by renormalisation equals masking with
$-\infty$ before the softmax. Then show numerically that, for route 1, the gradient of position 2's
output with respect to position 5's input is exactly zero.
:::

::: exercise
[head bookkeeping]{.box-title}

GPT-2 medium has $d=1024$ and 16 heads. Give the shapes of every intermediate tensor in
[@lst:attention:mha] for $B=4$, $T=1024$, and the number of parameters in the attention
block (with Q/K/V biases, as in the released weights).
:::

::: exercise
[FLOP crossover]{.box-title}

For a 32-layer model with $d=4096$, $H=32$ and SwiGLU MLP width $11008$, find the sequence length at
which attention-score FLOPs ($\mathbf{Q}\mathbf{K}^\top$ and $\mathbf{A}\mathbf{V}$, with causal
masking exploited) equal all other per-token forward FLOPs.
:::

::: exercise
[online softmax algebra]{.box-title}

Prove by induction that [@eq:attention:online] yields the global $(m,\ell)$. Then show that
two partial states $(m_1,\ell_1,\boldsymbol{o}_1)$ and $(m_2,\ell_2,\boldsymbol{o}_2)$ computed over
*disjoint* key sets can be merged into the state for their union, and write the merge function.
Where does this merge appear in split-KV decoding and in ring attention?
:::

::: exercise
[the $\delta$ identity]{.box-title}

Verify numerically that
$\operatorname{rowsum}(\mathbf{P}\odot\mathbf{dP})=\operatorname{rowsum}(\mathbf{dO}\odot\mathbf{O})$,
then derive the gradient of a row softmax from scratch to see why $\boldsymbol{\delta}$ appears.
:::

::: exercise
[IO model]{.box-title}

Using `llmsys.flash.attention_hbm_bytes`, tabulate naive vs. tiled HBM traffic for
$T\in\{1\text{k},4\text{k},16\text{k},64\text{k}\}$ and $D\in\{64,128,256\}$ with 228 KB of SRAM. At
what $D$ does tiling stop helping in this model? Discuss what the model leaves out.
:::

::: exercise
[skipping masked tiles]{.box-title}

For causal attention with $T=8192$ and square tiles of 128, what fraction of tiles is fully masked,
fully visible, and on the diagonal? How much of the naive $\mathbf{Q}\mathbf{K}^\top$ work does a
tile-skipping kernel avoid?
:::

::: exercise
[sliding window cache]{.box-title}

A model uses a sliding window of $w=4096$ in every layer. Derive the KV-cache size as a function of
generated length and compare to full attention at $T=32$k. What breaks if you want to add prefix
caching across requests?
:::

::: exercise
[measurement]{.box-title}

Run `labs/lab07_attention/bench.py` on a GPU for $T$ from 512 to 16k. Plot time and peak memory of
the explicit implementation and SDPA. Identify the length at which the explicit version runs out of
memory and compare with the napkin estimate of [@sec:attention:cost].
:::

## Lab 5: Attention, from dot products to IO-aware tiling {#sec:attention:lab}

::: labbox
[Attention, from dot products to IO-aware tiling]{.box-title}

**Code:** `code/labs/lab07_attention`

**Goals**

Implement every attention variant of this chapter, prove to yourself (with tests) that the fast
versions compute the same function as the simple ones, and measure where the time and memory go as
the sequence grows.

**Setup**

``` bash
cd code
pytest -q labs/lab07_attention                     # grades starter.py
LAB_IMPL=solution pytest -q labs/lab07_attention   # reference
python labs/lab07_attention/bench.py [--device cuda]
```

**Tasks**

1.  **Weight-free attention.** `simple_attention`: scores, softmax, weighted sum. The test checks
    one row against a hand-written loop.

2.  **Trainable self-attention.** `SelfAttention.forward` with the $1/\sqrt{d_k}$ scale.

3.  **Causal masking.** `causal_softmax` and `mask_after_softmax`; they must agree.
    `CausalAttention.forward` with dropout: the test perturbs future inputs and checks past outputs
    are unchanged, and that dropout is active only in `train()` mode.

4.  **Multi-head.** `MultiHeadAttentionWrapper.forward` and `MultiHeadAttention.forward`. Your
    weight-split module must produce the same output as the core `llmsys.layers.MultiHeadAttention`
    after `load_state_dict`, and with one head it must reduce to `out_proj(CausalAttention(x))`.

5.  **Online softmax.** `online_softmax` for arbitrary block sizes.

6.  **Tiled attention.** `tiled_attention`: never allocate a $T\times T$ tensor; skip key tiles that
    are entirely in the causal future. Must match naive attention to $10^{-10}$ in float64 for
    non-divisible tile sizes.

7.  **Napkin.** `score_matrix_bytes`.

**Measurements**

- Wrapper vs. weight-split MHA latency for GPT-2-small shapes. On a GPU expect the weight-split
  version to win clearly, by more at small batch; on a CPU the gap is smaller.

- Time and peak memory of explicit attention vs. SDPA for $T=256\ldots16$k. Expect explicit peak
  memory to grow as $T^2$ and SDPA's as $T$, and SDPA to be faster beyond a few hundred tokens.

- Your tiled reference is slow (Python loops); measure it only to see that its memory stays flat.

**Deliverables**

All tests passing; plots of time and memory vs. $T$; a paragraph comparing the measured
out-of-memory point to the napkin estimate.

**Stretch goals**

- Implement the tiled *backward* pass yourself and check it with `torch.autograd.gradcheck`; compare
  with `llmsys.flash.FlashAttentionFunction`.

- Add document masks (block-diagonal causal) to `tiled_attention`, skipping tiles that cross
  document boundaries entirely.

- Port `tiled_attention` to Triton following Appendix [-@app:triton].
:::
