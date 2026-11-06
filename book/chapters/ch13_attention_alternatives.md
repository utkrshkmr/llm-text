# Beyond Softmax Attention: Linear Attention, State-Space Models, Hybrids, and Sparse Experts {#ch:attnalt}

Softmax attention has two costs that grow with context. Training does $O(T^2)$ work per layer;
FlashAttention ([@sec:attention:flash]) removes the $O(T^2)$ *memory*, but not the FLOPs.
Decoding reads a KV cache that grows linearly with every token generated, so long-context inference
is bounded by memory capacity and bandwidth ([@ch:inference]). For a decade of chat-length
contexts these costs were tolerable. Reasoning traces of tens of thousands of tokens, agents with
long tool histories, and million-token documents have changed that, and several recent production
models now replace most of their attention layers with something cheaper.

This chapter covers that "something cheaper". We start from the incremental toolkit: windows,
sparsity and compressed caches. We then derive linear attention and show that it has three
equivalent computational forms, add decay and gating to reach the modern gated linear-attention and
Mamba-2 family, add the delta rule, and connect all of it to state-space models and their parallel
scans. We then see how hybrids combine the pieces. A final section revisits mixture-of-experts
layers with the design details that [@sec:modern:moe] left out. Every operator comes with
code in `llmsys/linear_attn.py` and `llmsys/ssm.py` that is verified to be equivalent across its
forms, because for these models the *form* of the computation is the systems story.

![Roadmap of this chapter: from cheap restrictions of softmax attention (top left), through
recurrent sequence mixers, to the hybrid and sparse architectures built from
them.](figures/ch13_attention_alternatives-1.svg){#fig:attnalt:roadmap}

## Why look beyond softmax attention? {#sec:attnalt:why}

::: napkinmath
[The KV cache at long context]{.box-title}

Qwen3-8B has $L=36$ layers and 8 KV heads of dimension 128. Its bf16 KV cache costs
$2\cdot36\cdot8\cdot128\cdot2\,\text{B}=147{,}456$ B per token. At a 128k-token context ($131{,}072$
tokens) a *single* sequence needs $147{,}456\times131{,}072\approx19.3$ GB, more than the 16 GB of
weights. Every decoded token must read all of it, which takes about 5.8 ms at 3.35 TB/s, so
attention alone caps one sequence at under 175 tokens/s before the weights are even counted.
Batching does not help, since each sequence brings its own cache. This is the pressure behind every
technique in this chapter.
:::

There are three broad responses:

1.  **Shrink what is cached** without changing what attention computes: GQA/MQA
    ([@sec:modern:gqa]), multi-head latent attention ([@sec:attnalt:mla]), and KV
    quantisation ([@ch:efficient]).

2.  **Attend to fewer keys**: sliding windows, sinks, block-sparse patterns and learned top-$k$
    selection ([@sec:attnalt:sparse]). The cache can then be bounded or pruned.

3.  **Replace the operator** with one whose state does not grow with $T$: linear attention, its
    gated and delta-rule descendants, and state-space models
    ([@sec:attnalt:linear; @sec:attnalt:gating; @sec:attnalt:delta; @sec:attnalt:ssm]).
    These usually appear in *hybrids* that keep a few softmax layers
    ([@sec:attnalt:hybrid]).

## Windows, sinks, and sparse attention {#sec:attnalt:sparse}

##### Sliding windows.

A sliding-window layer lets query $i$ see only keys $j\in(i-w,i]$, which makes compute $O(Tw)$ and
lets the cache be a ring buffer of $w$ entries. Information still travels further than $w$ because
each layer extends the receptive field by $w$, but precise long-range retrieval suffers. Hence the
now-standard *interleaving*: most layers are local and every $k$-th layer is global. Gemma 2
alternates local (window 4096) and global layers; Gemma 3 uses five local layers per global one;
Cohere reported a 3:1 local:global pattern in which the global layers use no positional encoding at
all, so that RoPE handles short-range order and NoPE layers handle long-range retrieval.

##### Attention sinks.

When a window slides past the first tokens of a sequence, quality can collapse. Xiao et al. (2023)
traced this to the model using the first few tokens as a place to park attention mass: softmax
weights must sum to one, so heads with nothing useful to attend to dump their weight on early
tokens. Keeping those few *sink* tokens permanently in the cache, alongside the window, restores
stable streaming generation. The cache then has a fixed layout of $s$ sink slots followed by a ring
buffer of $w$ recent slots, and the index arithmetic is all a serving engine needs:

``` {#lst:attnalt:sink .python caption="Slot assignment for a sink-plus-window cache of fixed size $s+w$."}
def cache_slot(pos, n_sink=4, window=4092):
    # where token `pos` lives in a cache with n_sink fixed slots + a ring of `window`
    if pos < n_sink:
        return pos                                    # sinks are never evicted
    return n_sink + (pos - n_sink) % window           # ring buffer overwrites the oldest

def live_positions(cur, n_sink=4, window=4092):
    # token positions visible to the query at position `cur`
    start = max(n_sink, cur - window + 1)
    return list(range(min(n_sink, cur + 1))) + list(range(start, cur + 1))
```

StreamingLLM assigns RoPE positions by slot order within the cache rather than by original text
position, so relative distances stay inside the trained range. That is a detail an engine must get
right when it applies rotations at cache-write time.

##### Learned sparsity.

Block-sparse and strided patterns (Child et al., 2019; Longformer's global tokens) fix the pattern
in advance. A newer line *selects* keys per query: a cheap scorer (an "indexer") ranks all past
tokens, and full attention runs only over the top-$k$. DeepSeek reported such a scheme (DeepSeek
Sparse Attention, in V3.2), applied by continued training of a dense model, which turns the
attention cost into $O(Tk)$ plus a much cheaper $O(T^2)$ indexing pass. The full cache is still
stored, but only $k$ entries are read per token.

::: napkinmath
[What does a top-$k$ indexer buy?]{.box-title}

Suppose an indexer with 4 heads of dimension 64, computed in FP8, scores every past token, and full
attention then runs over $k=2048$ selected tokens of a 128-head MLA-style layer. At $T=131{,}072$,
the indexer does about $T\cdot4\cdot64=3.4\times10^7$ multiply-adds per query token and reads
$T\cdot64$ bytes of indexer keys (8 MB). The main attention does
$k\cdot128\cdot(576+512)\approx2.9\times10^8$ multiply-adds against
$2048\cdot576\cdot2\approx2.4$ MB of latent cache. Dense attention at the same length would need
$1.8\times10^{10}$ multiply-adds and 151 MB of reads. The indexer is still $O(T)$ per token, but
with a constant small enough that the sparse layer is dominated by the $O(k)$ term up to very long
contexts. (All dimensions here are illustrative, chosen to be close to published MLA shapes.)
:::

::: systemslens
[Sparsity has to align with tiles]{.box-title}

Kernels skip work at tile granularity, so a pattern is fast only if whole $B_r\times B_c$ tiles can
be skipped ([@fig:attention:patterns]). Sliding windows and block-sparse masks satisfy
this; arbitrary per-query top-$k$ selections do not, and need gather-based kernels whose memory
access is less regular. For decoding, the relevant question is bytes read per step: a window bounds
them by $w$, and top-$k$ selection bounds the attention read at $k$ entries but still pays the
indexer's full scan, at lower precision and width.
:::

## Linear attention and its three forms {#sec:attnalt:linear}

Write causal attention for one head as $\boldsymbol{y}_i=\sum_{j\le i} a_{ij}\boldsymbol{v}_j$ with
$a_{ij}=\kappa(\boldsymbol{q}_i,\boldsymbol{k}_j)/\sum_{j'\le i}\kappa(\boldsymbol{q}_i,\boldsymbol{k}_{j'})$.
Softmax attention uses
$\kappa(\boldsymbol{q},\boldsymbol{k})=\exp(\boldsymbol{q}^\top\boldsymbol{k}/\sqrt d)$. Linear
attention (Katharopoulos et al., 2020) uses a kernel that factorises through a feature map,
$\kappa(\boldsymbol{q},\boldsymbol{k})=
\phi(\boldsymbol{q})^\top\phi(\boldsymbol{k})$; with $\phi(x)=\mathrm{elu}(x)+1$ the features stay
positive. Then $$\begin{equation}
\boldsymbol{y}_i^\top \;=\; \frac{\phi(\boldsymbol{q}_i)^\top\sum_{j\le i}\phi(\boldsymbol{k}_j)\boldsymbol{v}_j^\top}
{\phi(\boldsymbol{q}_i)^\top\sum_{j\le i}\phi(\boldsymbol{k}_j)}
\;=\;\frac{\phi(\boldsymbol{q}_i)^\top\mathbf{S}_i}{\phi(\boldsymbol{q}_i)^\top\boldsymbol{z}_i},
\qquad \mathbf{S}_i=\sum_{j\le i}\phi(\boldsymbol{k}_j)\boldsymbol{v}_j^\top .
\label{eq:attnalt:linear}
\end{equation}$$ The trick is associativity:
$(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}=\mathbf{Q}(\mathbf{K}^\top\mathbf{V})$ when nothing nonlinear
sits between the products. Below we write $\boldsymbol{q},\boldsymbol{k}$ for the feature-mapped
vectors and drop the normaliser, which is just the same computation with $\boldsymbol{v}$ replaced
by 1 (many modern variants drop it in favour of a normalisation layer on the output).

![Without the softmax, the two products can be associated either way. Contracting over time first
produces a $d\times d$ state that does not grow with $T$; causality then turns this into a running
sum over time,
[@eq:attnalt:recurrent].](figures/ch13_attention_alternatives-2.svg){#fig:attnalt:assoc}

##### Form 1: parallel (quadratic).

Compute $\mathbf{Y}=(\mathbf{Q}\mathbf{K}^\top\odot\mathbf{M})\mathbf{V}$ with the causal mask
$\mathbf{M}$. This is ordinary masked attention without the softmax, so it uses matmuls and tensor
cores and costs $O(T^2d)$.

##### Form 2: recurrent.

Causality makes the state a running sum: $$\begin{equation}
\mathbf{S}_t=\mathbf{S}_{t-1}+\boldsymbol{k}_t\boldsymbol{v}_t^\top,\qquad \boldsymbol{y}_t^\top=\boldsymbol{q}_t^\top\mathbf{S}_t .
\label{eq:attnalt:recurrent}
\end{equation}$$ This is an RNN whose hidden state is a $d_k\times d_v$ matrix. Each step costs
$O(d_kd_v)$ regardless of $t$, which makes decoding constant-time and constant-memory. Training this
way is a sequential loop of rank-1 updates, which is hopeless on a GPU.

##### Form 3: chunkwise.

Split the sequence into chunks of $C$ tokens. Within a chunk use the parallel form; across chunks
carry the state. For chunk $c$ with rows $\mathbf{Q}_c,\mathbf{K}_c,\mathbf{V}_c$ and incoming state
$\mathbf{S}$, $$\begin{equation}
\mathbf{Y}_c=\underbrace{(\mathbf{Q}_c\mathbf{K}_c^\top\odot\mathbf{M})\mathbf{V}_c}_{\text{intra-chunk}}
+\underbrace{\mathbf{Q}_c\mathbf{S}}_{\text{inter-chunk}},
\qquad
\mathbf{S}\leftarrow\mathbf{S}+\mathbf{K}_c^\top\mathbf{V}_c .
\label{eq:attnalt:chunk}
\end{equation}$$ All three terms are matmuls of size $C\times d$ or $d\times d$, so tensor cores do
the work, and the sequential dependency is only across $T/C$ chunks. With $C=1$ this is the
recurrent form and with $C=T$ the parallel form; in between, the cost is $O(TCd+Td^2)$.
[@fig:attnalt:chunk] sketches the dataflow, and [@lst:attnalt:chunked] is the
reference implementation, including the decay factors introduced in the next section.

![Chunkwise-parallel linear attention. Each chunk computes its intra-chunk term with a small masked
matmul, reads the incoming state for the inter-chunk term, and passes an updated state on. Kernels
such as those in the flash-linear-attention library keep $\mathbf{S}$ in on-chip memory between
chunks.](figures/ch13_attention_alternatives-3.svg){#fig:attnalt:chunk}

``` {#lst:attnalt:chunked .python caption="Chunkwise linear attention with optional decay (`llmsys/linear_attn.py`)."}
def linear_attention_chunked(q, k, v, gamma=None, normalize=False, chunk=16):
    B, H, T, dk = q.shape
    va = _augment(v, normalize)                   # append ones -> normaliser for free
    log_g = _log_gamma(gamma, B, H, T, q.device, q.dtype)
    S = q.new_zeros(B, H, dk, va.shape[-1])
    outs = []
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        qc, kc, vc, lg = q[..., s:e, :], k[..., s:e, :], va[..., s:e, :], log_g[..., s:e]
        cum = torch.cumsum(lg, dim=-1)            # log decay from chunk start to i
        intra = (torch.einsum("bhik,bhjk->bhij", qc, kc) * decay_mask(lg)) @ vc
        inter = torch.einsum("bhik,bhkv->bhiv", qc * cum.exp()[..., None], S)
        outs.append(intra + inter)
        total = cum[..., -1]                      # decay across the whole chunk
        w = (total[..., None] - cum).exp()        # decay from token j to chunk end
        S = total.exp()[..., None, None] * S \
            + torch.einsum("bhjk,bhjv->bhkv", kc * w[..., None], vc)
    return _maybe_normalize(torch.cat(outs, dim=2), normalize)
```

::: napkinmath
[When is linear attention cheaper?]{.box-title}

Per token and per layer, causal softmax attention does about $2Td$ multiply-adds for scores and
outputs ($d=Hd_{h}$). Chunkwise linear attention does about $Cd$ for the intra-chunk part and
$2dd_{h}$ for reading and updating the per-head states. With $d_{h}=128$ and $C=64$ the linear form
is cheaper once $T\gtrsim (C+2d_{h})/2\approx160$. During decoding the comparison is in bytes:
softmax reads $2Td_{h}$ values per KV head, while linear attention reads a $d_{h}\times d_{h}$ state
per head, equivalent to the KV cache of only $d_{h}/2=64$ tokens per head (or $d_{h}$ tokens when
the heads are not grouped).
:::

##### Choosing $\phi$ and the normaliser.

The feature map decides how well $\phi(\boldsymbol{q})^\top\phi(\boldsymbol{k})$ imitates
$\exp(\boldsymbol{q}^\top\boldsymbol{k})$. Performer (Choromanski et al., 2021) uses positive random
features whose expectation equals the softmax kernel exactly, at the cost of variance and an extra
projection. Simpler deterministic maps include $\mathrm{elu}+1$, ReLU, and low-order Taylor
expansions of $\exp$ (as in the Based architecture), which approximate softmax well near zero and
expand $d_k$ to $O(d_k^2)$ features. The division by $\phi(\boldsymbol{q})^\top\boldsymbol{z}_i$ in
[@eq:attnalt:linear] is numerically awkward: it can approach zero and it couples every
output to a running sum. Most recent designs (RetNet, GLA, Mamba-2, DeltaNet) drop it entirely and
put a normalisation layer (RMSNorm or GroupNorm per head) on the output instead, which is why our
layer makes `normalize` optional.

##### Why plain linear attention underperforms.

Two weaknesses kept linear attention a curiosity for years. First, it never forgets: $\mathbf{S}$ is
a sum of everything ever seen, so old content blurs new retrievals. Second, a $d_k\times d_v$ matrix
can store only about $d_k$ key--value associations without interference, while softmax attention can
retrieve any of $T$ tokens sharply. The next two sections fix the first problem with decay and
gating and attack the second with the delta rule. The capacity limit remains, which is the reason
hybrids keep some softmax layers.

## Decay and gating: RetNet, GLA, and Mamba-2 {#sec:attnalt:gating}

Add a forgetting factor to [@eq:attnalt:recurrent]: $$\begin{equation}
\mathbf{S}_t=\gamma_t\,\mathbf{S}_{t-1}+\boldsymbol{k}_t\boldsymbol{v}_t^\top,\qquad\boldsymbol{y}_t^\top=\boldsymbol{q}_t^\top\mathbf{S}_t,
\qquad 0<\gamma_t\le1 .
\label{eq:attnalt:gated}
\end{equation}$$ Unrolling gives
$\boldsymbol{y}_i^\top=\sum_{j\le i}\big(\prod_{l=j+1}^{i}\gamma_l\big)(\boldsymbol{q}_i^\top\boldsymbol{k}_j)\boldsymbol{v}_j^\top$:
the parallel form is masked attention with a *decay mask* $D_{ij}=\prod_{l=j+1}^i\gamma_l$ in place
of the 0/1 causal mask. The chunked form needs only cumulative products of $\gamma$ within each
chunk, computed stably as cumulative sums of $\log\gamma$ ([@lst:attnalt:chunked]). The
variants differ in where $\gamma$ comes from:

- **RetNet** (Sun et al., 2023): a fixed per-head constant, $\gamma_h=1-2^{-5-h}$, so different
  heads have different memory horizons.

- **Gated linear attention** (GLA; Yang et al., 2024) and **Mamba-2** (Dao and Gu, 2024): $\gamma_t$
  is computed from the input, so the model decides when to forget, for example at a document
  boundary. Mamba-2 uses a scalar per head and per step. Its "state space duality" (SSD) shows that
  the selective SSM of [@sec:attnalt:ssm] with scalar decay is exactly this structured
  masked attention, and derives the chunked algorithm from that equivalence.

``` {#lst:attnalt:layer .python caption="A gated linear-attention layer with a constant-memory decoding step."}
class LinearAttention(nn.Module):
    def __init__(self, d_model, n_heads, mode="gated", chunk=16):
        super().__init__()
        self.H, self.dh, self.chunk = n_heads, d_model // n_heads, chunk
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model, n_heads)          # one forget gate per head/step
        nn.init.constant_(self.gate.bias, 3.0)           # sigmoid(3) = 0.95: slow decay

    def forward(self, x):                                # training: chunkwise parallel
        q, k, v, g = self._proj(x)                       # g = sigmoid(gate(x)), (B,H,T)
        y = linear_attention_chunked(q, k, v, g, normalize=True, chunk=self.chunk)
        return self.out(merge_heads(y))

    @torch.no_grad()
    def step(self, x_t, state=None):                     # decoding: O(1) per token
        q, k, v, g = self._proj(x_t)
        y, state = linear_attention_recurrent(q, k, v, g, True, state)
        return self.out(merge_heads(y)), state
```

The test suite of Lab 8 checks that the parallel, recurrent and chunked forms agree to $10^{-10}$ in
float64 for every chunk size and every kind of decay, and that `step` reproduces `forward` token by
token. This kind of equivalence test is how you should develop any new sequence mixer: write the
slow, obviously correct form first, then the fast one, and compare them.

::: systemslens
[Choosing the chunk size]{.box-title}

The chunk size $C$ trades the $O(TCd)$ intra-chunk work, which is matmul-friendly, against $T/C$
sequential state updates, each reading and writing a $d_k\times d_v$ state per head. On GPUs $C=64$
or 128 matches tensor-core tiles, and production kernels keep the state in shared memory or
registers across chunks so that only $\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{Y}$ touch HBM. The
decay mask within a chunk involves $\exp$ of differences of cumulative log-gates, so kernels compute
it in fp32. Products of many $\gamma$s underflow quickly, and a careless implementation divides by
them, which produces NaNs. The reference code only ever exponentiates *non-positive* differences.
:::

## The delta rule and gated DeltaNet {#sec:attnalt:delta}

Viewed as memory, $\mathbf{S}$ is an associative store: writing $\boldsymbol{k}\boldsymbol{v}^\top$
files value $\boldsymbol{v}$ under key $\boldsymbol{k}$, and reading with
$\boldsymbol{q}=\boldsymbol{k}$ returns $\mathbf{S}^\top\boldsymbol{k}$. With additive writes,
re-filing a key *adds* the new value to the old one. The delta rule (Widrow--Hoff, adopted for
fast-weight transformers by Schlag et al., 2021) instead writes only the *error*: $$\begin{equation}
\mathbf{S}_t=\mathbf{S}_{t-1}+\beta_t\,\boldsymbol{k}_t\big(\boldsymbol{v}_t-\mathbf{S}_{t-1}^\top\boldsymbol{k}_t\big)^\top
=\big(\mathbf{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^\top\big)\mathbf{S}_{t-1}+\beta_t\boldsymbol{k}_t\boldsymbol{v}_t^\top .
\label{eq:attnalt:delta}
\end{equation}$$ With a unit-norm key and $\beta_t=1$, the update replaces whatever was stored under
$\boldsymbol{k}_t$ by $\boldsymbol{v}_t$ exactly: it is one step of gradient descent with learning
rate $\beta_t$ on the regression loss
$\frac12\|\mathbf{S}^\top\boldsymbol{k}_t-\boldsymbol{v}_t\|^2$. This view connects linear recurrent
models to *test-time training*, where a model's state is a set of weights updated by an inner
optimiser as it reads. Gated DeltaNet (Yang et al., 2025) multiplies the whole update by a forget
gate $\gamma_t$, combining the two mechanisms:
$$\mathbf{S}_t=\gamma_t\big(\mathbf{I}-\beta_t\boldsymbol{k}_t\boldsymbol{k}_t^\top\big)\mathbf{S}_{t-1}+\gamma_t\beta_t\boldsymbol{k}_t\boldsymbol{v}_t^\top .$$
Qwen's recent "Next" models are reported to interleave gated-DeltaNet layers with full attention at
roughly 3:1.

The transition matrix $\mathbf{I}-\beta\boldsymbol{k}\boldsymbol{k}^\top$ is no longer a scalar, so
the simple decay-mask trick does not apply. The chunked algorithm instead represents a product of
such Householder-like matrices compactly with the WY representation from numerical linear algebra,
which again reduces the work to matmuls (Yang et al., 2024). The lab implements the recurrent step,
and the test demonstrates the overwrite property directly.

``` {#lst:attnalt:delta .python caption="One delta-rule step ([@lab:attnalt], Task 4)."}
def delta_rule_step(S, q_t, k_t, v_t, beta_t, g_t=None):
    pred = torch.einsum("bhk,bhkv->bhv", k_t, S)          # what memory returns for k_t
    S = S + beta_t[..., None, None] * k_t[..., :, None] * (v_t - pred)[..., None, :]
    if g_t is not None:
        S = g_t[..., None, None] * S                      # gated DeltaNet: forget
    return torch.einsum("bhk,bhkv->bhv", q_t, S), S
```

## State-space models and parallel scans {#sec:attnalt:ssm}

State-space models arrive at the same place from signal processing. A continuous linear system maps
an input signal $x(t)$ to an output $y(t)$ through a hidden state $h(t)\in\mathbb{R}^N$:
$$h'(t)=\mathbf{A}h(t)+\mathbf{B}x(t),\qquad y(t)=\mathbf{C}h(t)+Dx(t).$$ Discretising with step
size $\Delta$ (zero-order hold for $\mathbf{A}$, a first-order approximation for $\mathbf{B}$) gives
the recurrence $$\begin{equation}
h_t=\bar{\mathbf{A}}_t\,h_{t-1}+\bar{\mathbf{B}}_t\,x_t,\qquad \bar{\mathbf{A}}_t=\exp(\Delta_t\mathbf{A}),\quad
\bar{\mathbf{B}}_t=\Delta_t\mathbf{B}_t,\qquad y_t=\mathbf{C}_th_t+Dx_t .
\label{eq:attnalt:ssm}
\end{equation}$$

##### S4 to Mamba.

In S4 (Gu et al., 2022) the parameters are time-invariant, so the whole map is a long convolution
that FFTs compute in parallel, with careful structure on $\mathbf{A}$ for long memory. Such LTI
models cannot *select*: they treat every token identically, so they cannot ignore filler or latch
onto a key. Mamba (Gu and Dao, 2023) makes $\Delta_t$, $\mathbf{B}_t$ and $\mathbf{C}_t$ functions
of the input. A large $\Delta_t$ resets the state toward the current input and a small one preserves
the past, so the model can choose what to remember, much like the gates above. With diagonal
$\mathbf{A}$, each of the $E$ channels runs $N$ independent scalar recurrences.

##### The scan.

Selectivity destroys the convolution view, but [@eq:attnalt:ssm] is still a *first-order
linear recurrence* $h_t=a_th_{t-1}+b_t$, and such recurrences compose associatively. Represent step
$t$ by the pair $(a_t,b_t)$ meaning "$h\mapsto a_th+b_t$". Applying step $s$ then step $t$ gives
$h\mapsto a_t(a_sh+b_s)+b_t$, so $$\begin{equation}
(a_s,b_s)\bullet(a_t,b_t)=(a_sa_t,\;a_tb_s+b_t),
\label{eq:attnalt:scanop}
\end{equation}$$ and $\bullet$ is associative. Any associative operator admits a *parallel prefix
scan*: all $T$ prefix results in $O(\log T)$ parallel rounds (Hillis--Steele, $O(T\log T)$ work) or
with $O(T)$ work in two sweeps (Blelloch). [@fig:attnalt:scan] shows the Hillis--Steele
pattern.

![Hillis--Steele inclusive scan over 8 steps. In round $r$, element $t$ combines with element
$t-2^r$ using [@eq:attnalt:scanop]; after $\lceil\log_2 T\rceil$ rounds element $t$ holds
the composition of steps $0..t$,
i.e. $h_t$.](figures/ch13_attention_alternatives-4.svg){#fig:attnalt:scan}

``` {#lst:attnalt:scan .python caption="Parallel scan for $h_t=a_th_{t-1}+b_t$ (`llmsys/ssm.py`)."}
def scan_parallel(a, b):
    a, b = a.clone(), b.clone()
    T, step = a.shape[1], 1
    while step < T:                               # ceil(log2 T) rounds
        a_prev = torch.ones_like(a);  a_prev[:, step:] = a[:, :-step]
        b_prev = torch.zeros_like(b); b_prev[:, step:] = b[:, :-step]
        b = a * b_prev + b                        # (a_prev, b_prev) . (a, b)
        a = a * a_prev
        step *= 2
    return b                                      # b[:, t] == h_t
```

##### The Mamba block.

`MambaBlock` in `llmsys/ssm.py` follows the published design: an input projection to two streams, a
short causal depthwise convolution (width 4) on one, SiLU, the selective SSM, multiplicative gating
by SiLU of the other stream, and an output projection. Its `step` method carries two pieces of
state, the last three conv inputs and $h\in\mathbb{R}^{E\times N}$, and reproduces the parallel
forward pass token by token.

##### Discretisation in detail.

For a single diagonal entry $a<0$ of $\mathbf{A}$ and an input held constant over an interval of
length $\Delta$ (zero-order hold), the exact solution of $h'=ah+bx$ is
$$h(t+\Delta)=e^{\Delta a}h(t)+\frac{e^{\Delta a}-1}{a}\,b\,x(t).$$ The first factor gives
$\bar{a}=\exp(\Delta a)\in(0,1)$, a decay just like $\gamma$ in [@eq:attnalt:gated]. For
small $\Delta a$, $(e^{\Delta a}-1)/a\approx\Delta$, which is the simpler $\bar{b}=\Delta b$ that
Mamba's implementation uses. Because $a<0$ and $\Delta>0$ (a softplus output), the decay always lies
in $(0,1)$ and the recurrence is stable by construction, which is why $\mathbf{A}$ is parameterised
as $-\exp(\mathbf{A}_{\log})$. When $\mathbf{A},\mathbf{B},\mathbf{C},\Delta$ do not depend on $t$
(S4), unrolling gives $y=\mathbf{K}*x$ with the convolution kernel
$\mathbf{K}=(\mathbf{C}\bar{\mathbf{B}},\,\mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}},\,\mathbf{C}\bar{\mathbf{A}}^2\bar{\mathbf{B}},\ldots)$,
which FFTs compute in $O(T\log T)$. Selectivity makes $\bar{\mathbf{A}}_t$ time-varying and removes
this option.

![The Mamba block as implemented in `llmsys/ssm.py`. Two pieces of state are carried during
decoding, both of fixed size. The short convolution gives each token a little local context before
the recurrence; the gate plays the role of a
GLU.](figures/ch13_attention_alternatives-5.svg){#fig:attnalt:mamba}

``` {#lst:attnalt:mamba .python caption="The selective scan and the Mamba block's decode step (`llmsys/ssm.py`, abridged)."}
def selective_scan(x, dt, A, Bm, Cm, D, parallel=True, h0=None):
    dA = torch.exp(dt[..., None] * A)                        # (B,T,E,N) decay
    dBx = dt[..., None] * Bm[:, :, None, :] * x[..., None]   # (B,T,E,N) input
    h = scan_parallel(dA, dBx) if parallel and h0 is None \
        else scan_sequential(dA, dBx, h0)
    y = torch.einsum("bten,btn->bte", h, Cm) + D * x
    return y, h[:, -1]

@torch.no_grad()
def step(self, u_t, state):                                  # u_t: (B, 1, d_model)
    conv_s, h = state                                        # (B,E,3), (B,E,N)
    x, z = self.in_proj(u_t).chunk(2, dim=-1)
    window = torch.cat([conv_s, x.transpose(1, 2)], dim=-1)  # last 4 inputs
    xc = F.silu((window * self.conv.weight.squeeze(1)).sum(-1) + self.conv.bias)[:, None]
    dt, Bm, Cm = self._ssm_params(xc)
    y, h = selective_scan(xc, dt, -torch.exp(self.A_log), Bm, Cm, self.D,
                          parallel=False, h0=h)
    return self.out_proj(y * F.silu(z)), (window[..., 1:], h)
```

::: systemslens
[Hardware-aware scans]{.box-title}

The state $h$ has $E\cdot N$ entries per token position; materialising it for every $t$ in HBM, as
our reference does, costs $T\cdot E\cdot N$ values, far more than the activations themselves.
Mamba's kernel avoids this: it loads $\Delta,\mathbf{A},\mathbf{B},\mathbf{C},x$ for a block into
SRAM, runs the discretisation and the scan there, writes only $y$, and *recomputes* the states in
the backward pass. This is the same IO-aware philosophy as FlashAttention
([@sec:attention:flash]). Mamba-2's SSD reformulation goes further by turning most of the
work into chunked matmuls ([@eq:attnalt:chunk] with decay), because a scan of elementwise
operations cannot use tensor cores and a matmul can.
:::

## Hybrid architectures {#sec:attnalt:hybrid}

Fixed-size states are cheap but lossy; softmax attention is expensive but can retrieve any token
exactly. Pure recurrent models struggle on tasks that need verbatim recall from far back, such as
copying, associative recall and needle-in-a-haystack retrieval. Hybrids keep a few attention layers
for retrieval and use cheap mixers everywhere else. Reported designs include Jamba (one attention
layer per seven Mamba layers, with MoE MLPs), MiniMax-01 (seven "lightning" linear-attention layers
per softmax layer), NVIDIA's Nemotron-H family (mostly Mamba-2 with a few attention layers), and
Qwen's "Next" models (gated-DeltaNet and attention at about 3:1). Controlled published ablations are
still scarce, but a consistent finding is that a small fraction of attention layers recovers most of
the gap to a full-attention model.

![Decoding memory for one sequence of a Qwen3-8B-shaped stack (36 layers, 32 query heads, 8 KV
heads, $d_{h}=128$, bf16) as the sequence mixer changes. The hybrid's cache still grows linearly but
four times more slowly; the pure recurrent state is constant at 37.7 MB, the KV cache of 256
tokens.](figures/ch13_attention_alternatives-6.svg){#fig:attnalt:cache}

##### Why some attention is needed: associative recall.

The cleanest diagnostic is *multi-query associative recall* (MQAR; Arora et al., 2023). The input
lists key--value pairs, and later queries repeat some of the keys; the model must output the
matching values ([@fig:attnalt:mqar]). Softmax attention solves it with one layer at any
number of pairs, because it can look up the exact earlier token. A fixed-size state must *compress*
all pairs into $d_k\times d_v$ numbers, so accuracy falls once the number of pairs exceeds the
state's capacity. Accuracy on MQAR across state sizes tracked language-modelling quality gaps
between attention and recurrent models remarkably well, and it is the quickest test to run on a new
mixer.

![Multi-query associative recall. Attention can point back at the exact earlier token (arrows); a
recurrent model must have stored the pair in its fixed-size
state.](figures/ch13_attention_alternatives-7.svg){#fig:attnalt:mqar}

We ran this probe with `labs/lab13_attnalt/recall.py`: two-layer `HybridLM`s with $d=64$, a 64-token
vocabulary, 8 key--value pairs and 4 queries per sequence, trained for 2,500 steps (one seed,
measured on CPU in this environment). [@tab:attnalt:mqar] shows the result. The pure
attention model switches from guessing to perfect recall abruptly at around 1,000 steps, which is
the familiar signature of an induction-style circuit forming. The two pure recurrent models never
get far above chance in the same budget, and a single attention layer on top of a linear layer is
enough to recover recall almost completely.

::: {#tab:attnalt:mqar .table-float}
| Pattern | Layers                     | Recall accuracy |
|:--------|:---------------------------|----------------:|
| `AA`    | softmax, softmax           |           1.000 |
| `LA`    | gated linear, softmax      |           0.998 |
| `MM`    | Mamba, Mamba               |           0.176 |
| `LL`    | gated linear, gated linear |           0.126 |
| chance  |                            |           0.031 |

: Associative recall after 2,500 steps (8 pairs, 4 queries, $d=64$; one seed; CPU). Recurrent states
of this size cannot hold the pairs well enough; one attention layer fixes it. Larger states and
longer training help the recurrent models, so treat this as a qualitative illustration, not a
ranking.
:::

::: systemslens
[Training recurrent layers on long sequences across GPUs]{.box-title}

For softmax attention, context parallelism shards the sequence and circulates KV blocks around a
ring ([@ch:distributed]). Recurrent layers have a simpler but more *sequential* structure:
device $r$ needs the state at the end of device $r-1$'s shard. One approach computes each shard's
local outputs and its local state contribution in parallel and then fixes up the cross-shard term
once the incoming states are known. Because the state transition is linear, the incoming states can
themselves be obtained with a scan over devices: each device sends a single $d_k\times d_v$ matrix
(plus a decay scalar) per head, which is far less traffic than shipping KV blocks.
:::

::: systemslens
[Serving a hybrid]{.box-title}

Hybrids complicate the serving engine of [@ch:serving]. Attention layers want a paged KV
cache, while recurrent layers want one fixed-size state slot per sequence, which needs no paging but
must be copied for beam search or parallel sampling and cannot be partially reused. Prefix caching
becomes harder because a recurrent state summarises an *entire* prefix: to reuse a shared prompt you
must have checkpointed the state exactly at the prefix boundary, not just kept per-token KV blocks.
Speculative decoding needs the state to roll back on rejection, so engines snapshot states per draft
position. Chunked prefill works naturally, since the chunkwise algorithm is chunked prefill. These
are solvable problems, but they are why support for new mixers in inference engines usually lags the
model releases.
:::

## Setting up a hybrid model end to end {#sec:attnalt:setup}

New sequence mixers fail in practice far more often through plumbing than through modelling. The
workflow that keeps them correct is the same one used throughout this chapter:

1.  **Write the obviously correct form first**: the recurrent loop, or the quadratic masked form, in
    float64, with no cleverness.

2.  **Write the fast form** (chunked, scan-based, or a fused kernel) and compare it against the slow
    one on random inputs, over many shapes, chunk sizes and gate regimes, including gates near 0 and
    near 1.

3.  **Write the decode step** and check that feeding a sequence one token at a time through `step`
    reproduces `forward` at every position. This catches off-by-one errors in conv windows, RoPE
    offsets and state updates.

4.  **Only then integrate** the mixer into a model, a training loop and a serving path.

``` {#lst:attnalt:test .python caption="The equivalence-test pattern used for every mixer in this chapter (from [@lab:attnalt])."}
@pytest.mark.parametrize("gamma", [None, per_head_decay, per_step_gates])
def test_three_forms_agree(gamma):
    q, k, v = random_qkv(dtype=torch.float64)            # float64: compare to 1e-10
    ref = linear_attention_quadratic(q, k, v, gamma)      # slow, obviously correct
    rec, _ = linear_attention_recurrent(q, k, v, gamma)   # decode path
    assert torch.allclose(ref, rec, atol=1e-10)
    for chunk in (1, 4, 7, 64):                           # include ragged last chunks
        assert torch.allclose(ref, linear_attention_chunked(q, k, v, gamma, chunk=chunk),
                              atol=1e-10)

def test_step_matches_forward():
    layer = LinearAttention(16, 2).eval()
    x = torch.randn(2, 20, 16)
    full, state, outs = layer(x), None, []
    for t in range(x.shape[1]):                           # one token at a time
        y, state = layer.step(x[:, t:t + 1], state)
        outs.append(y)
    assert torch.allclose(torch.cat(outs, 1), full, atol=1e-5)
```

Two details are easy to miss. Test in float64 so that a tolerance of $10^{-10}$ separates bugs from
rounding, and include chunk sizes that do not divide $T$, because ragged last chunks are where index
arithmetic breaks.

`HybridLM` in `llmsys/linear_attn.py` is a small but complete example of step 4. A pattern string
such as `"LLLA"` chooses the mixer for each layer: gated linear attention (L), Mamba (M) or softmax
attention with RoPE and a KV cache (A). Each layer is a pre-norm residual block with an MLP. The
same weights serve a parallel `forward` for training and a `decode_step` that threads a
heterogeneous list of per-layer states through the network, as in [@lst:attnalt:hybrid].

``` {#lst:attnalt:hybrid .python caption="Decoding a hybrid model: one state object per layer, of a type that depends on the layer."}
@torch.no_grad()
def decode_step(self, ids_t, states):          # ids_t: (B, 1); states: list per layer
    x = self.emb(ids_t)
    new = []
    for p, mix, n1, n2, mlp, st in zip(self.pattern, self.mixers, self.n1,
                                       self.n2, self.mlps, states):
        h = n1(x)
        if p == "A":                            # KV cache: grows by one entry per step
            y, st = mix(h, st)
        elif p == "L":                          # (B, H, dk, dv+1) matrix state, fixed
            y, st = mix.step(h, st)
        else:                                   # Mamba: (conv window, SSM state), fixed
            st = st if st is not None else mix.init_state(ids_t.shape[0], ids_t.device)
            y, st = mix.step(h, st)
        new.append(st)
        x = x + y
        x = x + mlp(n2(x))
    return self.head(self.norm(x)), new

model = HybridLM(pattern="LMLA").eval()
states = [None] * len(model.pattern)
for t in range(ids.shape[1]):                   # equals model(ids) position by position
    logits_t, states = model.decode_step(ids[:, t:t + 1], states)
print(model.cache_numel(states))                # only the 'A' entry grows with t
```

![The decode-time state of a hybrid stack. Only the attention layers hold memory that grows with the
generated length; the serving engine must manage paged blocks for them and fixed slots for the
others.](figures/ch13_attention_alternatives-8.svg){#fig:attnalt:hybridstate}

To train it, use any loop from [@ch:pretrain] or `llmsys.ablate.train_lm`'s structure: the
model's `forward(ids)` returns logits like every other model in the book. To profile it, time
`forward` at increasing $T$ for patterns `"AAAA"`, `"LLLA"` and `"LLLL"`: the quadratic term appears
only in proportion to the number of A layers. To serve it, the engine of [@ch:serving] needs
a second kind of cache slot, as discussed in the Systems Lens above. Lab 8's stretch goals walk
through each of these steps. [@fig:attnalt:flops] shows what the profile should look like
analytically.

![Analytic per-token cost of the sequence mixer for a $d=4096$, $d_{h}=128$ layer, averaged over a
causal prefix of length $T$, with the MLP's fixed cost for comparison. Softmax attention overtakes
the MLP near $T=18$k tokens; a 1:3 hybrid pushes that point four times further out; a pure linear
mixer stays constant and small.](figures/ch13_attention_alternatives-9.svg){#fig:attnalt:flops}

::: deepdive
[Test-time training and learned memories]{.box-title}

The delta rule is one step of SGD on a regression loss whose "weights" are the state. Generalising
that view gives *test-time training* layers, in which the state is the parameters of a small model
(linear or an MLP) trained online, as the sequence is read, by an inner optimiser with its own
learning rate, momentum or weight decay. Recent proposals in this family (TTT layers; Titans, with a
neural long-term memory) differ in the inner model and update rule, and they inherit the systems
questions of this chapter: how to chunk the inner updates so that tensor cores do the work, and how
large a state each sequence must carry at decode time.
:::

## Multi-head latent attention, revisited {#sec:attnalt:mla}

[@ch:modern] introduced MLA as "caching a compressed latent". Here are the details that make
it work. For each token, MLA projects the hidden state down to a small latent
$\boldsymbol{c}_t=\mathbf{W}^{DKV}\boldsymbol{h}_t\in\mathbb{R}^{d_c}$ and caches only that. Keys
and values for head $i$ are reconstructed by up-projections:
$\boldsymbol{k}_t^{(i)}=\mathbf{W}^{UK}_i\boldsymbol{c}_t$,
$\boldsymbol{v}_t^{(i)}=\mathbf{W}^{UV}_i\boldsymbol{c}_t$.

##### Absorption.

At decode time nobody actually materialises the keys. The score
$\boldsymbol{q}^{(i)\top}\boldsymbol{k}_t^{(i)}=(\mathbf{W}^{UK\top}_i\boldsymbol{q}^{(i)})^\top\boldsymbol{c}_t$
can be computed against the latent directly by folding $\mathbf{W}^{UK}_i$ into the query
projection; similarly, $\mathbf{W}^{UV}_i$ folds into the output projection. Attention then runs
over $d_c$-dimensional cached vectors shared by all heads, much like MQA with a wide head.

##### The RoPE problem.

Absorption requires the score to be bilinear in fixed matrices, but RoPE inserts a
position-dependent rotation $\mathbf{R}_{t}$ between $\boldsymbol{q}$ and $\boldsymbol{k}$, which
cannot be folded into a fixed $\mathbf{W}^{UK}$. DeepSeek-V2's fix is *decoupled RoPE*: a small
extra key component $\boldsymbol{k}^{R}_t\in\mathbb{R}^{d_R}$, shared across heads and rotated
normally, is cached alongside $\boldsymbol{c}_t$, and the score becomes the sum of an absorbed
no-position part and a small rotated part.

::: napkinmath
[MLA cache size]{.box-title}

DeepSeek-V2's published configuration uses 128 heads of key dimension 128 (plus 64 decoupled RoPE
dimensions), $d_c=512$ and $d_R=64$. Plain MHA would cache $2\times128\times128=32{,}768$ values per
token per layer; MLA caches $512+64=576$, which is 57 times less and comparable to GQA with only
about two KV heads. Because the absorbed decode attention reads one 576-wide vector per token for
all 128 heads, its arithmetic intensity per byte of cache is much higher than GQA's, which makes
long-context decoding far less memory-bound.
:::

##### Multi-token prediction.

DeepSeek-V3 also trains lightweight extra heads that predict one token further ahead (MTP). It is a
training signal, and at inference time the extra head can serve as a built-in draft model for
speculative decoding ([@ch:efficient]).

## Sparse experts, revisited {#sec:attnalt:moe}

[@sec:modern:moe] implemented a top-$k$ MoE layer with an auxiliary balancing loss and
expert parallelism. Current MoE designs add several refinements.

##### Fine-grained and shared experts.

DeepSeekMoE splits each expert into $m$ smaller ones and routes to $m$ times as many, which keeps
FLOPs fixed while making the combinatorics of expert selection far richer. It also adds a few
*shared* experts that every token uses, so common knowledge need not be replicated across routed
experts. [@tab:attnalt:moe] lists published routing setups.

::: {#tab:attnalt:moe .table-float}
| Model | routed experts | active per token | shared | notes |
|:---|---:|---:|---:|:---|
| Switch Transformer | up to 2048 | 1 | 0 | top-1 routing |
| Mixtral 8x7B | 8 | 2 | 0 | 47B total, 13B active |
| OLMoE | 64 | 8 | 0 | fine-grained |
| DeepSeek-V2 | 160 | 6 | 2 | fine-grained + shared |
| DeepSeek-V3 | 256 | 8 | 1 | aux-loss-free balancing, 671B total / 37B active |
| Qwen3-30B-A3B | 128 | 8 | 0 | see [@sec:modern:moe] |

: Expert configurations of some published MoE models (per MoE layer).
:::

![Fine-grained experts with a shared expert (right) activate the same number of MLP FLOPs as a
coarse top-2 design (left) but choose among $\binom{32}{8}\approx10^7$ combinations instead of
$\binom{8}{2}=28$. The always-on shared expert (S) holds common
knowledge.](figures/ch13_attention_alternatives-10.svg){#fig:attnalt:moe}

##### Keeping routers stable.

Routers are small softmax classifiers, and their logits are prone to the same growth as LM logits.
ST-MoE (Zoph et al., 2022) computes the router in fp32 and adds a *router z-loss* on
$\log\sum_e\exp(r_e)$, analogous to [@eq:archdesign:zloss]. Older recipes add
multiplicative jitter noise to router inputs to make experts less brittle.

##### Balancing without an auxiliary loss.

The auxiliary loss of [@eq:modern:aux] trades quality for balance. DeepSeek-V3 instead adds
a per-expert bias $b_e$ to the routing scores used only for *selection* (not for the combination
weights) and nudges it after each step: decrease $b_e$ if expert $e$ was overloaded and increase it
if underloaded. This is an online control loop that keeps loads balanced with only a tiny
sequence-level auxiliary term.

##### Capacity, dropping, and nondeterminism.

With a fixed capacity per expert ([@sec:modern:moe]), tokens beyond capacity are dropped
and pass through on the residual stream only. Which tokens are dropped depends on everything else in
the batch, so in a serving system *another user's request can change your output*. Dropless
implementations such as MegaBlocks use block-sparse or grouped GEMMs to process variable-size expert
batches without capacity limits.

::: systemslens
[Dispatching tokens to experts]{.box-title}

An MoE layer's runtime is dominated by data movement and load imbalance, not by FLOPs. With expert
parallelism ([@ch:distributed]) each token's hidden vector is sent to the devices holding its
$k$ experts and the results are sent back: two all-to-all exchanges of $k\cdot d$ values per token
per MoE layer. For DeepSeek-V3-like shapes ($k=8$, $d=7168$, bf16) that is about 115 KB per token
each way, which is why it restricts each token to experts on at most four nodes and overlaps
communication with computation. On each device, the tokens routed to its experts form groups of
different sizes; a *grouped GEMM* (one kernel launch over many small matrices of varying row counts)
or block-sparse formulation, as in MegaBlocks, processes them without padding to a capacity. The
slowest expert sets the step time, so balance is a throughput problem as much as a modelling one.
:::

##### Upcycling and fine-tuning.

An MoE can be initialised from a dense checkpoint by copying its MLP into every expert and adding a
fresh router ("sparse upcycling", Komatsuzaki et al., 2023). Several released MoEs were reportedly
built this way. MoEs fine-tuned on small datasets can overfit, which some recipes mitigate by
fine-tuning only non-expert parameters or by using much larger SFT sets.

## Choosing a sequence mixer {#sec:attnalt:choose}

::: {#tab:attnalt:choose .table-float}
| Mixer | Training cost | Decode state | Exact long-range recall | Ecosystem |
|:---|:---|:---|:---|:---|
| Softmax (GQA) | $O(T^2)$ FLOPs, IO-optimal kernels | grows with $T$ | strong | mature everywhere |
| Sliding window | $O(Tw)$ | bounded by $w$ | only within window | mature |
| MLA | $O(T^2)$ | grows, $\sim$`<!-- -->`{=html}57$\times$ smaller | strong | engine support growing |
| Linear / GLA / Mamba-2 | $O(TCd)$, chunked matmuls | constant | weak (capacity-limited) | specialised kernels |
| Gated DeltaNet | chunked, WY form | constant | better than additive | specialised kernels |
| Hybrid (1:3--1:7) | mostly linear | small + constant | close to full attention | needs dual cache management |

: A qualitative comparison of sequence mixers. "Exact recall" refers to tasks such as associative
recall and needle retrieval; the hybrid entry summarises reported results, not a controlled
comparison.
:::

[@tab:attnalt:choose] summarises the trade-offs. For a new model whose deployment is
dominated by long generations, a hybrid with a small fraction of full-attention layers is now a
credible default. For a model that must be served everywhere tomorrow, plain GQA attention with
sliding-window layers remains the lowest-risk choice, because every kernel library and inference
engine supports it.

## Summary {#sec:attnalt:summary}

- Long-context cost is dominated by the KV cache at decode time; shrinking it (GQA, MLA,
  quantisation), bounding it (windows, sinks) or replacing it (recurrent states) are the three
  strategies.

- Removing the softmax makes attention associative: the same operator has parallel ($O(T^2)$),
  recurrent ($O(T)$, constant state) and chunkwise forms. Training uses the chunked form and
  decoding the recurrent one.

- Decay ($\gamma$) adds forgetting: constant per head (RetNet) or data-dependent (GLA, Mamba-2). It
  turns the causal mask into a decay mask.

- The delta rule writes errors instead of sums, giving overwrite semantics and a test-time-training
  interpretation; gated DeltaNet adds forgetting.

- Selective SSMs are first-order linear recurrences, parallelisable with associative scans, and run
  efficiently only with IO-aware kernels or chunked-matmul reformulations.

- Hybrids keep a few attention layers for exact retrieval; they complicate caching, prefix reuse and
  speculative decoding in serving engines.

- Modern MoEs use fine-grained and shared experts, fp32 routers with z-loss, bias-based balancing
  and dropless kernels; capacity-based dropping makes outputs batch-dependent.

## Exercises {#sec:attnalt:exercises}

::: exercise
[Normaliser for free]{.box-title}

Show that appending a column of ones to $\mathbf{V}$ in any of the three linear-attention forms
yields the denominator of [@eq:attnalt:linear] in the last output column. Why can this
denominator approach zero with an unbounded feature map, and why does $\mathrm{elu}+1$ avoid sign
problems but not small values?
:::

::: exercise
[Chunked derivation]{.box-title}

Starting from [@eq:attnalt:gated], derive the chunked form with decay: express
$\boldsymbol{y}_i$ for $i$ in chunk $c$ in terms of the state at the end of chunk $c-1$ and the
tokens of chunk $c$, and give the state update. Verify that your expressions match
[@lst:attnalt:chunked].
:::

::: exercise
[Crossover]{.box-title}

For a model with $d=4096$, 32 heads of $d_{h}=128$ and chunk size 64, plot per-token forward FLOPs
of causal softmax attention and chunked linear attention against $T$. At $T=32{,}768$, what is the
ratio?
:::

::: exercise
[Capacity of a matrix memory]{.box-title}

Write $n$ random unit keys in $\mathbb{R}^{d}$ with additive updates
$\mathbf{S}=\sum_i\boldsymbol{k}_i\boldsymbol{v}_i^\top$ and retrieve with
$\boldsymbol{q}=\boldsymbol{k}_j$. Derive the signal-to-interference ratio as a function of $n/d$.
Measure it with `llmsys.linear_attn`, then repeat with the delta rule ($\beta=1$, sequential writes)
and explain the difference.
:::

::: exercise
[Scan work and depth]{.box-title}

Count the combine operations and parallel rounds used by Hillis--Steele and by Blelloch scans for
$T=2^{16}$. Which would you choose on a GPU with 132 SMs for a single sequence? For a batch of 256
sequences?
:::

::: exercise
[Underflow]{.box-title}

A gate of $\gamma=0.9$ applied over 1000 steps gives $0.9^{1000}\approx10^{-46}$, which underflows
bf16 and fp16. Identify the operations in [@lst:attnalt:chunked] that would fail if
implemented with explicit products and divisions, and explain why working with differences of
cumulative log-gates is safe.
:::

::: exercise
[Hybrid budget]{.box-title}

For the Qwen3-8B shape of [@fig:attnalt:cache], find the largest attention fraction (one
full layer every $k$ layers) such that a batch of 32 sequences of 128k tokens fits in 40 GB of cache
memory alongside a sliding-window alternative for the remaining layers with $w=4096$.
:::

::: exercise
[MLA absorption]{.box-title}

Show algebraically that $\boldsymbol{q}^\top\mathbf{W}^{UK}\boldsymbol{c}$ can be computed as
$(\mathbf{W}^{UK\top}\boldsymbol{q})^\top\boldsymbol{c}$ and count the FLOPs of both orders during
decode with $H$ heads, $d_{h}=128$ and $d_c=512$. Why does absorption help decode but not
necessarily training?
:::

::: exercise
[Batch-dependent MoE outputs]{.box-title}

Construct a small example of top-1 routing with capacity factor 1.0 where the output for token A
changes when token B is added to the batch. What does this imply for testing an MoE inference engine
for determinism?
:::

::: exercise
[Serving a recurrent layer]{.box-title}

Design the data structures a continuous-batching engine ([@ch:serving]) needs for a hybrid
with paged KV for attention layers and fixed-size state slots for recurrent layers. How do you
support (a) prefix caching of a shared system prompt, (b) parallel sampling of 8 continuations, and
(c) preemption with swap-out?
:::

## Lab 9: Linear attention, the delta rule, and selective scans {#sec:attnalt:lab}

::: labbox
[Linear attention, the delta rule, and selective scans]{.box-title}

**Code:** `code/labs/lab13_attnalt`

**Goals**

Build the core operators of non-softmax sequence mixers in their parallel, recurrent and chunked
forms; prove their equivalence with tests; and quantify their memory advantage at decode time.

**Setup**

``` bash
cd code
LAB_IMPL=starter  python -m pytest -q labs/lab13_attnalt
LAB_IMPL=solution python -m pytest -q labs/lab13_attnalt
python labs/lab13_attnalt/bench.py --device cpu     # or --device cuda
```

Reference layers: `llmsys/linear_attn.py` (`LinearAttention`, all three forms, delta rule, memory
calculators) and `llmsys/ssm.py` (`scan_parallel`, `MambaBlock`). Tests use float64 so equivalences
can be checked to $10^{-10}$.

**Tasks**

1.  `linear_attention_quadratic`: masked $(\mathbf{Q}\mathbf{K}^\top\odot\mathbf{D})\mathbf{V}$ with
    a decay mask built from cumulative log-gates; support no decay, per-head decay $(H,)$ and
    per-step gates $(B,H,T)$.

2.  `linear_attention_recurrent`: the RNN form, returning the final state so that a sequence can be
    continued later (tested by splitting a sequence in two).

3.  `linear_attention_chunked`: the chunkwise form; it must agree for chunk sizes 1, 4, 7 and 64.

4.  `delta_rule_step`: the (gated) delta update. The test checks exact overwrite of repeated keys.

5.  `associative_scan`: Hillis--Steele scan for $h_t=a_th_{t-1}+b_t$.

6.  `selective_ssm_step`: one discretised Mamba step; the test plugs it into `MambaBlock` and
    compares against the block's own forward pass.

7.  `sliding_window_mask` and `decode_cache_bytes` for full, windowed, linear and hybrid stacks.

**Measurements**

- Run `bench.py` at increasing context lengths. On a GPU, per-token decode time of softmax attention
  should grow linearly with $T$ while the linear and SSM steps stay flat; find the crossover length.
  (On a busy CPU the numbers are noisy; look for the trend over a wide range of $T$.)

- Reproduce [@fig:attnalt:cache] with `decode_cache_bytes` for a model of your choice.

- Run `recall.py` for the patterns `AA`, `LL`, `MM` and `LA` at 8 and 16 pairs (reproducing
  [@tab:attnalt:mqar]), then increase $d$ for the recurrent models until they solve 8
  pairs. How does the required state size scale with the number of pairs?

**Deliverables**

Passing tests; a decode-time plot; a cache-memory plot; a recall table with a paragraph explaining
it in terms of state capacity.

**Stretch goals**

Implement the Blelloch (work-efficient) scan; implement a chunked delta rule using the WY
representation and test it against the recurrent form; write a Triton kernel for the chunked forward
that keeps $\mathbf{S}$ in shared memory ([@app:triton]).
:::
