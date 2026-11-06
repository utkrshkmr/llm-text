## Chapter 5: Attention Mechanisms {#chapter-5-attention-mechanisms .unnumbered}

#### Exercise 5.1.

The "chip" row of $\mathbf{A}$ is $(0.137,0.196,0.219,0.154,0.123,0.171)$, giving
$\boldsymbol{z}^{(6)}=(0.442,0.556,0.522)$. "chip" is the last token, so the causal row is
identical: it may see every position.

#### Exercise 5.2.

Without scaling, score standard deviation grows as $\sqrt{d_k}$, so the maximum softmax probability
approaches 1 as $d_k$ grows (nearly one-hot rows); with scaling the scores have unit variance and
the maximum probability stays roughly constant in $d_k$ (it depends on $T$). The softmax Jacobian
has entries $\alpha_i(\delta_{ij}-\alpha_j)$: for a one-hot row all entries are $\approx0$, so
almost no gradient reaches the queries and keys.

#### Exercise 5.3.

If the parameter version computes `x @ W` with `W` of shape $(d_{\text{in}},d_{\text{out}})$, set
`linear.weight.data = W.T.clone()` for each of Q, K, V (`nn.Linear` computes `x @ weight.T`). The
outputs then agree to float precision.

#### Exercise 5.4.

Route 1 gives
$\tilde\alpha_{ij}=\frac{e^{s_{ij}}/Z_i}{\sum_{k\le i}e^{s_{ik}}/Z_i}=\frac{e^{s_{ij}}}{\sum_{k\le i}e^{s_{ik}}}$
for $j\le i$ and 0 otherwise, which is the route-2 softmax over the visible prefix; $Z_i$ (which
contains future scores) cancels. Because the result is algebraically independent of $s_{ij}$ for
$j>i$, the Jacobian of output 2 with respect to input 5 is zero; autograd returns zero up to
rounding (fp32: around $10^{-8}$ or exactly 0).

#### Exercise 5.5.

$x$: $(4,1024,1024)$; $q,k,v$: $(4,1024,1024)\to(4,1024,16,64)\to(4,16,1024,64)$; scores and
weights: $(4,16,1024,1024)$; $\mathbf{A}\mathbf{V}$: $(4,16,1024,64)\to(4,1024,1024)$; output
$(4,1024,1024)$. Parameters: $3(1024^2+1024)+(1024^2+1024)=4{,}198{,}400$.

#### Exercise 5.6.

Per token per layer, the other forward FLOPs are projections $8d^2$ plus SwiGLU $6d\,d_{\text{ff}}$.
With causal masking exploited, a token's score and value products average $2Td$ over a sequence of
length $T$. Equality: $T=(8d^2+6dd_{\text{ff}})/(2d)=4d+3d_{\text{ff}}=16{,}384+33{,}024=49{,}408$
tokens.

#### Exercise 5.7.

Induction: with $(m,\ell)$ correct for prefix $P$, i.e. $\ell=\sum_{j\in P}e^{x_j-m}$, after block
$B$ we have $\ell e^{m-m'}+\sum_{j\in B}e^{x_j-m'}=\sum_{j\in P\cup B}e^{x_j-m'}$ with
$m'=\max_{P\cup B}x_j$. Merge of disjoint partial states:

``` python
def merge(m1, l1, o1, m2, l2, o2):     # o = un-normalised sum e^{s-m} v
    m = torch.maximum(m1, m2)
    a1, a2 = torch.exp(m1 - m), torch.exp(m2 - m)
    return m, a1 * l1 + a2 * l2, a1[..., None] * o1 + a2[..., None] * o2
```

Split-KV decoding computes partial states over KV chunks in parallel and reduces them with `merge`;
ring attention merges the partial state from each key/value block as it arrives from the
neighbouring device.

#### Exercise 5.8.

For $\boldsymbol{y}=\operatorname{softmax}(\boldsymbol{s})$,
$\partial y_j/\partial s_k=y_j(\delta_{jk}-y_k)$, so
$ds_k=\sum_j dy_j\,y_j(\delta_{jk}-y_k)=y_k(dy_k-\sum_j y_j dy_j)$. In matrix form,
$\mathbf{dS}=\mathbf{P}\odot(\mathbf{dP}-\boldsymbol{\delta})$ with $\delta_i=\sum_jP_{ij}dP_{ij}$.
Since $\mathbf{dP}=\mathbf{dO}\mathbf{V}^\top$,
$\delta_i=\sum_j P_{ij}\,\boldsymbol{dO}_i\cdot\boldsymbol{v}_j=\boldsymbol{dO}_i\cdot\sum_jP_{ij}\boldsymbol{v}_j=\boldsymbol{dO}_i\cdot\boldsymbol{O}_i$.

#### Exercise 5.9.

With 228 KB of SRAM the model's naive/tiled traffic ratio is about 6.5--10.6 for $D=64$, 2.3--2.7
for $D=128$, and below 1 for $D=256$ (tiling "loses"). The model counts only the score matrix's
write/read passes and ignores that the naive path also launches separate mask, softmax and dropout
kernels (more $T^2$ passes), stores $\mathbf{A}$ for backward, and that real kernels choose tile
shapes and use registers as well as shared memory. The qualitative lesson holds: the benefit shrinks
as $D^2/M$ grows.

#### Exercise 5.10.

$8192/128=64$ tiles per side, 4096 tiles in total: 64 diagonal tiles (1.6%), $64\cdot63/2=2016$
fully visible (49.2%) and 2016 fully masked (49.2%). Skipping masked tiles avoids about 49% of the
dense $\mathbf{Q}\mathbf{K}^\top$ work, close to the ideal 50%; diagonal tiles are computed in full
and masked element-wise.

#### Exercise 5.11.

Cache bytes $=2L\,H_{kv}d_{h}\,\min(t,w)\cdot b$: they grow linearly until $t=w$ and then stay
constant. At $t=32$k the windowed cache is $32\text{k}/4096=8\times$ smaller. Implementations keep a
rolling (circular) buffer that overwrites the oldest entries, so a cached shared prefix is destroyed
as soon as any request sharing it advances more than $w$ tokens past it. Prefix sharing would
require copy-on-write of the overwritten blocks or keeping prefix blocks pinned, and a prefix longer
than $w$ is only partially useful.

#### Exercise 5.12.

Expect the explicit implementation's peak memory to scale as $BHT^2$ with several live copies
(scores, probabilities, possibly the mask), running out of memory when
$\approx 2$--$3\cdot BHT^2\cdot b$ exceeds free memory; SDPA's peak scales linearly in $T$. The
measured out-of-memory length should be within a factor of $\sqrt{2}$--$\sqrt{3}$ of the single-copy
estimate.
