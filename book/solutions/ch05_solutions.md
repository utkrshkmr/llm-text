## Chapter 5: Implementing a GPT Model {#chapter-5-implementing-a-gpt-model .unnumbered}

#### Exercise 5.1.

With [@eq:gpt:params] (untied head, no QKV bias, $V=50257$, $T_\text{max}=1024$): GPT-2
medium has 406212608 parameters (attention 100.7 , FFN 201.4 ); large has 838220800 (attention
236.0 , FFN 472.1 ); XL has 1637792000 (attention 491.6 , FFN 983.4 ). For $L=12$ the FFN holds
$12(8d^2+5d)$ parameters and the embeddings plus head hold $(2V+T_\text{max})d$. Equality gives
$96d+60=2\cdot50257+1024$, i.e. $d\approx1057$; any width above that (in practice the next multiple
of 64, $d=1088$) puts more parameters in the FFNs than in the vocabulary-sized matrices.

#### Exercise 5.2.

Add three keys, e.g. `drop_emb`, `drop_attn` and `drop_resid`, pass `drop_attn` to
`MultiHeadAttention` and `drop_resid` to `drop_shortcut`, and use `drop_emb` in `GPTModel`. Removing
the *attention-weight* dropout saves the most: it acts on the largest activation ($BHT^2$ elements
per layer), its mask must otherwise be stored or regenerated, and many fused attention kernels have
a faster code path when dropout is zero.

#### Exercise 5.3.

The next multiple of 128 above 50257 is 50304: 47 extra rows, $47\times768=36096$ parameters per
vocabulary matrix (72192 if the head is untied). Two fp32 Adam moments add 8 bytes per parameter:
289  (tied) or 578  (untied)---negligible. During generation either slice the logits to the true
vocabulary (`logits[..., :50257]`) or set the padded entries to $-\infty$ before argmax/sampling, so
a padded ID can never be emitted.

#### Exercise 5.4.

From $8Nd^2=r\,(10Nd+8d^2)$ with ridge point $r$ we get $N^\star=8rd/(8d-10r)$. For the A100,
$r=312/2.039\approx153$ FLOP/byte, giving $N^\star\approx204$ tokens for $d=768$ and $\approx160$
for $d=4096$. The second FFN GEMM, $(N\times4d)\cdot(4d\times d)$, also performs $8Nd^2$ FLOPs and
moves $2(4Nd)+2(4d^2)+2(Nd)=10Nd+8d^2$ bytes: its intensity is identical, so (ignoring tiling
effects) both GEMMs reach the ridge at the same $N^\star$.

#### Exercise 5.5.

Without scaling, each of the $2L$ sublayers adds an update of roughly constant variance $\sigma_u^2$
to the stream, so after $\ell$ sublayers the variance is $\sigma_0^2+\ell\sigma_u^2$ and the RMS
grows like $\sqrt{\ell}$: a concave, square-root-shaped curve. Scaling the output projections by
$1/\sqrt{2L}$ shrinks each update's variance by $2L$, so the total added variance after all $2L$
sublayers is about $\sigma_u^2$ and the curve becomes nearly flat.

#### Exercise 5.6.

Post-LN computes `x = norm(x + f(x))`. Without warmup, the post-LN model typically diverges (loss
spikes to NaN or plateaus at the unigram entropy) at a learning rate several times lower than the
pre-LN model tolerates, because the gradients of the last layers are large at initialization (Xiong
et al., 2020); adding a linear warmup closes much of the gap.

#### Exercise 5.7.

\(a\) The explicit tanh-GELU saves, per element of the $4d$-wide hidden tensor, the input (for `pow`
and the outer multiply), the `tanh` output, the half-scaled input and the $(1+\tanh)$ factor: about
four $4d$-wide tensors. (b) `nn.GELU` saves only its input, removing roughly three of them
(1387 $\to$`<!-- -->`{=html}994  on the tiny configuration). (c) The explicit LayerNorm saves
$x-\mu$, the denominator and the normalized tensor; `nn.LayerNorm` saves the input plus one mean and
one reciprocal standard deviation per row, so roughly one to two $d$-wide tensors disappear per
norm.

#### Exercise 5.8.

Causal attention computes, per head and sequence, $\sum_{i=1}^{T} i\,d_{h}\cdot2 = T(T+1)d_{h}$
FLOPs for $QK^\top$ and the same for $AV$, i.e. $2T(T+1)d$ per layer instead of $4T^2d$---a factor
$(T+1)/2T\approx\tfrac12$. For GPT-2 small the total forward FLOPs per token drop from 284.8M to
265.9M at $T=1024$ (6.6% less) and from 549.1M to 398.1M at $T=8192$ (27.5% less). If one paper
counts full attention and another counts only the causal half, their MFU numbers differ by this
amount for the same hardware efficiency, increasingly so at long context.

#### Exercise 5.9.

Neglecting attention terms, a forward pass over $t$ tokens costs $\approx2Pt$. Without a cache,
generating $n$ tokens costs $\sum_{t=p}^{p+n-1}2Pt\approx2P(pn+n^2/2)$; with a cache, $2P(p+n)$
(prefill plus one token per step). The ratio is $(pn+n^2/2)/(p+n)$, which for $p=100$ reaches 100 at
$n^2/2=10^4$, i.e. $n\approx141$. Including attention makes the uncached version slightly worse
still.

#### Exercise 5.10.

Loop over sequence chunks of 128 positions: compute the chunk's logits, its cross-entropy summed
over tokens, and accumulate; wrap each chunk in `torch.utils.checkpoint` so the chunk logits are
recomputed in backward rather than stored. Peak logits memory falls from
$8\cdot1024\cdot50257\cdot4\,\text{B}=1.65\,\mathrm{\giga\byte}$ to
$8\cdot128\cdot50257\cdot4\,\text{B}=206\,\mathrm{\mega\byte}$ (and the same for the softmax
gradient). The price is recomputing the head GEMM in the backward pass, about $2NVd$ extra FLOPs,
i.e. one third more work for the output layer.
