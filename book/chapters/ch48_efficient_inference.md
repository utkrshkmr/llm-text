# Making Inference Cheaper and Deploying It {#ch:efficient}

[@ch:inference] showed that a single decode step of a dense transformer is almost always
*memory-bound*: to emit one token we stream every weight (and the whole KV cache) from HBM through
the compute units, do roughly two FLOPs per byte, and wait. [@ch:serving] attacked that
inefficiency by putting more tokens behind each weight read: continuous batching, paged KV caches
and prefix sharing. This chapter covers the remaining levers that a production inference stack
pulls:

1.  **Move fewer bytes.** Store weights (and KV) in 8, 4 or even fewer bits (*quantization*) so each
    decode step reads less memory.

2.  **Get more tokens per expensive step.** Let a cheap model guess several tokens and verify them
    with *one* forward pass of the big model (*speculative decoding*) --- without changing the
    output distribution at all.

3.  **Split the model across devices** so that aggregate bandwidth grows with the number of GPUs
    (tensor, pipeline and expert parallel inference).

4.  **Constrain and ship the output**: structured decoding, OpenAI-compatible servers, streaming,
    load testing --- and, finally, a small browser chat interface around our own Qwen3
    implementation.

For a systems reader the theme is familiar: every technique trades a little extra compute
(dequantisation arithmetic, draft-model steps, collective communication) for a large reduction in
the scarcest resource, HBM bandwidth per generated token.

![Roadmap of this chapter. Stages 2--4 reduce the cost per generated token; stages 5--7 turn an
engine into something people can talk
to.](figures/ch48_efficient_inference-1.svg){#fig:efficient:roadmap}

## Where the cost of a token goes {#sec:efficient:cost}

Let a dense decoder have $N$ parameters stored with $b_w$ bytes each, serve a batch of $B$ sequences
whose average context is $T$, and keep $b_{kv}$ bytes per cached element. With $L$ layers, $G$ KV
heads and head dimension $d_{h}$, one decode step must read $$\begin{equation}
  \text{bytes}_{\text{step}} \;\approx\;
  \underbrace{N\,b_w}_{\text{weights, once per step}} \;+\;
  \underbrace{2\,L\,G\,d_{h}\,T\,B\,b_{kv}}_{\text{KV cache of all sequences}}
  \label{eq:efficient:bytes}
\end{equation}$$ and performs roughly $2NB$ FLOPs for the matrix multiplications plus
$4 L H d_{h}T B$ for attention scores and value mixing. Dividing by the hardware's bandwidth and
peak FLOP rate gives two lower bounds on the step time; the larger one wins ([@ch:machine]'s
roofline). With $B=1$ the arithmetic intensity is about one FLOP per byte, two orders of magnitude
below the H100's ridge point of $989/3.35\approx 295$ FLOP/byte.

:::: napkinmath
[Qwen3 on one H100, batch size 1]{.box-title}

Using `llmsys.flops` ([@ch:landscape]) and 3.35 TB/s of HBM bandwidth:

::: center
| model      | params | bf16 weights | ms/token (bf16) | ms/token (int4) |
|:-----------|-------:|-------------:|----------------:|----------------:|
| Qwen3-0.6B | 0.60 B |      1.19 GB |            0.36 |            0.09 |
| Qwen3-8B   | 8.19 B |      16.4 GB |            4.89 |            1.22 |
| Qwen3-32B  | 32.8 B |      65.5 GB |            19.6 |            4.89 |
:::

These are *floors* from weight traffic alone: 205 tokens/s for Qwen3-8B in bf16, 818 tokens/s if the
weights were 4-bit. No kernel can beat them at batch size 1; the only escapes are fewer bytes (this
section), more tokens per byte (batching, speculation), or more bandwidth (more GPUs).
::::

The KV term matters as soon as batches and contexts grow. For Qwen3-8B the cache costs
$2\cdot 36\cdot 8\cdot 128\cdot 2$ bytes $=147.5$ KB per token. A batch of 64 requests at 4096
tokens of context holds $64\cdot 4096\cdot 147.5\,\text{KB}\approx 38.7$ GB of KV, more than twice
the weights. [@eq:efficient:bytes] then predicts a 16.4 ms step (55 GB at 3.35 TB/s), and
the step is *still* memory-bound: its arithmetic intensity is only about 20 FLOP/byte.

::: systemslens
[Three levers, three different bottlenecks]{.box-title}

- **Weight quantization** shrinks the first term of [@eq:efficient:bytes]. It helps most at
  small batch and short context, where weights dominate.

- **KV quantization** (and GQA, MLA, paging, eviction) shrinks the second term, which dominates at
  large batch or long context, i.e. exactly the regime of high-throughput serving and long reasoning
  traces.

- **Speculative decoding** keeps the bytes the same but amortises them over several accepted tokens.
  It helps most when the GPU is idle-ish (small batch) and helps little when large batches already
  make the step compute-bound.

Always ask which term of [@eq:efficient:bytes] dominates *your* workload before choosing a
technique.
:::

## Quantization fundamentals {#sec:efficient:quant}

### Number formats

A quantized model stores tensors in a narrow format and keeps enough metadata (scales, zero points)
to map them back to real numbers. [@fig:efficient:formats] lays out the formats you will
meet. Floating-point formats spend bits on an exponent, which buys dynamic range: bf16 keeps fp32's
8-bit exponent and is therefore the default training and inference format. FP8 comes in two
flavours: E4M3 (more precision, max $\pm 448$) for weights and activations, and E5M2 (more range,
max $\pm 57344$) traditionally for gradients. Integer formats spend every bit on resolution but have
*uniform* spacing, so they need an explicit scale to place that grid where the values are.

![Bit layouts of the formats used for LLM inference. Exponent bits buy dynamic range; mantissa or
integer bits buy resolution. Formats narrower than 8 bits are only usable together with fine-grained
(per-group or per-block) scales.](figures/ch48_efficient_inference-2.svg){#fig:efficient:formats}

### Uniform affine quantization

A $b$-bit uniform quantizer maps a real tensor $x$ to integer codes with a scale $s>0$ and a zero
point $z$: $$\begin{equation}
  q = \operatorname{clamp}\!\Big(\big\lfloor x/s \big\rceil + z,\; q_{\min},\; q_{\max}\Big),
  \qquad \hat x = s\,(q - z).
  \label{eq:efficient:affine}
\end{equation}$$ **Symmetric** (absmax) quantization sets $z=0$ and $s=\max|x|/(2^{b-1}-1)$, which
suits roughly zero-centred weights and makes the integer GEMM cheap (no zero-point correction
terms). **Asymmetric** quantization uses the full range $[\min x,\max x]$ with
$s=(\max x-\min x)/(2^b-1)$ and $z=\lfloor -\min x/s\rceil$, which wastes no codes on one-sided
distributions.

If values are spread across the grid, the rounding error is roughly uniform on $[-s/2,s/2]$ and has
variance $s^2/12$. Each extra bit halves $s$, which gives the familiar rule of thumb of about 6 dB
of signal-to-quantization-noise ratio per bit. The catch is the word *spread*: one outlier stretches
$\max|x|$, inflates $s$ for *everything else* in the same group, and pushes small values to zero.
Nearly every modern quantization method is a way of fighting outliers.

``` {#lst:efficient:absmax .python caption="Symmetric absmax and asymmetric group-wise quantization (from `llmsys/quant.py`)."}
def quantize_absmax(w, bits=8, dim=-1):
    qmax = 2 ** (bits - 1) - 1
    amax = w.abs().max() if dim is None else w.abs().amax(dim=dim, keepdim=True)
    scale = (amax / qmax).clamp(min=1e-12).float()
    q = torch.clamp(torch.round(w.float() / scale), -qmax, qmax).to(torch.int8)
    return q, scale

def quantize_groupwise(w, bits=4, group_size=128):
    out_f, in_f = w.shape
    qmax = 2 ** bits - 1
    g = w.float().reshape(out_f, in_f // group_size, group_size)
    wmin, wmax = g.amin(dim=-1), g.amax(dim=-1)
    scale = ((wmax - wmin) / qmax).clamp(min=1e-12)
    zero = torch.round(-wmin / scale).clamp(0, qmax)
    q = torch.clamp(torch.round(g / scale[..., None]) + zero[..., None], 0, qmax)
    return q.to(torch.uint8).reshape(out_f, in_f), scale, zero
```

### Granularity: per-tensor, per-channel, per-group

[@fig:efficient:granularity] shows the three granularities for a weight matrix
$W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$ (one row per output channel, as stored by
`nn.Linear`). Per-tensor uses one scale and is fragile. Per-output-channel uses one scale per row
and is the default for int8 weights. Since $y_j=\sum_i W_{ji}x_i$, a per-row scale factors out of
the dot product and can be applied once to the GEMM result. Per-group splits each row into groups of
$g$ consecutive input channels (typically $g=128$, sometimes 32 or 64), each with its own scale and
zero point. This is what makes 4-bit weights usable.

![Quantization granularity for a weight matrix. Red boxes enclose the elements that share one scale.
Finer groups track local magnitudes better at the cost of more
metadata.](figures/ch48_efficient_inference-3.svg){#fig:efficient:granularity}

::: napkinmath
[The metadata tax of group-wise int4]{.box-title}

With $g=128$, an fp16 scale and a 4-bit zero point per group cost $(16+4)/128\approx0.16$ extra bits
per weight; with an fp16 zero point, $32/128=0.25$ bits. So "4-bit" Qwen3-8B actually stores about
$4.16$--$4.25$ bits per parameter: roughly 4.3 GB instead of the naive 4.1 GB. With $g=32$ the tax
rises to 0.5--1 bit, which is why tiny groups are reserved for formats with very cheap (8-bit
exponent-only) scales such as the microscaling (MX) formats.
:::

### Packing sub-byte values

Memory is byte-addressed, so two 4-bit codes share one byte. Our `pack_int4` places even columns in
the low nibble and odd columns in the high nibble ([@fig:efficient:pack]). Production
kernels choose an interleaving that matches how their GEMM loads registers. The Marlin kernel, for
instance, pre-shuffles weights offline so that one 128-bit load feeds tensor-core fragments
directly. Packing order is therefore part of the *kernel's* contract, not the model's, and
checkpoints are often repacked at load time.

![Packing pairs of 4-bit codes into bytes: the low nibble holds the even element and the high nibble
the odd element.](figures/ch48_efficient_inference-4.svg){#fig:efficient:pack}

### Weight-only quantization (W8A16, W4A16)

In weight-only schemes the activations stay in bf16 and the weights are dequantized *inside* the
GEMM: a kernel loads packed integers from HBM, expands them in registers, multiplies by the group
scale and feeds tensor cores in bf16. Because decode is memory-bound, the extra ALU work is free and
the step time drops nearly in proportion to the weight bytes. [@lst:efficient:quantlinear]
is the reference layer we use in the lab. It dequantizes the full weight in PyTorch, which is
correct but *not* faster: it materialises the fp32 weight in memory, the very traffic we wanted to
avoid. [@app:triton] writes the fused kernel.

``` {#lst:efficient:quantlinear .python caption="A weight-only quantized linear layer (reference semantics, not a fast kernel)."}
class QuantLinear(nn.Module):
    def __init__(self, linear, bits=8, group_size=128):
        super().__init__()
        w = linear.weight.data
        if bits == 8:                       # per-output-channel symmetric
            q, s = quantize_absmax(w, 8, dim=-1)
            self.register_buffer("qweight", q); self.register_buffer("scale", s)
        else:                               # 4-bit asymmetric groups, packed
            q, s, z = quantize_groupwise(w, 4, group_size)
            self.register_buffer("qweight", pack_int4(q))
            self.register_buffer("scale", s.half()); self.register_buffer("zero", z)
        ...
    def forward(self, x):
        w = self.dequantized_weight(x.dtype)   # a fused kernel never builds this
        return x @ w.T + (0 if self.bias is None else self.bias)
```

::: pitfall
[Quantizing the wrong layers]{.box-title}

Embeddings and the output head are often kept in bf16. The head maps to a vocabulary of about 150k
entries, so its errors land directly on the next-token distribution. Our `quantize_model` skips
`out_head` by default. Norm weights and biases are a negligible fraction of bytes; never quantize
them.
:::

### GPTQ: rounding with error feedback

Round-to-nearest (RTN) treats each weight independently. But what we care about is the layer's
*output* on realistic inputs. Given calibration activations $X\in\mathbb{R}^{n\times d_{\text{in}}}$
for a layer, GPTQ (Frantar et al., 2023) minimises the output error row by row: $$\begin{equation}
  \min_{Q}\; \big\| X W^\top - X Q^\top \big\|_F^2
  \;=\; \sum_j (\boldsymbol{w}_j - \boldsymbol{q}_j)^\top H\, (\boldsymbol{w}_j - \boldsymbol{q}_j),
  \qquad H = X^\top X .
  \label{eq:efficient:gptq}
\end{equation}$$ It builds on Optimal Brain Surgeon: quantize one input column $i$, and update the
remaining unquantized columns of every row to absorb the error, using the inverse Hessian:
$$\begin{equation}
  \boldsymbol{w}_{:,>i} \;\leftarrow\; \boldsymbol{w}_{:,>i} \;-\;
  \frac{w_{:,i}-q_{:,i}}{[H^{-1}]_{ii}}\; [H^{-1}]_{i,>i} .
\end{equation}$$ GPTQ's engineering contributions make this fast enough for 100B-parameter models.
It quantizes columns in a fixed order, which makes $H^{-1}$ the same for every row. It uses a
Cholesky factorisation of $H^{-1}$, whose rows provide the update coefficients. And it uses lazy
block updates, which turn many rank-1 updates into GEMMs. [@lst:efficient:gptq] is the
unblocked version.

``` {#lst:efficient:gptq .python caption="GPTQ without lazy batching. On correlated calibration inputs it gives lower output error than RTN on the same grid."}
def gptq_quantize(w, x, bits=4, damp=0.01):
    W = w.clone().float(); d = W.shape[1]
    H = x.float().T @ x.float()
    H += damp * torch.mean(torch.diag(H)) * torch.eye(d)        # dampening
    Hinv = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    qmax = 2 ** (bits - 1) - 1
    scale = (W.abs().amax(dim=1) / qmax).clamp(min=1e-12)      # per-row grid
    Q = torch.zeros_like(W)
    for i in range(d):
        col = W[:, i]
        q = torch.clamp(torch.round(col / scale), -qmax, qmax) * scale
        Q[:, i] = q
        err = (col - q) / Hinv[i, i]
        W[:, i + 1:] -= err[:, None] * Hinv[i, i + 1:][None, :]  # error feedback
    return Q
```

AWQ (Lin et al., 2024) starts from a different observation: a small fraction of weight input
channels matter far more because their *activations* are large. Instead of keeping those channels in
high precision (which breaks the uniform kernel), AWQ multiplies salient input channels of $W$ by a
factor $s>1$ and divides the corresponding activations by $s$. The product is unchanged, but the
important weights now occupy more of the quantization grid. The factors are searched on calibration
data. Both GPTQ and AWQ produce ordinary group-wise int4 checkpoints that the same W4A16 kernels can
serve.

### Quantizing activations: outliers and SmoothQuant

Weight-only schemes do nothing for compute-bound prefill or large-batch decode. There we want int8
(or FP8) *matmuls*, which means quantizing activations too (W8A8). LLM activations are hard. Beyond
a few billion parameters, a handful of hidden dimensions carry values 10--100$\times$ larger than
the rest, consistently across tokens. LLM.int8() (Dettmers et al., 2022) handles them with a
mixed-precision decomposition: outlier columns in fp16, the rest in int8. SmoothQuant (Xiao et al.,
2023) instead migrates the difficulty into the weights with a per-input-channel factor:
$$\begin{equation}
  Y = X W^\top = \big(X\,\mathrm{diag}(\boldsymbol{s})^{-1}\big)\big(W\,\mathrm{diag}(\boldsymbol{s})\big)^\top,
  \qquad s_j = \frac{\max|X_{:,j}|^{\alpha}}{\max|W_{:,j}|^{1-\alpha}} .
  \label{eq:efficient:smooth}
\end{equation}$$ With $\alpha\approx0.5$ the per-channel ranges of activations and weights are
balanced. The division by $\boldsymbol{s}$ is folded offline into the preceding RMSNorm's scale, so
it costs nothing at runtime ([@fig:efficient:smooth]).

![SmoothQuant moves per-channel magnitude from the activations into the weights. The layer computes
the same function, but both operands become
int8-friendly.](figures/ch48_efficient_inference-5.svg){#fig:efficient:smooth}

### FP8 and microscaling formats

Hopper-class GPUs execute FP8 GEMMs at twice the bf16 rate, and Blackwell adds FP4. FP8 E4M3 has
enough dynamic range that a single per-tensor scale (computed from the running absmax, possibly from
the previous iteration: "delayed scaling") is often sufficient for W8A8 inference with negligible
quality loss. Below 8 bits even floating point needs block scales. The OCP microscaling formats
(MXFP8/MXFP6/MXFP4) attach one shared power-of-two (E8M0) scale to every block of 32 elements.
NVIDIA's NVFP4 uses blocks of 16 with an FP8 scale plus a per-tensor scale. Hardware applies the
block scales inside the tensor-core instruction, so the fine granularity no longer costs a separate
dequantization pass.

``` {#lst:efficient:fp8 .python caption="Simulating FP8 storage with a per-tensor scale."}
def fp8_roundtrip(t, fmt="e4m3"):
    dtype = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[fmt]
    scale = (t.abs().max() / torch.finfo(dtype).max).clamp(min=1e-12)
    return (t / scale).to(dtype).to(torch.float32) * scale
```

### Quantizing the KV cache

[@eq:efficient:bytes] shows that the KV cache dominates long-context and large-batch
decode, so quantizing it pays twice: fewer bytes per step, and more sequences fit in memory, which
raises the achievable batch size. Values behave like ordinary activations and quantize well per
token. Keys are different. Empirically, a few *channels* of $K$ are consistently large across
tokens, so KIVI (Liu et al., 2024) quantizes keys per channel and values per token. FP8 KV caches
are a common production default; int4 and 2-bit caches need care, usually by keeping a recent window
in full precision. Since each attention kernel reads the cache, the dequantization must be fused
into the attention kernel as well ([@app:triton]).

:::: napkinmath
[Which bytes to cut for a throughput server]{.box-title}

Qwen3-8B, batch 64, 4096-token contexts, one H100 (numbers from [@sec:efficient:cost]):

::: center
| configuration         | weights | KV cache | step floor |
|:----------------------|--------:|---------:|-----------:|
| bf16 weights, bf16 KV | 16.4 GB |  38.7 GB |    16.4 ms |
| int4 weights, bf16 KV |  4.1 GB |  38.7 GB |    12.8 ms |
| bf16 weights, fp8 KV  | 16.4 GB |  19.3 GB |    10.7 ms |
| int4 weights, fp8 KV  |  4.1 GB |  19.3 GB |     7.0 ms |
:::

In this regime KV quantization alone beats 4-bit weights. And the first row would not even fit next
to activations and CUDA graphs on an 80 GB card.
::::

### Did quantization break the model?

Measure the quantized model against the reference model, not just against a benchmark score:

- **Per-token KL divergence** $\operatorname{KL}(p_{\text{ref}}\,\|\,p_{\text{quant}})$ on held-out
  text. It is sensitive, cheap, and needs no labels.

- **Perplexity delta**, the classic metric. It hides catastrophic errors on rare tokens.

- **Task accuracy with generation** (MATH-500, [@ch:eval]). Reasoning models are especially
  sensitive: small per-token errors compound over traces of thousands of tokens, and a flipped digit
  is a wrong answer.

- **Flip rate**: the fraction of benchmark items whose correctness changed. Equal average accuracy
  can hide many flips in both directions.

::: deepdive
[Larger models: quantize or go smaller?]{.box-title}

For a fixed memory budget, a larger model at 4 bits usually beats a smaller model at 16 bits. This
was a central finding of the "k-bit inference scaling laws" study (Dettmers & Zettlemoyer, 2023),
whose optimum sat around 4 bits. Practically: if Qwen3-8B in bf16 does not fit your card but
Qwen3-14B in int4 (about 7.5 GB of weights) does, try the latter first. Below about 3 bits, quality
typically falls off a cliff unless the method is specialised (vector quantization,
quantization-aware fine-tuning).
:::

## Speculative decoding {#sec:efficient:spec}

At batch size 1, verifying five tokens costs about as much as generating one. The forward pass over
a $(1\times 5)$ block reads the same weights as a $(1\times1)$ step, and the extra FLOPs are
invisible below the roofline ridge. Speculative decoding (Leviathan et al., 2023; Chen et al., 2023)
exploits this. A small *draft* model $q$ proposes $\gamma$ tokens autoregressively (cheap steps).
The large *target* model $p$ then scores all of them in *one* forward pass, and a rejection-sampling
rule keeps the longest acceptable prefix plus one extra token
([@fig:efficient:spec-timeline]).

![Speculative decoding timeline. In each round the draft model takes $\gamma$ cheap steps, then the
target verifies the pending token plus all $\gamma$ drafts in a single forward pass. Every round
commits between 1 and $\gamma+1$
tokens.](figures/ch48_efficient_inference-6.svg){#fig:efficient:spec-timeline}

### The acceptance rule and why it is exact

Let $p(\cdot)$ and $q(\cdot)$ be the target and draft distributions at some position (after
temperature and top-$k$/top-$p$ filtering, the distributions we would actually sample from). The
draft proposes $x\sim q$. We accept it with probability $\min\!\big(1, p(x)/q(x)\big)$. If we
reject, we sample a replacement from the *residual* distribution $$\begin{equation}
  r(x) = \frac{\max\big(0,\,p(x)-q(x)\big)}{\sum_y \max\big(0,\,p(y)-q(y)\big)} .
\end{equation}$$

::: {#prop:efficient:exact .proposition}
\[Speculative sampling is lossless\] The token emitted by this procedure is distributed exactly
according to $p$.
:::

::: proof
*Proof.* The probability of accepting is
$\beta=\sum_y q(y)\min(1,p(y)/q(y))=\sum_y \min(p(y),q(y))$. Since $\sum_y p(y)=1$, the residual's
normaliser is
$$\textstyle\sum_y \max(0,p(y)-q(y)) = \sum_y \big(p(y)-\min(p(y),q(y))\big) = 1-\beta .$$ Hence
$$\begin{align*}
\Pr[\text{emit } x] &= \underbrace{q(x)\min\!\Big(1,\tfrac{p(x)}{q(x)}\Big)}_{\text{draft accepted}}
\;+\; \underbrace{(1-\beta)\,\frac{\max(0,p(x)-q(x))}{1-\beta}}_{\text{rejected, then resampled}}\\
&= \min(p(x),q(x)) + \max(0,p(x)-q(x)) \;=\; p(x). \qedhere
\end{align*}$$ ◻
:::

Applied position by position, the proposition shows that the whole generated sequence has the
target's distribution. Each later draft was proposed conditioned on accepted predecessors, and
verification stops at the first rejection. With temperature 0 both distributions are one-hot. The
rule then reduces to "accept while the draft equals the target's argmax, then emit the target's
argmax", so greedy speculative decoding reproduces greedy target decoding *token for token*. The lab
tests exactly this. Note that $\beta = 1-\mathrm{TV}(p,q)$: the acceptance rate is one minus the
total variation distance between the two models.

``` {#lst:efficient:accept .python caption="The accept-or-resample rule (`llmsys/specdec.py`)."}
def accept_or_resample(p, q, draft_token, generator=None):
    px, qx = float(p[draft_token]), float(q[draft_token])
    u = torch.rand((), generator=generator).item()
    if qx > 0 and u < min(1.0, px / qx):
        return True, draft_token
    residual = torch.clamp(p - q, min=0)
    s = residual.sum()
    residual = p if s <= 0 else residual / s
    return False, int(torch.multinomial(residual, 1, generator=generator))
```

### The full loop with KV caches and rollback

The systems content of speculative decoding is mostly cache management. We keep both models' KV
caches covering "all committed tokens except the last". The last committed token is *pending*: it is
the next input to feed. A round then proceeds as follows:

1.  Feed the pending token to the draft, sample $d_1$; feed $d_1$, sample $d_2$; ...; sample
    $d_\gamma$. The draft cache grows by $\gamma$ entries ($d_\gamma$ itself is not fed).

2.  Feed the block $[\text{pending}, d_1,\dots,d_\gamma]$ to the target in one call. Its $\gamma+1$
    output rows are the distributions $p_1,\dots,p_{\gamma+1}$.

3.  Accept a prefix $d_1..d_k$ and append one extra token: the residual sample at the first
    rejection, or a "bonus" sample from $p_{\gamma+1}$ if all were accepted.

4.  Roll both caches back by truncation to the new committed length minus one. If all drafts were
    accepted, the draft model never saw $d_\gamma$, so feed it before the next round.

``` {#lst:efficient:specloop .python caption="One round of speculative decoding (abridged from `speculative_generate`)."}
nxt = torch.tensor([[seq[-1]]])                      # pending token
for _ in range(g):                                   # 1) draft proposes
    q = probs_from_logits(draft(nxt, cache=dc)[0, -1], temperature, top_k, top_p)
    tok = int(torch.multinomial(q, 1, generator=generator))
    drafts.append(tok); qs.append(q); nxt = torch.tensor([[tok]])
block = torch.tensor([[seq[-1]] + drafts])           # 2) one target pass
ps = probs_from_logits(target(block, cache=tc)[0], temperature, top_k, top_p)
k, extra = 0, None                                   # 3) verify
for i, tok in enumerate(drafts):
    ok, t = accept_or_resample(ps[i], qs[i], tok, generator)
    if not ok:
        extra = t; break
    k += 1
if extra is None:
    extra = int(torch.multinomial(ps[g], 1, generator=generator))
seq.extend(drafts[:k] + [extra])                     # 4) commit + roll back
truncate_cache(tc, len(seq) - 1)
```

::: pitfall
[Rollback in paged and static caches]{.box-title}

Truncating our concatenation-based `KVCache` is a slice. In a paged engine ([@ch:serving])
rollback means decrementing the sequence length and possibly freeing the last block. In a static
cache it only means resetting a length counter. Rejected entries are simply overwritten next round.
Engines that capture the decode step in a CUDA graph must also capture a separate "verify $\gamma+1$
tokens" graph, because shapes differ.
:::

### How many tokens per round, and how much faster?

Suppose each draft is accepted independently with probability $\alpha$. The number of committed
tokens per round is one plus a truncated geometric variable: $$\begin{equation}
  \mathbb{E}[\text{tokens per round}] \;=\; \sum_{i=0}^{\gamma}\alpha^i \;=\;
  \frac{1-\alpha^{\gamma+1}}{1-\alpha}.
  \label{eq:efficient:tpr}
\end{equation}$$ Let $c$ be the draft-step time relative to a target step, and assume verifying
$\gamma+1$ tokens costs one target step (memory-bound). The wall-clock speedup is then
$$\begin{equation}
  S(\alpha,\gamma,c) \;=\; \frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c + 1)} .
  \label{eq:efficient:speedup}
\end{equation}$$ [@fig:efficient:spec-curves] plots both. Tokens per round saturate at
$1/(1-\alpha)$, while the cost of drafting grows linearly in $\gamma$, so every $(\alpha,c)$ pair
has an optimal $\gamma$, typically 3--8 for a good draft.

![Analytic model of speculative decoding ([@eq:efficient:tpr; @eq:efficient:speedup]).
Left: committed tokens per target forward pass. Right: speedup when a draft step costs 5% of a
target step. Each curve has an interior
optimum.](figures/ch48_efficient_inference-7.svg){#fig:efficient:spec-curves}

::: napkinmath
[Qwen3-0.6B drafting for Qwen3-8B]{.box-title}

Both models share a tokenizer, a hard requirement for vanilla speculative decoding. At batch 1 their
weight-read floors are 0.36 ms and 4.89 ms, so $c\approx 0.07$. With an acceptance rate of
$\alpha=0.8$ and $\gamma=4$: $\mathbb{E}[\text{tokens}]=(1-0.8^5)/0.2=3.36$ and the round costs
$4(0.07)+1=1.28$ target steps, giving $S\approx 2.6\times$. Real systems see less: the 0.6B model's
step is far from its bandwidth floor (kernel-launch overheads dominate tiny models,
[@ch:inference]), which makes the effective $c$ larger, and $\alpha$ varies wildly by domain.
Code and math are often highly predictable; creative writing at high temperature is not.
:::

### Measuring acceptance correctly

It is tempting to report "accepted drafts / proposed drafts". That is a *biased* estimate of
$\alpha$: after the first rejection in a round, the remaining drafts are never examined, yet they
are still counted as proposed. A consistent estimator counts only examined positions,
$\hat\alpha=\text{accepted}/(\text{accepted}+\text{rejections})$, and this is what
`SpecStats.alpha_hat` reports.

We measured both on CPU with toy models in this environment. The target is a 4-layer random
Qwen3-shaped model (untied embeddings), sampled at temperature 0.25. The drafts are copies of the
target with Gaussian noise added to every weight, so more noise means a worse draft. The run
generates 160 tokens per configuration (`bench.py toy` in the lab):

::: center
| draft noise | $\gamma$ | kept/proposed | $\hat\alpha$ | tokens/target call | [@eq:efficient:tpr] at $\hat\alpha$ |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 1 | 0.916 | 0.916 | 1.88 | 1.92 |
| 0.10 | 2 | 0.856 | 0.894 | 2.67 | 2.69 |
| 0.10 | 4 | 0.842 | 0.911 | 4.21 | 4.19 |
| 0.10 | 8 | 0.409 | 0.803 | 4.10 | 4.37 |
| 0.30 | 1 | 0.710 | 0.710 | 1.68 | 1.71 |
| 0.30 | 2 | 0.516 | 0.623 | 2.00 | 2.01 |
| 0.30 | 4 | 0.378 | 0.644 | 2.46 | 2.50 |
| 0.30 | 8 | 0.213 | 0.633 | 2.62 | 2.68 |
| 1.00 | 1 | 0.233 | 0.233 | 1.22 | 1.23 |
| 1.00 | 2 | 0.116 | 0.195 | 1.22 | 1.23 |
| 1.00 | 4 | 0.056 | 0.182 | 1.21 | 1.22 |
| 1.00 | 8 | 0.037 | 0.226 | 1.28 | 1.29 |
:::

![Measured on CPU in this environment (toy models, 160 generated tokens per point). The naive
kept/proposed" ratio falls with $\gamma$ although the per-position acceptance $\hat\alpha$ is
roughly constant: the bias described in the
text.](figures/ch48_efficient_inference-8.svg){#fig:efficient:alpha-bias}

Two things stand out. First, "kept/proposed" falls with $\gamma$ even when the per-position
acceptance does not, which is exactly the bias described above. Second, the measured tokens per call
track [@eq:efficient:tpr] evaluated at $\hat\alpha$, within the noise of a 160-token run.
Third, a poor draft (noise 1.0, $\hat\alpha\approx0.2$) never gets past about 1.25 tokens per call
however long it drafts, so every extra draft step is pure overhead.

### Variants: where do drafts come from?

- **Separate small model** from the same family (as above). Simple; costs extra memory and its own
  KV cache; needs the same tokenizer.

- **Self-speculation / early exit** (e.g. LayerSkip, Elhoushi et al., 2024): run only the first $k$
  layers of the target, then its head, as the draft. There are no extra weights, and the draft's KV
  for those layers can be reused. Our lab's `layer_skip_draft` builds this.

- **Extra decoding heads** (Medusa, Cai et al., 2024): $k$ small heads on top of the last hidden
  state predict tokens $t+2,\dots,t+k+1$ in parallel. Candidates are combined into a *tree* and
  verified at once.

- **Feature-level drafting** (EAGLE, Li et al., 2024): a one-layer model autoregresses on the
  target's hidden states rather than on tokens, which yields much higher acceptance than an
  independent small LM. EAGLE-style drafts are the default in several production engines.

- **Prompt lookup / $n$-gram drafting**: propose the continuation of the most recent earlier
  occurrence of the last $n$ tokens. It is free, and remarkably effective for editing,
  retrieval-augmented answers and code, where outputs copy the input. Lookahead decoding (Fu et
  al., 2024) generates $n$-grams on the fly with Jacobi iterations.

![Tree verification (Medusa, EAGLE, SpecInfer). Several candidate continuations share prefixes. They
are flattened into one block and verified in a single forward pass with a mask that lets each node
attend only to its ancestors and the committed context. The longest accepted root-to-leaf path is
kept.](figures/ch48_efficient_inference-9.svg){#fig:efficient:tree}

::: systemslens
[Speculation and batching compete for the same slack]{.box-title}

Speculation converts idle compute into fewer sequential steps. A continuous-batching server at high
load has *no* idle compute: its decode steps approach the ridge point and verification of $\gamma+1$
tokens per sequence now costs real FLOPs. So speculative decoding gives large latency wins at low
load and can *reduce* throughput at high load. Production engines therefore adapt $\gamma$ to load
or disable speculation above a batch-size threshold. Decide with the roofline: if $B(\gamma+1)$
tokens per step pushes arithmetic intensity past the ridge point, you are paying for rejected drafts
with real compute.
:::

## Splitting a model across GPUs for inference {#sec:efficient:parallel}

[@ch:distributed] derived tensor, pipeline and expert parallelism for training. Inference
reuses the same partitionings, with different trade-offs because there is no backward pass and
decode moves tiny activations.

##### Tensor parallelism (TP).

Shard each attention block by heads and each MLP by columns then rows (Megatron-style). Each layer
needs two all-reduces of a $B\times d_{\text{model}}$ activation (after attention and after the
MLP). Every GPU reads only $1/P$ of the weights and of the KV cache (KV heads are sharded too), so
the bandwidth floor drops by $P$. The price is $2L$ all-reduces per decode step, which are
latency-bound rather than bandwidth-bound at small $B$.

::: napkinmath
[Qwen3-32B with TP=2 on NVLink]{.box-title}

The weights are 65.5 GB in bf16, so they do not fit one 80 GB GPU with a useful KV cache. With TP=2
each GPU holds 32.8 GB, and the step floor falls from 19.6 ms to about 9.8 ms. Per decode step and
per sequence, the all-reduces move $2\cdot 64 \cdot 5120 \cdot 2\,\text{B} \approx 1.3$ MB,
negligible in bandwidth. But 128 collectives at roughly 5--10 $\mu$s each add 0.6--1.3 ms of pure
latency per token. This is why TP is kept within an NVLink domain, and why engines fuse the
all-reduce with the following norm or use one-shot all-reduce kernels for small messages.
:::

##### Pipeline parallelism (PP).

Assign contiguous layers to GPUs. For a single sequence this adds latency (the stages run one after
another) but no speedup. It pays off for throughput when many micro-batches are in flight, and it is
the usual way to span nodes connected by slower networks.

##### Expert parallelism (EP) for MoE.

A mixture-of-experts layer ([@ch:modern]) activates only a few experts per token, so decode
reads a fraction of the weights. But the experts are spread across GPUs, and every MoE layer becomes
two all-to-all exchanges (dispatch tokens to experts, combine results). Large-EP serving designs,
such as DeepSeek's published inference setup, place one or a few experts per GPU across many nodes
and overlap the all-to-all with computation by splitting a batch into two micro-batches. The new
failure mode is *load imbalance*: one hot expert throttles the whole step. Systems therefore
replicate popular experts and rebalance placement from routing statistics.

![Left: tensor-parallel decode shards heads, KV cache and MLP across GPUs, with two all-reduces per
layer. Right: expert-parallel MoE layers route tokens to experts on different GPUs with all-to-all
exchanges; imbalance turns one expert into a
straggler.](figures/ch48_efficient_inference-10.svg){#fig:efficient:parallel}

## Structured output and constrained decoding {#sec:efficient:structured}

Applications often need output that *parses*: JSON matching a schema, a function call, a number in a
`\boxed{}` ([@ch:eval]). Rather than hoping, we can *constrain* sampling. Compile the grammar
(regular expression, JSON schema or context-free grammar) into an automaton over *tokens*. At each
step, set the logits of tokens that would leave the language to $-\infty$, then sample as usual. The
model's probabilities are renormalised over the legal tokens.

``` {#lst:efficient:constrain .python caption="Masking logits to a legal token set. The hard part, computing `allowed_ids` efficiently from a grammar state, is what libraries such as Outlines and XGrammar provide."}
def constrained_step(logits, allowed_ids, temperature=0.0):
    mask = torch.full_like(logits, float("-inf"))
    mask[..., allowed_ids] = 0.0
    return sample_next_token(logits + mask, temperature)
```

The systems problem is cost. A BPE vocabulary of about 150k tokens must be checked against the
grammar at *every* step, on the CPU, while the GPU waits. Outlines (Willard & Louf, 2023)
precomputes, for each state of a regular expression's automaton, the set of tokens that keep the
automaton alive, which turns the per-step cost into an index lookup. Context-free grammars (JSON
with nesting) need a pushdown automaton, whose stack makes the state space unbounded. XGrammar (Dong
et al., 2024) splits tokens into context-independent ones (checkable ahead of time) and a small
context-dependent remainder checked at runtime, and overlaps mask computation with the GPU forward
pass. Constraints also interact with tokenization: forcing `"}"` as a single token when the model
would naturally emit `"}\n"` pushes it off-distribution. This is the "token healing" problem, and it
is why good libraries allow every token whose *characters* are consistent with the grammar.

::: keyidea
Constrained decoding changes the output distribution: it samples from $p$ restricted to the
language, not from $p$ conditioned on producing a valid string. For short structured fields the
difference is negligible. For long reasoning followed by an answer, let the model think
unconstrained and constrain only the final answer span.
:::

## The serving stack around the engine {#sec:efficient:stack}

An inference engine ([@ch:serving]) turns token ids into token ids. A service must also
accept HTTP requests, apply chat templates, tokenize, schedule, stream text back, and survive load.
[@fig:efficient:lifecycle] traces one request through an OpenAI-compatible server such as
those shipped with vLLM, SGLang, TensorRT-LLM or TGI.

![Life of a streaming chat request. Tokenization, detokenization and HTTP handling run on the CPU.
Production engines move them into separate processes so a slow client or a long prompt cannot stall
the batched GPU loop.](figures/ch48_efficient_inference-11.svg){#fig:efficient:lifecycle}

##### Streaming.

Responses are streamed as server-sent events (SSE), one small JSON delta per step. This decouples
*time to first token* (TTFT) from total latency. Two details matter. First, detokenization must be
*incremental and UTF-8 aware*: decoding one BPE token in isolation can split a multi-byte character
(an emoji is 4 bytes and may span several tokens). The usual fix decodes a small window of recent
ids and emits only text that no longer ends in the replacement character U+FFFD. Our chat app uses
exactly this (`IncrementalDetokenizer`). Second, the CPU threads that serialise and send deltas must
never block the GPU loop. In Python servers that means a separate process or thread for the engine,
communicating through queues.

##### Load testing.

To measure a serving system, drive it with an *open-loop* workload: requests arrive by a Poisson
process at rate $\lambda$, independent of how fast the server responds. Closed-loop clients ("$N$
users each waiting for their reply") self-throttle and hide queueing collapse. Sweep $\lambda$. At
each rate record the percentiles (p50/p90/p99) of TTFT and inter-token latency (ITL), and the
*goodput*: completed requests per second that met their latency SLOs. Use realistic length
distributions (prompt and output lengths sampled from logs or from datasets such as ShareGPT),
because the prefill/decode mix determines the result.

::: napkinmath
[Little's law sizes your KV cache]{.box-title}

Little's law says $\bar N = \lambda\,\bar W$: the mean number of requests in the system equals
arrival rate times mean time in the system. At $\lambda=20$ requests/s with replies that take
$\bar W=12$ s, about 240 sequences are resident on average. At 3000 tokens each and 147.5 KB per
token (Qwen3-8B), that is $240\cdot 3000\cdot 147.5$ KB $\approx 106$ GB of KV cache, more than an
80 GB GPU. The server must shard, quantize the KV, or queue, and queueing raises $\bar W$, which
raises $\bar N$ further. This feedback loop is why latency explodes near capacity.
:::

::: systemslens
[Disaggregation, revisited]{.box-title}

Prefill is compute-bound and bursty; decode is memory-bound and steady. Running them on the same
GPUs makes long prompts inflate the ITL of everyone else, and chunked prefill ([@ch:serving])
only mitigates this. Disaggregated serving sends prefill to one pool, ships the KV cache over
NVLink, InfiniBand or RDMA, and decodes in another pool, so each pool can use its own parallelism
and even its own hardware generation. The KV transfer is the new bottleneck. For Qwen3-8B a 4k-token
prompt produces about 0.6 GB of KV, roughly 12 ms at 50 GB/s.
:::

## A chat interface for our models {#sec:efficient:chat}

We now put a browser front end on the from-scratch Qwen3 implementation. It is useful for poking at
the checkpoints we train later (instruction-tuned in [@ch:instruct], RL-trained in
[@ch:grpo], distilled in [@ch:distill]), and it is a compact case study of everything
above: chat templates, streaming, detokenization, event loops and KV reuse. We use Chainlit, an
open-source Python package that provides a ChatGPT-like web UI and calls our Python functions when
messages arrive.

### Installing Chainlit

Chainlit is a regular PyPI package:

``` bash
pip install chainlit          # or, with uv:  uv add chainlit
```

It is deliberately *not* in the lab's `requirements.txt`: nothing else in the book depends on it,
and it pulls in a web stack (an ASGI server, websockets).

### Running an app as a script

A Chainlit app is an ordinary Python file that Chainlit *imports* and inspects for decorated
handlers: `@cl.on_chat_start` runs when a browser session opens, and `@cl.on_message` runs for every
user message. You start the server with

``` bash
chainlit run chat_single_turn.py      # serves http://localhost:8000
```

Because `chainlit run` owns the process's command line, our scripts cannot use `argparse`.
Configuration therefore comes from environment variables (`QWEN3_DIR`, `CHECKPOINT_PATH`,
`MAX_NEW_TOKENS`, `THINKING`, `QUANT_BITS`), read once at import time in `chat_common.py`. Module
import also happens exactly once per server process, so that is where the model is loaded and shared
by all browser sessions.

### Where the scripts live

Everything is in `code/labs/lab48_quant_specdec/`: `chat_common.py` (model loading, streaming,
detokenization, history trimming), `chat_single_turn.py` and `chat_multi_turn.py`. They import only
`llmsys` and the lab's `solution.py` (for optional quantized serving and prefix matching), so there
is nothing to download beyond the Qwen3 weights.

### The single-turn app

[@lst:efficient:single] is the whole single-turn app. Each message is wrapped in the ChatML
template, tokenized, and streamed back token by token into an initially empty message bubble.

``` {#lst:efficient:single .python caption="`chat_single_turn.py`: every message is answered independently."}
import chainlit as cl
import torch
from chat_common import (DEVICE, THINKING, IncrementalDetokenizer,
                         load_model_and_tokenizer, stop_token_ids, stream_tokens)

MODEL, TOKENIZER = load_model_and_tokenizer()   # once per server process
STOP = stop_token_ids(TOKENIZER)                # <|im_end|>, <|endoftext|>

@cl.on_chat_start
async def on_start():
    await cl.Message(content="Single-turn mode.").send()

@cl.on_message
async def on_message(message: cl.Message):
    prompt = TOKENIZER.apply_chat_template(message.content, enable_thinking=THINKING)
    ids = torch.tensor([TOKENIZER.encode(prompt)], device=DEVICE)
    reply = cl.Message(content="")
    await reply.send()                           # empty bubble to stream into
    detok = IncrementalDetokenizer(TOKENIZER)
    async for tok in stream_tokens(MODEL, ids, STOP):
        if tok in STOP:
            break
        piece = detok.push(tok)
        if piece:
            await reply.stream_token(piece)
    await reply.update()
```

Two systems details hide in `chat_common.py`. First, `load_model_and_tokenizer` loads the official
snapshot by default, or, if `CHECKPOINT_PATH` is set, builds a fresh `Qwen3Model` and loads your own
state dict. The tokenizer always comes from the snapshot directory, so it must match the checkpoint:
base and reasoning variants use different chat conventions, and a checkpoint fine-tuned on one
should be served with the same template. Second, `stream_tokens` runs generation in a worker thread
and passes tokens to the asyncio event loop through a queue ([@lst:efficient:stream]).
Calling the model directly inside the `async` handler would block the event loop for the whole
reply: the UI would not repaint and every other session would freeze.

``` {#lst:efficient:stream .python caption="Bridging blocking generation and an async web framework."}
async def stream_tokens(model, input_ids, stop_ids, cache=None):
    loop = asyncio.get_running_loop()
    queue, done = asyncio.Queue(), object()
    def worker():                                 # runs in a plain thread
        try:
            for tok in generate_stream(model, input_ids, MAX_NEW_TOKENS, cache=cache):
                t = int(tok[0, 0])
                loop.call_soon_threadsafe(queue.put_nowait, t)
                if t in stop_ids:
                    break
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)
    threading.Thread(target=worker, daemon=True).start()
    while (item := await queue.get()) is not done:
        yield item
```

### Running it, with your own checkpoints

``` bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir qwen3-0.6b
QWEN3_DIR=qwen3-0.6b chainlit run chat_single_turn.py
# serve a checkpoint from the GRPO or distillation chapter instead, 4-bit quantized:
QWEN3_DIR=qwen3-0.6b CHECKPOINT_PATH=runs/grpo/step500.pth QUANT_BITS=4 \
    chainlit run chat_single_turn.py
```

Chainlit usually opens a browser tab. Otherwise visit <http://localhost:8000>. With `THINKING=1` the
reasoning model first emits a `<think> … </think>` block; a simple prompt such as "What is 1+1?" can
trigger several hundred tokens of deliberation, which makes `MAX_NEW_TOKENS` a real cost knob
([@ch:scaling]).

### Multi-turn conversations

##### What multi-turn means.

The single-turn app *displays* a conversation but the model sees only the latest message. Ask "What
is 1+1?" and then "What did I just ask?" and the model has no idea. A multi-turn chat re-sends the
history with every request, because the model itself is stateless: the "memory" of a chat assistant
is nothing but its prompt.

##### The multi-turn app.

`chat_multi_turn.py` keeps a per-session list of messages in Chainlit's per-session store
(`user_session`), seeded with a system message. On each turn it appends the user message, serialises
the whole history with the ChatML template ([@fig:efficient:chatml]), generates, and
appends the assistant reply.

![The multi-turn prompt is the whole history serialised in ChatML. Because each turn's prompt
extends the previous one, the session can keep its KV cache and prefill only the new
suffix.](figures/ch48_efficient_inference-12.svg){#fig:efficient:chatml}

##### Fitting the context window.

Histories grow without bound, so before each turn we must leave room for `MAX_NEW_TOKENS`. The
simplest strategy left-truncates the token sequence. That can cut a message in half and even drop
the `<|im_start|>` marker, producing a prompt shape the model never saw in training. Our
`trim_history` instead drops the oldest *whole* user/assistant turns and never drops the system
message. More elaborate strategies summarise old turns with the model itself.

##### How history meets the KV cache.

Re-sending the history means re-prefilling it, an $O(T)$ cost that grows every turn, so a 20-turn
chat does quadratic total work. The app therefore keeps the session's `KVCache` together with the
token ids it covers. Each turn it computes the longest common prefix between the cached ids and the
new prompt (`common_prefix_len`), truncates the cache to that length, and prefills only the suffix
([@lst:efficient:multiturn]). This is a single-session version of the prefix caching that
engines such as vLLM and SGLang perform across all requests ([@ch:serving]).

``` {#lst:efficient:multiturn .python caption="The heart of `chat_multi_turn.py`: history in, KV cache reused."}
history.append({"role": "user", "content": message.content})
history = trim_history(history, TOKENIZER, CONTEXT - MAX_NEW_TOKENS)
prompt_ids = TOKENIZER.encode(TOKENIZER.apply_chat_template(history, THINKING))
cache, cached = cl.user_session.get("cache"), cl.user_session.get("cached_ids")
reuse = min(common_prefix_len(cached, prompt_ids), len(prompt_ids) - 1)
truncate_cache(cache, reuse)                     # drop anything that diverged
new_ids = torch.tensor([prompt_ids[reuse:]], device=DEVICE)
...                                              # stream as in single-turn
cl.user_session.set("cached_ids", (prompt_ids + generated)[: cache.seq_len(0)])
history.append({"role": "assistant", "content": reply.content})
```

::: pitfall
[When the prefix silently stops matching]{.box-title}

Prefix reuse is only valid if re-serialising the history reproduces *exactly* the tokens that were
cached. Three things break it. (1) Templates that rewrite history: Qwen3's official template removes
earlier turns' `<think>` blocks, so the re-serialised assistant turn is shorter than what was
generated. Our simplified template keeps them, so the prefix holds, but the model sees its old
reasoning (more tokens, and different from its training distribution). (2) Tokenization is not
compositional: the text of a reply, re-encoded in context, can tokenize differently from the ids
that were sampled. (3) Trimming old turns changes the prefix from the start. `common_prefix_len`
makes all three cases *safe*, since it falls back to recomputation, but they silently cost
performance. Log the reuse ratio (the app prints it after every reply).
:::

##### Recommendations.

Use the multi-turn app with models trained on multi-turn chat data, such as the official Qwen3
reasoning checkpoint. Checkpoints trained only on single prompt/response pairs, such as the
distilled models of [@ch:distill] or our small instruction-tuned models, were never taught to
condition on earlier assistant turns. Inspect those with the single-turn app; the multi-turn app
tends to confuse them.

## Summary

- A decode step costs, to first order, the bytes of weights plus KV cache divided by HBM bandwidth
  ([@eq:efficient:bytes]). Decide which term dominates before choosing an optimisation.

- Uniform quantization trades resolution for bytes; outliers are the enemy, and finer granularity
  (per-channel, per-group, per-block microscaling) is the main weapon. Weight-only int8/int4 speeds
  up memory-bound decode; W8A8/FP8 speeds up compute-bound prefill and large batches; KV
  quantization raises both speed and capacity in long-context serving.

- GPTQ uses second-order information to compensate rounding error; AWQ and SmoothQuant rescale
  channels so that what matters lands on the grid.

- Speculative decoding is *lossless*: accept with $\min(1,p/q)$, resample from $\max(p-q,0)$.
  Expected tokens per round are $(1-\alpha^{\gamma+1})/(1-\alpha)$; its benefit shrinks as batching
  consumes the idle compute it relies on.

- Tensor parallelism divides the bandwidth floor at the cost of $2L$ latency-bound collectives per
  step; expert parallelism trades all-to-alls and load balance for sparse compute.

- A service adds templates, streaming, incremental detokenization, async I/O, open-loop load testing
  and SLO-based goodput around the engine.

- A chat UI is a thin layer: the model is stateless, memory is the prompt, and the KV cache can make
  re-sending history nearly free.

## Exercises

::: exercise
[bytes per token]{.box-title}

Using [@eq:efficient:bytes], find the batch size at which Qwen3-0.6B's KV cache bytes equal
its weight bytes at 2048 tokens of context (bf16 for both). What does this say about weight-only
quantization for small models in a busy server?
:::

::: exercise
[rounding noise]{.box-title}

Show that if $x$ is uniformly distributed over a quantization bin of width $s$, the mean squared
rounding error is $s^2/12$. Use this to derive the "6 dB per bit" rule for a uniform quantizer of a
signal spanning its full range.
:::

::: exercise
[outliers]{.box-title}

A weight row has 127 values drawn from $\mathcal N(0,1)$ and one value equal to 40. Estimate the
fraction of the 127 normal values that round to zero under per-row symmetric int4 quantization, and
under group-wise int4 with $g=32$.
:::

::: exercise
[metadata]{.box-title}

Compute the effective bits per weight of: (a) int4 with $g=64$, fp16 scale and fp16 zero; (b) MXFP4
(blocks of 32, one 8-bit shared scale); (c) NVFP4 (blocks of 16, one FP8 scale, ignoring the
per-tensor scale).
:::

::: exercise
[GPTQ derivation]{.box-title}

Starting from [@eq:efficient:gptq] for a single row $\boldsymbol{w}$, show that if column
$i$ is fixed to $q_i$, the optimal update of the remaining columns is
$\delta\boldsymbol{w}_{-i} = -\frac{w_i-q_i}{[H^{-1}]_{ii}}[H^{-1}]_{-i,i}$, and that the increase
in loss is $(w_i-q_i)^2/[H^{-1}]_{ii}$.
:::

::: exercise
[SmoothQuant]{.box-title}

Show that [@eq:efficient:smooth] leaves $XW^\top$ unchanged, and that with $\alpha=0.5$ the
transformed per-channel maxima satisfy $\max|X'_{:,j}|=\max|W'_{:,j}|$. How would you fold
$\mathrm{diag}(\boldsymbol{s})^{-1}$ into a preceding RMSNorm?
:::

::: exercise
[lossless speculation]{.box-title}

Extend the proof of [@prop:efficient:exact] to two consecutive positions: show that the
pair of tokens emitted when the first draft is accepted and the second is verified has distribution
$p(x_1)\,p(x_2\mid x_1)$.
:::

::: exercise
[optimal draft length]{.box-title}

For $c=0.05$, find numerically the $\gamma$ that maximises [@eq:efficient:speedup] for
$\alpha\in\{0.6,0.8,0.9\}$. Derive the condition for $\gamma\to\gamma+1$ to help.
:::

::: exercise
[speculation under load]{.box-title}

A server runs Qwen3-8B on one H100 at batch 128 with 2k-token contexts. Using the roofline, estimate
whether verifying $\gamma+1=5$ tokens per sequence keeps the step memory-bound. What $\gamma$ would
you choose?
:::

::: exercise
[TP latency]{.box-title}

Model a tensor-parallel decode step as $t = t_{\text{mem}}/P + 2L\,(a + m/\beta)$ with
per-collective latency $a$, message size $m$ and link bandwidth $\beta$. For Qwen3-32B with
$a=8\,\mu$s, find the $P$ that minimises per-token latency at batch 1.
:::

::: exercise
[load testing]{.box-title}

Explain why a closed-loop benchmark with 64 concurrent users can report excellent p99 latency for a
server that collapses under an open-loop load of the same average rate.
:::

::: exercise
[chat memory]{.box-title}

In the multi-turn app, after 20 turns of 150 tokens each, how many prompt tokens are prefilled in
total with and without KV reuse? Which of the three pitfalls in [@sec:efficient:chat] would
you expect to hit first with the official Qwen3 template, and how would you fix it?
:::

## Lab 33: Quantize, speculate, and serve {#sec:efficient:lab}

::: labbox
[Quantize, speculate, and serve]{.box-title}

**Code:** `code/labs/lab48_quant_specdec`

**Goals**

Implement weight-only int8 and int4 quantization from the bit level up, build a speculative decoder
with KV-cache rollback that is provably lossless, and serve a model through a streaming multi-turn
chat UI with cross-turn KV reuse.

**Setup**

`cd code; LAB_IMPL=starter python -m pytest -q labs/lab48_quant_specdec`. All tests run on CPU with
tiny random Qwen3/GPT models. `bench.py` contains the real-model experiments (Qwen3-0.6B/1.7B
snapshots), and the chat apps need `pip install chainlit`.

**Tasks**

1.  `quantize_absmax` / `dequantize_absmax`: symmetric int8, per-tensor or per-channel. The test
    injects an outlier row; per-channel must win.

2.  `quantize_groupwise` / `dequantize_groupwise`: asymmetric 4-bit groups with zero points. Then
    `pack_int4` / `unpack_int4` (two codes per byte).

3.  `quantize_linear_layers`: deep-copy a model and replace every `nn.Linear` except the head with
    `WeightOnlyLinear`. Check relative output error ($<1\%$ for int8, $<10\%$ for int4) and the byte
    savings.

4.  `accept_or_resample`: the speculative sampling rule. The test checks by Monte Carlo that the
    output follows $p$.

5.  `verify_drafts`: accept a prefix and emit one extra token (residual or bonus).

6.  `speculative_generate`: the full loop with both KV caches, rollback, and the "draft never saw
    $d_\gamma$" fix. Greedy output must equal plain greedy decoding of the target, using
    `layer_skip_draft` as the draft.

7.  `expected_tokens_per_round`, and `common_prefix_len` for the chat app.

**Measurements**

1.  `python bench.py toy`: reproduce the acceptance table of [@sec:efficient:spec]. Plot
    measured tokens per target call against [@eq:efficient:tpr] evaluated at $\hat\alpha$,
    for each noise level.

2.  With real checkpoints: `bench.py quant` reports KL to the bf16 reference and linear-layer memory
    for int8 and int4. Expect int8 to be nearly lossless and int4 to show a measurable but small KL.
    Then time decode with your PyTorch `WeightOnlyLinear` and note that it is *slower* than bf16.
    Explain why with [@lst:efficient:quantlinear] and fix it with the fused kernel of
    Lab A.

3.  `bench.py spec` with Qwen3-0.6B drafting for Qwen3-1.7B (or 4B): tokens/s and $\hat\alpha$ for
    $\gamma\in\{2,4,6,8\}$ on a math prompt and a creative prompt, at temperature 0 and 0.7.

4.  Run `chat_multi_turn.py` for ten turns and record the reuse ratio it prints. Switch `THINKING`
    on and off and explain the difference.

**Deliverables**

Passing tests; a one-page report with the acceptance plot, the quantization KL/memory table, the
speculative speedup table, and an answer to "at what batch size would you turn speculation off for
this pair of models, and why?"

**Stretch goals**

- Implement `gptq_quantize` for a real layer of Qwen3-0.6B, using 128 calibration sequences, and
  compare layer output error with RTN at int4 and int3.

- Implement tree verification: draft the top-2 tokens at each of 2 levels (a tree of 6 nodes), build
  the ancestor mask of [@fig:efficient:tree], and verify with one forward pass. (You will
  need to pass a custom mask into attention.)

- Add int8 KV-cache quantization (`llmsys.quant.quantize_kv`) to a copy of `KVCache` and measure the
  KL it induces at 4k context.

- Write a tiny OpenAI-compatible `/v1/chat/completions` endpoint with SSE streaming (FastAPI), and
  load-test it open-loop at increasing Poisson rates.
:::
