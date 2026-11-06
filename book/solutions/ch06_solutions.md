## Chapter 6: Modern Architectures: Qwen3 from Scratch {#chapter-6-modern-architectures-qwen3-from-scratch .unnumbered}

#### Exercise 6.1.

If $\mu=0$ then
$\sigma^2=\tfrac1d\sum_i(x_i-\mu)^2=\tfrac1d\sum_i x_i^2=\operatorname{RMS}(\boldsymbol{x})^2$, so
$(\boldsymbol{x}-\mu)/\sqrt{\sigma^2+\epsilon}=\boldsymbol{x}/\sqrt{\operatorname{RMS}^2+\epsilon}$
and the outputs coincide. In the explicit implementations the LayerNorm saves $x-\mu$, the
denominator and the normalized tensor (plus the input), while RMSNorm saves the fp32 input, the
reciprocal RMS (one value per row) and the normalized tensor; for a $(4,256,1024)$ fp32 input expect
RMSNorm to save roughly one full $4\cdot256\cdot1024\cdot4$ B $=4.2\,\mathrm{\mega\byte}$ tensor
less.

#### Exercise 6.2.

$8d^2=3dh$ gives $h=\tfrac83d$. For $d=4096$ parity is $h\approx10923$; Llama 2-7B's $h=11008$ is
0.8% larger. $11008=43\times256$: rounding to a multiple of 256 makes every GEMM tile full and lets
$h$ be split evenly across 2, 4 or 8 tensor-parallel ranks while keeping each shard a multiple of 32
or 64.

#### Exercise 6.3.

Reorder the coordinates so that dimension $i$ and $i+d_{h}/2$ are adjacent; in that basis RoPE at
position $m$ is the block-diagonal matrix
$R_m=\mathrm{diag}(R(m\theta_0),\ldots,R(m\theta_{d_{h}/2-1}))$. Then
$\langle R_mq,R_nk\rangle=q^\top R_m^\top R_nk=q^\top R_{n-m}k$ block by block, as in the 2-D case;
permuting both vectors identically does not change a dot product. Shifting $m$ and $n$ by the same
$c$ leaves $n-m$ unchanged. Left padding by $c$ tokens shifts every real token's position by $c$
within that row, so all scores between real tokens are unchanged (pad keys are masked out).

#### Exercise 6.4.

With `p = halves_to_interleaved_perm(D)`, the interleaved model's $q$ satisfies
$q_\text{inter}=q_\text{halves}[p]$ per head. Hence, for each head $h$, the rows `W[h*D:(h+1)*D]` of
the rotate-half checkpoint equal the interleaved rows reindexed by the inverse permutation:
`W_halves[h*D + j] = W_inter[h*D + inv(p)[j]]`; apply this to $\mathbf{W}_Q$ and $\mathbf{W}_K$ (and
to the QK-norm gains if present). $\mathbf{W}_V$ is untouched: RoPE is not applied to values, and
the attention scores---the only place the permutation acts---are invariant.

#### Exercise 6.5.

Qwen3-8B stores 144  per token in bf16, so 131072 tokens need 18 
($\approx19.3\,\mathrm{\giga\byte}$); in fp8, 9.7 . Next to 16.4  of bf16 weights a single context
needs 35.7 : it fits on an 80 GB GPU, and with a 10% reserve two such contexts fit
($\lfloor(72-16.4)/19.3\rfloor=2$). With 8-bit weights (8.2 ) three fit with a bf16 cache and six
with an fp8 cache.

#### Exercise 6.6.

Per decode token and layer, attention costs $4THd_{h}$ FLOPs and reads $2TGd_{h}b$ bytes of cache
($b$ bytes per element), so $I=2H/(Gb)$. In bf16 ($b=2$): $I=H/G=2,4,8$ for 0.6B, 8B, 32B; in fp8
($b=1$): $4,8,16$. Materializing `repeat_interleave` writes and then reads an $H$-head copy, so the
bytes per FLOP grow by at least $H/G$ and the effective intensity falls back to about 1 (or below,
counting the extra write).

#### Exercise 6.7.

If the cos/sin tables are computed in fp32 and only then cast to bf16, each entry has a relative
error of about $2^{-9}$, so the angle error stays near $0.004$ radians at every position. If,
however, the angles $m\theta_i$ themselves are formed in bf16 (e.g. because the model, including its
buffers, was cast before the tables were built), the fastest frequency has $\theta_0=1$ and the
angle equals $m$; bf16's spacing at magnitude $m\in[2^e,2^{e+1})$ is $2^{e-7}$, so the rounding
error ($\le2^{e-8}$) exceeds $0.1$ rad once $e\ge5$, i.e. already at positions of a few dozen
tokens.

#### Exercise 6.8.

$\beta'=10^6\cdot4^{128/126}\approx4.09e6$. The frequency ratio for pair $i$ is
$\theta_i'/\theta_i=s^{-2i/(d_{h}-2)}$, so the wavelength is multiplied by $s^{2i/(d_{h}-2)}$:
unchanged for $i=0$ (smallest change) and multiplied by exactly $s=4$ for $i=d_{h}/2-1=63$
(largest).

#### Exercise 6.9.

Capacity $=\lceil c\,Nk/E\rceil=1.25\cdot4096\cdot2/64=160$ slots per expert. Under uniform random
routing an expert's load is approximately $\mathcal N(128,\,11.2^2)$. The expected overflow
$\mathbb{E}[(X-160)^+]=\sigma[\varphi(z)-z(1-\Phi(z))]$ with $z=2.85$ is $\approx0.007$ slots per
expert, i.e. $\approx0.006\%$ dropped. With $c=1.0$ ($z=0$) it is $\sigma\varphi(0)\approx4.5$ slots
per expert, about 3.5% dropped; with $c=2.0$ ($z\approx11$) essentially none. Trained routers are
far from uniform, so real drop rates at $c=1$ are much higher, which is why dropless kernels became
popular.

#### Exercise 6.10.

Assume the 16384 tokens are spread evenly, 2048 per GPU. Each token sends $k=8$ copies of a $d=2048$
bf16 vector, 7/8 of which leave the GPU: dispatch is $2048\cdot8\cdot2048\cdot2\cdot\tfrac78
\approx58.7\,\mathrm{\mega\byte}$ per GPU, and the combine step returns the same, 117  per layer, or
about 0.26  at 450 . Each GPU's experts process about 16384 token-slots at $6dh=9.4\,\mathrm{M}$
FLOPs each, 155 , i.e. $\approx0.16\,\mathrm{\milli\second}$ at peak on an H100. Communication is
comparable to (or larger than) compute, so production systems overlap the all-to-all of one
micro-batch with the expert GEMMs of another.

#### Exercise 6.11.

Naive loading of the 1.7B model in default fp32 holds the randomly initialized fp32 model (6.9 )
plus the bf16 state dict (3.4 ); even in bf16 it holds two copies (6.9 ). The meta-device path peaks
at one copy plus the largest single tensor, the $151936\times2048$ embedding (0.62 ): about 4.1 .
