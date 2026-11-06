## Chapter 8: The Architecture Design Space {#chapter-8-the-architecture-design-space .unnumbered}

#### Exercise 8.1

\(a\) `"bgrqd,bgkd->bgrqk"`: the KV group axis $g$ is shared and kept, $r$ and $q$ come only from
the queries, and $d$ is contracted. (b) `"bi,bj->bij"`. (c) `"bii->b"` (a repeated index on one
operand selects the diagonal, and dropping it from the output sums it). (d) `"bhtd,hdD->btD"`: both
$h$ and $d$ are contracted, which is the per-head output projection followed by the sum over heads,
i.e. exactly $\mathrm{concat}_h(\mathbf{O}_h)\mathbf{W}_O$.

#### Exercise 8.2

$(\mathbf{A}\mathbf{B})\mathbf{C}$: forming the $n\times n$ product costs $2n^2d$ FLOPs and
multiplying it by $\mathbf{C}$ another $2n^2d$, so $4n^2d$ in total.
$\mathbf{A}(\mathbf{B}\mathbf{C})$: the $d\times d$ product costs $2nd^2$ and the final product
another $2nd^2$, so $4nd^2$. The second order is cheaper when $n>d$, i.e. beyond $n=128$; at
$n=32{,}768$ it is $n/d=256$ times cheaper. In attention the row-wise softmax applied to
$\mathbf{Q}\mathbf{K}^\top$ prevents re-association.

#### Exercise 8.3

With fan-average initialisation $\sigma_1^2=\sigma_2^2=2/(d+4d)=2/(5d)$. Then
$\operatorname{Var}(h_j)=d\sigma_1^2=2/5$, and since $h_j$ is a centred Gaussian,
$\mathbb{E}[\mathrm{ReLU}(h_j)^2]=\operatorname{Var}(h_j)/2=1/5$. Hence
$\operatorname{Var}(y_o)=4d\,\sigma_2^2\cdot\frac15
=4d\cdot\frac{2}{5d}\cdot\frac15=\frac{8}{25}=0.32$: the block shrinks the signal. Keeping unit
variance requires $\sigma_2^2=\frac{1}{4d\cdot(1/5)}=\frac{5}{4d}$, i.e. He-style scaling for the
layer after a ReLU.

#### Exercise 8.4

\(a\) $\sigma_z\approx0.02\sqrt{1024}=0.64$, so the loss is about
$\ln151{,}936+0.64^2/2\approx11.93+0.20=12.13$. (b) With $s=1$, $\sigma_z\approx32$: the logits are
essentially random with a huge spread, so $\log Z$ is dominated by the maximum of $V$ Gaussians,
$\approx\sigma_z\sqrt{2\ln V}\approx32\times4.9\approx156$, while the target logit has mean 0. The
initial loss is on the order of 150 nats, a clear sign of a broken init. (Qwen3-0.6B ties its head
to the embedding table, so in the real model this exercise is about the embedding's init scale.)

#### Exercise 8.5

Parameters: $3\cdot d\cdot\frac83d=8d^2=2\cdot d\cdot4d$; matmul FLOPs per token are $2\times$
parameters in both cases. Saved activations per token, counting GEMM inputs and activation inputs:
plain MLP stores $x$ ($d$), the pre-activation ($4d$) and the activation output, which is the input
to $\mathbf{W}_2$ ($4d$), for $9d$ values. The GLU stores $x$ ($d$), the gate pre-activation, the up
projection and their product ($3\times\frac83d=8d$), also $9d$. In bf16 both are $18d$ bytes per
token. Unfused GLU code also materialises $\mathrm{SiLU}(\text{gate})$, adding $\frac83d$ values
unless the elementwise ops are fused or recomputed.

#### Exercise 8.6

With uniform weights the output at position $i$ is $\frac{1}{i+1}\sum_{j\le i}\boldsymbol{v}_j$. If
the first token is a distinguishable BOS token whose value vector is $\boldsymbol{e}_1$, and all
other tokens have value vectors with zero first coordinate, the first output coordinate is
$1/(i+1)$, strictly decreasing in $i$. The causal mask's varying denominator alone encodes position;
later layers can read it off and build sharper positional features.

#### Exercise 8.7

$\partial(\log Z)^2/\partial z_j=2\log Z\,p_j$ with
$\boldsymbol{p}=\operatorname{softmax}(\boldsymbol{z})$. Its projection on the all-ones direction is
$\frac{2\log Z}{V}\boldsymbol{1}$ (since $\sum_jp_j=1$), and the orthogonal part is
$2\log Z\,(\boldsymbol{p}-\frac1V\boldsymbol{1})$. The cross-entropy gradient
$\boldsymbol{p}-\boldsymbol{y}$ sums to zero, so it has no all-ones component: CE never moves
$\log Z$ as such. The z-loss provides the only force along $\boldsymbol{1}$, pulling $\log Z$ toward
0. Its orthogonal part does perturb the distribution, but it is scaled by $\alpha\approx10^{-4}$ and
vanishes as $\log Z\to0$, so predictions are essentially unaffected.

#### Exercise 8.8

Qwen3-8B, 8192 tokens: each RMSNorm reads and writes $8192\times4096\times2$ B, i.e. 134 MB, or
about 40 $\mu$s; two per block give 80 $\mu$s (the QK-norms act on smaller tensors and add little).
The block has about $1.93\times10^8$ weight parameters
($2\cdot4096^2+2\cdot4096\cdot1024+3\cdot4096\cdot12288$), so its matmuls take
$2\times1.93\times10^8\times8192\approx3.2\times10^{12}$ FLOPs, plus about $2.8\times10^{11}$ for
causal attention scores and outputs: about 3.5 ms at peak, or 6--7 ms at a realistic 50% MFU.
Unfused norms add 1--2%. A fused version needs one extra read to compute the RMS (or folds the
scaling into the GEMM epilogue), about 20 $\mu$s per norm, roughly halving the overhead. The general
lesson: norm cost is small relative to a full block but not negligible, and it grows in relative
terms for smaller $d$.

#### Exercise 8.9

With $N\approx12Ld^2=1.3\times10^9$: for $d/L=32$, $d^3=1.3\times10^9\cdot32/12$ gives
$d\approx1514$ and $L\approx47$; for $d/L=128$, $d\approx2402$ and $L\approx19$. Decode reads the
same 2.6 GB of bf16 weights (about 0.78 ms at 3.35 TB/s) in both cases, plus per-layer overhead of
$47\times15\,\mu$s$=0.71$ ms versus $0.28$ ms. The wide model decodes faster (about 1.06 ms versus
1.49 ms per token). It also suits tensor parallelism better: each layer's GEMMs are larger, which
gives better per-GPU efficiency after splitting, and there are fewer layers and so fewer all-reduces
per token.

#### Exercise 8.10

There is no single answer. A good report states the hypothesis before running anything, shows that
the difference exceeds the seed-to-seed spread of the baseline, includes the LR sweep for both
variants, and offers a mechanism. For example: "QK-norm without z-loss: max attention logit stays
bounded and the run tolerates $3\times$ higher LR; final loss unchanged within noise at the tuned
LR."
