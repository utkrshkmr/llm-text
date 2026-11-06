## Chapter 16: Scaling Laws in Practice {#chapter-16-scaling-laws-in-practice .unnumbered}

#### Exercise 16.1

Minimise $E+AN^{-\alpha}+BD^{-\beta}$ subject to $6ND=C$. With $D=C/(6N)$ the objective is
$f(N)=E+AN^{-\alpha}+B(6N/C)^{\beta}$, and $f'(N)=0$ gives
$\alpha AN^{-\alpha}=\beta B(6N/C)^{\beta}=\beta BD^{-\beta}$. That is the stated $1:1$ ratio: at
the optimum the *marginal* gains from a relative increase in $N$ and in $D$ are equal, so no
reallocation helps. Solving $\alpha AN^{-\alpha}=\beta B(6N/C)^{\beta}$ gives
$N^{\alpha+\beta}=\frac{\alpha A}{\beta B}(C/6)^{\beta}$, so
$N_{\text{opt}}=G(C/6)^{\beta/(\alpha+\beta)}$ with $G=(\alpha A/\beta B)^{1/(\alpha+\beta)}$, and
$D_{\text{opt}}=(C/6)/N_{\text{opt}}=G^{-1}(C/6)^{\alpha/(\alpha+\beta)}$.

#### Exercise 16.2

The minimum of each IsoFLOP profile is exactly the constrained optimum of 12.1, so the minima lie on
$N_{\text{opt}}\propto C^{\beta/(\alpha+\beta)}$ with no approximation. With $C=6(N+Vd)D$, the
tokens a budget buys are $D=C/(6(N+Vd))$. At small $N$, where $Vd$ is comparable with $N$, a model
gets fewer tokens than the $6ND$ rule assumes. This shifts the small-budget optima towards smaller
$N$, where the per-token cost is relatively lower. As $C$ grows the embedding term fades, the optima
move back onto the ideal line, and the fitted exponent over a small-scale range comes out *larger*
than $\beta/(\alpha+\beta)$. This is one of the mechanisms behind Kaplan's larger exponent.

#### Exercise 16.3

Qwen3-0.6B: $L=28$, $d=1024$, 16 query heads of dimension 128 (query width 2048), 8 KV heads (K and
V width 1024), $d_{\text{ff}}=3072$, $V=151{,}936$, tied embeddings. Per layer: $Q$ $1024\cdot2048$,
$K$ and $V$ $1024\cdot1024$ each, $O$ $2048\cdot1024$, which is $6.29$M for attention; MLP
$3\cdot1024\cdot3072=9.44$M; per layer $15.73$M, so $N_{\text{ne}}=440.4$M. The tied embedding holds
$155.6$M ($26\%$ of all weights). Training FLOPs per token: $6(440.4+155.6)\text{M}=3.58\times10^9$
for the matmuls (the tied table is still a full $V\times d$ matmul at the output), plus attention
$6\cdot L\cdot T\cdot(Hd_h)=6\cdot28\cdot4096\cdot2048
=1.41\times10^9$, which is 39% on top of $6N$. At long context the attention term is not negligible.
The assumptions behind $12Ld^2$ that break here: query width $\neq d$ ($Hd_h=2d$), GQA shrinks K and
V, $d_{\text{ff}}/d=3$ rather than $8/3$, and tied embeddings.

#### Exercise 16.4

Write $x=\log N$ and assume locally $L(x)=L^*+\tfrac\kappa2(x-x^*)^2$. On a grid with spacing
$\ln r$, the nearest grid point is at most $\tfrac12\ln r$ from $x^*$, so the min reading has
$|\Delta x|\le\tfrac12\ln r$ and $0\le\Delta L\le\kappa(\ln r)^2/8$. With noise of standard
deviation $\sigma$ per run, the min reading can also pick the wrong neighbour whenever the loss gap
between neighbours ($\approx\kappa(\ln r)^2/2$ one step from the optimum) is comparable to $\sigma$.
A parabola through $k$ points averages the noise and interpolates, so it helps when
$\sigma\lesssim\kappa(\ln r)^2/2$ and the valley is well approximated by a parabola over the grid;
for very wide grids the quartic terms bias the vertex, which favours several sizes close to the
minimum over a very wide sweep.

#### Exercise 16.5

With $E=BS$ and $E_{\min}=B_{\text{crit}}S_{\min}$, write $s=S/S_{\min}$: then
$E/E_{\min}=(B/B_{\text{crit}})s$ and $(s-1)\bigl((B/B_{\text{crit}})s-1\bigr)=1$ reduces to
$s\bigl((B/B_{\text{crit}})s-B/B_{\text{crit}}-1\bigr)=0$, so $s=1+B_{\text{crit}}/B$. Hence
$S=S_{\min}(1+B_{\text{crit}}/B)$ and $E=BS=E_{\min}(1+B/B_{\text{crit}})$. The time to target is
$T(B)=S_{\min}(1+B_{\text{crit}}/B)(t_0+B/R)$, and $T'(B)=0$ gives $1/R=B_{\text{crit}}t_0/B^2$, so
$B^*=\sqrt{B_{\text{crit}}\,t_0R}$. This exceeds $B_{\text{crit}}$ exactly when
$t_0R>B_{\text{crit}}$: when the fixed per-step overhead (launches, communication latency, optimizer
step) is worth more than $B_{\text{crit}}$ examples of throughput, it pays to take fewer, larger
steps even though they waste data.

#### Exercise 16.6

In the synthetic service, intermediate losses of a cosine run sit above finished runs of the same
length by an amount proportional to the not-yet-decayed LR. The early points are inflated most, so
loss appears to fall faster with $D$ than it really does. One reference run (a 50M-parameter model,
16 evaluations, separate runs for comparison) gave $\hat\beta\approx0.35$ from the intermediate
losses versus $0.29$ from separate runs, which matches the hidden truth. An overestimated $\beta$
makes data look more valuable and biases the allocation toward smaller models. With WSD, a short
decay from a stable-phase checkpoint produces a genuinely finished run of that length, which removes
the bias at the cost of the decays.

#### Exercise 16.7

Durations $D_{\max}2^{-j}$, $j=0..k-1$. Cosine: $m\sum_j D_{\max}2^{-j}\to
2mD_{\max}$. WSD: $m\bigl(D_{\max}+f\sum_jD_{\max}2^{-j}\bigr)\to mD_{\max}(1+2f)$. The saving tends
to $1-(1+2f)/2=\tfrac12-f$, which is $40\%$ for $f=0.1$. For $k=4$ the totals are $15/8$ versus
$1+0.1\cdot15/8$ (in units of $mD_{\max}$), a $37\%$ saving, as in the napkin box.

#### Exercise 16.8

Adam's update has entries $\pm\eta$ (the sign of the normalised gradient). For a linear layer the
gradient is an outer product $g x^\top$, so the update is aligned with $x$:
$\Delta W\approx\eta\,\mathrm{sign}(g)\,\mathrm{sign}(x)^\top$. Then
$(\Delta W x)_i\approx\eta\sum_j|x_j|=\Theta(\eta n)$, because the $n$ terms add coherently. Keeping
the change $\Theta(1)$ requires $\eta\propto1/n$. With SGD the update is $\eta g x^\top$, and
$(\Delta Wx)_i=\eta g_i\|x\|^2=\Theta(\eta n g_i)$. Since the backpropagated $g_i$ to a hidden layer
scales as $1/n$ in $\mu$P, SGD's hidden-layer LR ends up width-independent: the rule depends on the
optimizer because Adam normalises the gradient's scale away.

#### Exercise 16.9

The constraint $6ND+2ND_{\text{inf}}=C$ gives $D=(C-2ND_{\text{inf}})/(6N)$, so each extra parameter
now also costs inference compute, which moves the stationary point to smaller $N$ than in 12.1 (the
extra term only increases the marginal cost of $N$). Numerically, for `GroundTruth()` at $C=10^{21}$
and $D_{\text{inf}}=10^{11}$: the Chinchilla optimum is $N\approx1.77\times10^9$ at about 53 tokens
per parameter, while the inference-aware optimum is $N\approx1.08\times10^9$, trained on
$1.2\times10^{11}$ tokens, about 112 tokens per parameter.

#### Exercise 16.10

Let the true runtime be $t=\hat t\,e^{\varepsilon}$ with $\varepsilon\sim\mathcal N(0,0.15^2)$, and
let $p(s)=\Phi(\ln s/0.15)$ be the completion probability. The expected charge per *useful* run is
$\bigl(\mathbb{E}[t;\,t<s\hat t]+s\hat t(1-p)\bigr)/p$. It decreases monotonically in $s$, from
$1.33\hat t$ at $s=1.1$ to $1.08\hat t$ at $1.25$, $1.015\hat t$ at $1.5$ and $1.011\hat t$ (the
mean runtime) beyond $2$, because refunds make large reservations free. Calendar throughput is
different: with at most $R/(s\hat t)$ runs in flight, each occupying a slot for
$\mathbb{E}[\min(t,s\hat t)]$, useful runs per hour are proportional to
$p(s)/(s\,\mathbb{E}[\min(t,s\hat t)])$, which peaks at $s\approx1.23$. A margin of 1.2--1.5$\times$
trades a few percent of budget for much better concurrency. Calibrate $\sigma$ on your own runs.

#### Exercise 16.11

Answers vary with the seeds. On the reference pipeline the regret is typically a few thousandths of
a nat on the default truth and a few hundredths on randomised truths (`GroundTruth.sample`). Most of
the spread comes from the LR transfer law and from the fixed aspect ratio, so the cheapest
improvements are a wider LR sweep at the largest affordable size and a small aspect-ratio sweep. The
prediction error is dominated by $E$ and is cut most by the verification run.
