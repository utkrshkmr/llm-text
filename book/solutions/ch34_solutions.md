## Chapter 34: Policy-Gradient Foundations for LLM Reinforcement Learning {#chapter-34-policy-gradient-foundations-for-llm-reinforcement-learning .unnumbered}

#### Exercise 34.1.

$f(b)=\mathbb{E}[(r-b)^2\|g\|^2]=\mathbb{E}[r^2\|g\|^2]-2b\,\mathbb{E}[r\|g\|^2]+b^2\mathbb{E}[\|g\|^2]$
is a convex quadratic, so $f'(b)=0$ gives $b^\star=\mathbb{E}[r\|g\|^2]/\mathbb{E}[\|g\|^2]$. Per
coordinate, minimising $\mathbb{E}[(r-b_k)^2g_k^2]$ separately gives
$b_k^\star=\mathbb{E}[r g_k^2]/\mathbb{E}[g_k^2]$. For an LLM this needs a running estimate of two
statistics per parameter (two extra model-sized buffers), and it needs per-sample gradients rather
than the batch-summed gradient autograd produces, so it is impractical. Group and critic baselines
are per-prompt scalars instead.

#### Exercise 34.2.

$\pi=(0.506,0.307,0.186)$. The exact gradient with respect to the logits is
$\sum_a\pi_a r_a(e_a-\pi)=\pi\odot r-\pi\,(\pi^\top r)=(0.219,-0.113,-0.106)$. The total variance at
$b=0$ is $0.126$ and $b^\star=0.384$ (variance $0.036$), matching [@fig:pg:variance].

#### Exercise 34.3.

$b_j=\frac{1}{G-1}\sum_{k\ne j}r_k$ is a function of samples independent of $y_j$, so conditioning
on them and applying [@prop:pg:baseline] gives
$\mathbb{E}[b_j\nabla\log\pi(y_j)]=\mathbb{E}[b_j]\cdot0=0$. Algebra:
$r_j-\frac{S-r_j}{G-1}=\frac{Gr_j-S}{G-1}=\frac{G}{G-1}\big(r_j-\frac{S}{G}\big)$ with
$S=\sum_k r_k$.

#### Exercise 34.4.

With $\kappa(\mu)=\mu$, the limit is $\eta(x)\nabla\eta(x)$, so $w(x)=\operatorname{sg}[\eta(x)]$:
the weight grows linearly with the success rate. The scheme emphasises prompts the model already
solves and nearly ignores hard ones, the opposite of the example in the text. Equivalently it
ascends $\tfrac12\mathbb{E}_x[\eta(x)^2]$.

#### Exercise 34.5.

Total weight per response is (tokens) $\times$ (per-token weight). *Sequence mean:* every response
gets $1/N=0.25$. *Token mean* ($\sum|y|=1860$): $0.016,\,0.016,\,0.161,\,0.806$. *Constant*
($Z=8192$): $0.0037,\,0.0037,\,0.037,\,0.183$. The long wrong answer ($-1$, 1500 tokens) receives
the smallest share of the batch's push under the sequence mean ($0.25$ versus $0.81$ under token
mean), so sequence-mean aggregation makes long wrong answers cheapest to keep.

#### Exercise 34.6.

The exact rule gives every token weight $1/1000$. The buggy rule gives
$\frac12\cdot\frac{1}{100}=1/200$ in the first micro-batch ($5\times$ too large) and
$\frac12\cdot\frac{1}{900}=1/1800$ in the second ($0.56\times$). The error depends on the arbitrary
split, so the same data give different gradients under different micro-batch sizes.

#### Exercise 34.7.

$\chi^2=\sum p_1^2/p_0-1\approx0.0112$. The variance is $(1.0112)^{256}-1\approx16$ at $L=256$ and
$\approx5.7\times10^{19}$ at $L=4096$. Since $\mathbb{E}[w^2]\approx5.7\times10^{19}$ while
$\mathbb{E}[w]=1$, the effective sample size $1024/\mathbb{E}[w^2]$ is far below one: in practice a
single sequence carries essentially all the weight.

#### Exercise 34.8.

Gradient flows for $(1.3,-1)$, $(0.7,+1)$, $(1.1,+1)$ and $(0.9,-1)$; it is masked for $(1.3,+1)$
(already raised beyond $1+\epsilon$) and $(0.7,-1)$ (already lowered below $1-\epsilon$). Clipping
only removes incentive to move *further* in the advantage's direction; moves that undo an over-shoot
are always allowed.

#### Exercise 34.9.

$4\times8192\times152{,}064\times4\,\text{B}\approx19.9$ GB. A chunk of $r$ rows costs
$r\times152{,}064\times4$ B, so $r\le1644$ keeps it under 1 GB; 1024 rows (0.62 GB) is a convenient
choice. Slicing hidden states to response positions before the LM head removes 25% of the LM-head
FLOPs and of the logit memory; with large vocabularies the LM head is a significant share of a
forward pass, so this is a cheap, real saving.

#### Exercise 34.10.

At initialisation $\eta$ is low (about 0.09 on the toy task), so most groups are all-zero and the
group baselines mostly produce zero rows; no-baseline variance is dominated by the few lucky
successes. After 20 GRPO steps $\eta$ is near 0.3, where baselines help most. Expect the
leave-one-out and group-mean variances to match up to the $(G/(G-1))^2$ factor, and both to be
several times lower than no baseline.

#### Exercise 34.11.

$\int_0^1\big(p^G+(1-p)^G\big)dp=\frac{2}{G+1}=2/17\approx11.8\%$ of groups. Pruning removes their
forward and backward passes, about 12% of trainer time when the step is trainer-bound. It cannot
save rollout time: a group's rewards, and therefore whether it is prunable, are known only after its
rollouts have been generated. Saving rollout time requires predicting difficulty in advance
(filtering or curriculum), not pruning.
