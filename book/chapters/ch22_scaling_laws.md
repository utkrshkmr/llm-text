# Scaling Laws in Practice {#ch:scalinglaws}

[@sec:pretrain:scaling] gave us a planning rule: spend a compute budget $C$ on a model of
$N\approx\sqrt{C/120}$ parameters trained on $D\approx 20N$ tokens. That rule is the *output* of a
scaling study somebody else ran, on their data, architecture, optimizer, tokenizer and hardware.
Change any of these and the constants move. This chapter is about running the study yourself: how to
turn a few dozen small training runs into a defensible prediction for one large run you can only
afford to do once.

For a systems engineer, a scaling study is an experiment-design problem under a hard resource
budget. The runs are jobs on a shared cluster. They have queueing delays, reservations, timeouts and
failures, and their wall-clock cost depends on how well each shape uses the hardware. The analysis
is numerical optimisation on a handful of noisy points, and the deliverable is an extrapolation over
several orders of magnitude, where small errors in an exponent become large errors in parameter
count. We will treat all three layers (accounting, fitting and orchestration) with code, and finish
with a complete budgeted study against a synthetic training service that you can run on a laptop.

![Roadmap of this chapter. Steps 1--4 build the theory and the fitting machinery; steps 5--7 cover
the hyperparameter and data questions a real study must answer; steps 8--10 are the systems side:
planning, executing and trusting a study under a wall-clock
budget.](figures/ch22_scaling_laws-1.svg){#fig:scalinglaws:roadmap}

## Why do power laws appear? {#sec:scalinglaws:why}

A *scaling law* is an empirical relation between a resource (data $D$, parameters $N$, compute $C$)
and a quality metric (usually held-out loss) that is a straight line on log-log axes over a wide
range: $$\begin{equation}
L(x) \;=\; E + A\,x^{-\alpha}
\quad\Longleftrightarrow\quad
\log\bigl(L(x)-E\bigr) = \log A - \alpha\log x.
\label{eq:scalinglaws:pl}
\end{equation}$$ The constant $E$ is the irreducible loss: the entropy of the text itself, which no
model can beat. The exponent $\alpha$ says how fast extra resources pay off.

Why a power law rather than an exponential or a logarithm? The cleanest intuition comes from
statistics. Estimating the mean of $n$ samples from $\mathcal N(\mu,\sigma^2)$ has mean-squared
error $\sigma^2/n$, so $\log\text{error} = -\log n + 2\log\sigma$: a power law with slope $-1$. Most
parametric estimators share this $1/n$ rate. Flexible, nonparametric estimators are slower. If you
estimate an arbitrary smooth function on a $d$-dimensional input by averaging over little boxes, you
trade the number of boxes against the samples per box, and the error falls only as $n^{-c/d}$.
Language models behave like the second case: measured data exponents are far smaller than 1 (Kaplan
et al. report roughly $0.1$ for loss versus data). One hypothesis is that the slope reflects the
*intrinsic dimension* of the data manifold (Bahri et al., 2021), but this is an argument for why
power laws are plausible, not a derivation of the numbers. Treat the exponents as things to be
measured.

![Error rates that are polynomial in the sample size are straight lines on log-log axes. The slope
is the exponent; higher-dimensional problems give shallower lines. LM data exponents look like the
shallow end of this picture (analytic curves, constants set to
1).](figures/ch22_scaling_laws-2.svg){#fig:scalinglaws:rates}

::: pitfall
[A straight line only after subtracting $E$]{.box-title}

With an irreducible term, $\log L$ is *not* linear in $\log x$: the curve flattens as $L\to E$.
Fitting a line to $\log L$ directly underestimates $\alpha$ at the small-loss end and overestimates
the payoff of scale when you extrapolate into the region where $E$ dominates. Either fit $E$ jointly
([@sec:scalinglaws:method3]) or stay in the regime where $L-E$ is still large compared with
the noise.
:::

Two more facts from the early literature shape how studies are run. First, *distribution shift*
mostly changes the offset of a scaling curve, not its slope (Hashimoto, 2021): better data moves the
whole line down. Second, data scaling has been observed across modalities and architectures
(Hestness et al., 2017), so the method generalises even when the constants do not.

## What scaling laws are used for {#sec:scalinglaws:uses}

The scaling-law design procedure has three steps: train a family of small models, fit a law to each
candidate choice, and pick the choice whose law is best *at the target scale*. The last clause
matters: a modification that helps at $10^{18}$ FLOPs can have a shallower slope and lose at
$10^{23}$. Kaplan et al. (2020) used this lens to compare transformers with LSTMs (the LSTM curves
bend away at large $N$), Adam with SGD, and different depth-to-width ratios. Three lessons from that
era are still the default assumptions of modern studies:

- **Shape matters little within a broad basin.** At fixed non-embedding parameter count, loss is
  within a few percent over a wide range of aspect ratios $d_{\text{model}}/L$. Studies therefore
  fix the aspect ratio (or a rule for it) and scale one "size" knob.

- **Embedding parameters are different.** Loss tracks *non-embedding* parameters much better than
  total parameters; at small scale the embedding table can be most of the model and would distort
  the fit ([@sec:scalinglaws:accounting]).

- **Hyperparameters that change with scale must be modelled.** Learning rate and batch size have
  scale-dependent optima ([@sec:scalinglaws:hparams]). Using one fixed value for every size
  silently bends the curve.

## Compute accounting: which $N$, which $C$? {#sec:scalinglaws:accounting}

[@sec:pretrain:scaling] derived $C\approx 6ND$: two FLOPs per parameter per token in the
forward pass and four in the backward pass. A scaling study needs this more precisely, because the
fits are only as consistent as the accounting.

For a dense pre-norm transformer with SwiGLU MLPs, width $d$, $L$ layers, MLP width $d_{\text{ff}}$,
vocabulary $V$ and context $T$, the matmul weights per layer are $4d^2$ (Q, K, V, O) plus
$3dd_{\text{ff}}$ (gate, up, down). With the common $d_{\text{ff}}= \tfrac83 d$ this gives the
familiar $$\begin{equation}
N_{\text{ne}} \;=\; L\,(4d^2 + 3dd_{\text{ff}}) \;=\; 12\,L\,d^2 \qquad(d_{\text{ff}}=\tfrac83d),
\label{eq:scalinglaws:12ld2}
\end{equation}$$ non-embedding parameters. Training FLOPs per token add the LM-head matmul ($Vd$
weights) and the attention score and value products, which have no weights but cost $2LTd$ forward
FLOPs per token: $$\begin{equation}
\frac{C}{D} \;\approx\; 6\,(N_{\text{ne}} + Vd) \;+\; 6\,L\,T\,d.
\label{eq:scalinglaws:fpt}
\end{equation}$$ The input embedding is a gather and costs no FLOPs.
[@lst:scalinglaws:runconfig] is the accounting class every tool in this chapter uses.

``` {#lst:scalinglaws:runconfig .python caption="Accounting for one run (from `llmsys/scaling.py`). Everything in this chapter---planning, fitting, the synthetic API---derives $N$ and $C$ from this one place."}
@dataclass(frozen=True)
class RunConfig:
    n_layer: int; d_model: int; n_head: int; d_ff: int; tokens: int
    batch_size: int = 128; lr: float = 3e-3; seq_len: int = 512
    vocab: int = 32_000; tie_embeddings: bool = False

    def non_embedding_params(self):
        d = self.d_model
        return self.n_layer * (4 * d * d + 3 * d * self.d_ff)

    def flops_per_token(self):
        matmul_params = self.non_embedding_params() + self.vocab * self.d_model
        return 6.0 * matmul_params + 6.0 * self.n_layer * self.seq_len * self.d_model

    def train_flops(self):
        return self.flops_per_token() * self.tokens
```

::: napkinmath
[How much of a small model is embedding?]{.box-title}

A model with $L=8$, $d=512$, $d_{\text{ff}}=1408$ and $V=32{,}000$ has
$N_{\text{ne}} = 8(4\cdot512^2 + 3\cdot512\cdot1408) \approx 25.7$M, but its untied input and output
embeddings hold $2\cdot32{,}000\cdot512\approx 32.8$M. More than half the weights are embeddings,
and the LM head alone adds $16.4$M to the FLOP count, about $64\%$ of the non-embedding work. If you
fit a law with total parameters at this scale and use it to extrapolate to a 1B model, where
embeddings are a few percent of the weights, your $N$ axis changes meaning along the way. Pick one
definition (non-embedding is the usual choice), use it for every run, and convert at the end.
:::

::: systemslens
[A wall-clock budget is not a FLOP budget]{.box-title}

Clusters and training services charge in device-hours, not FLOPs. A run of $C$ FLOPs on a device
with peak $P$ takes
$$t \;=\; \frac{C}{P\cdot\mathrm{MFU}(\text{shape},\text{batch})} \;+\; t_{\text{startup}} + t_{\text{eval}},$$
and MFU ([@ch:pretrain]) is far from constant: narrow models launch small matmuls that cannot
fill the tensor cores, small batches leave the GPU waiting on kernel launches, and every run pays a
fixed startup cost (container start, data-loader warm-up, compilation). Consequences:

1.  The tiny runs at the bottom of a study cost more device-time per FLOP than the big ones. Budget
    them in seconds, not FLOPs.

2.  The *wall-clock*-optimal model for a fixed number of device-hours can shift towards shapes with
    high MFU (wider, larger batch) relative to the FLOP-optimal model. The shift is largest where
    MFU still changes quickly with shape ([@fig:scalinglaws:wall]).

3.  Measure MFU for each shape you might pick in the final run. Do not assume it.
:::

## Kaplan versus Chinchilla {#sec:scalinglaws:kaplan}

Given $C$, how should we split it between $N$ and $D$? Two influential answers disagree. Kaplan et
al. (2020) found the loss-optimal model size grows as roughly $N_{\text{opt}}\propto
C^{0.73}$: most extra compute should go into a bigger model, trained on comparatively few tokens.
Hoffmann et al. (2022, "Chinchilla") found $N_{\text{opt}}\propto C^{0.5}$ and
$D_{\text{opt}}\propto C^{0.5}$: scale both equally, at about 20 tokens per parameter. The 70B
Chinchilla model, trained on 1.4T tokens for the same compute as the 280B Gopher model, beat it.

![The two allocation rules, normalised to agree at $10^{18}$ FLOPs. Over seven decades of compute
they disagree by a factor of $10^{7\times0.23}\approx 40$ in model size. A small difference in an
exponent becomes a large difference in what you
build.](figures/ch22_scaling_laws-3.svg){#fig:scalinglaws:kaplan}

Later analyses attribute most of the gap to methodology rather than to the models (Porian et al.,
2024, and Pearce and Song, 2024, give careful reconciliations). The main suspects:

- **Learning-rate schedules.** Kaplan's runs used one schedule length and read off intermediate
  losses for shorter token counts. A cosine schedule that has not finished decaying reports a loss
  well above what a run of that length would reach, which penalises long-data configurations.
  Chinchilla matched the schedule to each run's length. Our synthetic API reproduces this effect
  ([@sec:scalinglaws:wsd]).

- **Which $N$.** Kaplan counted non-embedding parameters and omitted the LM head from compute; at
  small scale that bends the curve.

- **Scale.** Kaplan's runs were smaller, where the corrections above matter most.

The lesson for practitioners is not "Chinchilla is right". It is that the allocation exponent is a
measured quantity whose value depends on choices you make in the study, so you must make those
choices deliberately and consistently.

## Three ways to fit an allocation law {#sec:scalinglaws:methods}

Hoffmann et al. estimated $N_{\text{opt}}(C)$ three ways. All three are worth knowing, because they
fail differently.

### Method 1: the lower envelope of training curves {#sec:scalinglaws:method1}

Train models of several sizes, each for a long run, and record loss against FLOPs consumed along the
way. For each compute value, the best model is whichever curve is lowest there: the *lower envelope*
([@fig:scalinglaws:envelope]). The envelope's contact points give $N_{\text{opt}}(C)$.
Method 1 is cheap (one run per size), but it needs the intermediate losses to be valid stand-ins for
completed runs, and with a cosine schedule they are not. A warmup--stable--decay schedule fixes this
([@sec:scalinglaws:wsd]).

![Method 1. Each grey curve is one model size ($3\times10^6$ to $3\times10^9$ non-embedding
parameters) trained for longer and longer; the dashed red line is their lower envelope. Data: the
noise-free part of the synthetic ground truth in `llmsys.scaling`, not a real
model.](figures/ch22_scaling_laws-4.svg){#fig:scalinglaws:envelope}

### Method 2: IsoFLOP profiles {#sec:scalinglaws:method2}

Pick a handful of compute budgets $C_1<C_2<\dots$. For each budget, train several model sizes
$N_{ij}$, each on exactly $D_{ij}=C_i/(6N_{ij})$ tokens with a schedule matched to that length, and
record the final loss $L_{ij}$. Plotted against $\log N$, each budget's losses form a valley: too
small a model cannot use the data, too large a model cannot take enough steps
([@fig:scalinglaws:isoflop]). The bottom of each valley is $N_{\text{opt}}(C_i)$; a power
law through those points is the allocation law.

![Method 2 on the synthetic ground truth. Left: four IsoFLOP profiles ($10^{17}$--$10^{20}$ FLOPs);
dots are noisy "runs", lines the noise-free valley, stars the minima. Right: the minima and the
fitted power law, extrapolated three decades. The fitted exponent recovers the hidden
$\beta/(\alpha+\beta)=0.453$ closely because the grid brackets every
minimum.](figures/ch22_scaling_laws-5.svg){#fig:scalinglaws:isoflop}

There are two ways to read off the minimum. Taking the best run is robust but quantised to the sizes
you tried. With sizes spaced by a factor $r$, the true optimum can be up to $\sqrt r$ away. Fitting
a parabola in $\log N$ and taking its vertex interpolates between grid points, but it trusts the
parabola, which is only a local approximation to the valley. The vertex is only meaningful if the
grid brackets it: if the best run sits at the edge of the grid, the budget's grid was mis-centred,
and the right response is to add runs, not to extrapolate the parabola.
[@lst:scalinglaws:isoflop] implements both readings with that guard.

``` {#lst:scalinglaws:isoflop .python caption="IsoFLOP minima with a bracketing check (abridged from `llmsys/scaling.py`)."}
def isoflop_optima(runs, method="min", rel_tol=0.05):
    out = []
    for c, group in group_by_budget(runs, rel_tol).items():  # budgets drift
        best = min(group, key=lambda r: r.loss)
        n_opt, loss, used = best.params, best.loss, "min"
        if method == "quadratic" and len(group) >= 3:
            ln = np.log([g.params for g in group]); mu = ln.mean()
            c2, c1, c0 = np.polyfit(ln - mu, [g.loss for g in group], 2)
            v = mu - c1 / (2 * c2)
            if c2 > 0 and ln.min() <= v <= ln.max():   # convex, bracketed
                n_opt, loss = math.exp(v), c0 - c1**2 / (4*c2)
                used = "quadratic"
        ...
```

::: pitfall
[Budgets are never exact]{.box-title}

Token counts must be whole numbers of batches, and model sizes come in multiples of the head
dimension, so the "$10^{18}$-FLOP" runs actually spend $0.98$--$1.02\times10^{18}$. Group runs by
*relative* tolerance, and use each group's mean compute when you fit the power law. Never use the
nominal budget. A 2% bias in $C$ is invisible on a plot but shifts the intercept of every
extrapolation.
:::

### Method 3: a parametric loss surface {#sec:scalinglaws:method3}

Instead of locating minima, fit a formula for the whole surface, $$\begin{equation}
\hat L(N,D) \;=\; E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},
\label{eq:scalinglaws:chin}
\end{equation}$$ to *every* run, and then minimise it analytically. Minimising $\hat L$ subject to
$6ND = C$ with a Lagrange multiplier gives $$\begin{equation}
N_{\text{opt}}(C) = G\Bigl(\frac{C}{6}\Bigr)^{\frac{\beta}{\alpha+\beta}},\qquad
D_{\text{opt}}(C) = G^{-1}\Bigl(\frac{C}{6}\Bigr)^{\frac{\alpha}{\alpha+\beta}},\qquad
G = \Bigl(\frac{\alpha A}{\beta B}\Bigr)^{\frac{1}{\alpha+\beta}}.
\label{eq:scalinglaws:closed}
\end{equation}$$ The derivation is three lines ([@ex:scalinglaws:lagrange]). The exponents
satisfy $a+b=1$, as they must when $C\propto ND$. Equal exponents $\alpha=\beta$ give the Chinchilla
$C^{0.5}$ rule; a smaller data exponent $\beta$ pushes compute towards data.

The fit itself is a small but delicate optimisation problem. Hoffmann et al. minimised a Huber loss
on the *log* of the loss, with the surface written as a log-sum-exp so that every parameter is
unconstrained: $$\begin{equation}
\min_{a,b,e,\alpha,\beta}\;\sum_i \mathrm{Huber}_\delta\Bigl(
\mathrm{LSE}\bigl(a-\alpha\log N_i,\; b-\beta\log D_i,\; e\bigr) - \log L_i\Bigr),
\qquad A=e^a,\;B=e^b,\;E=e^e,
\end{equation}$$ with $\delta=10^{-3}$, from a grid of initial points, keeping the best. The Huber
loss down-weights outlier runs, such as one with a loss spike. Working in log space treats relative
errors equally at all scales. The grid of starts is not optional: with few runs the objective is
nearly flat along directions that trade $E$ against $A$ and $B$, and local optimisers stop in very
different places.

``` {#lst:scalinglaws:chin .python caption="Huber/LSE fit of the parametric law from a grid of starts (from `llmsys/scaling.py`)."}
def fit_chinchilla(n, d, loss, delta=1e-3, inits=None, max_iter=100):
    ln, ld, ll = (torch.tensor(np.log(v), dtype=torch.float64)[None]
                  for v in (n, d, loss))
    inits = inits or [(a0, b0, e0, al, be) for a0 in (0., 5., 10.)
                      for b0 in (0., 5., 10.) for e0 in (-1., .5)
                      for al in (.1, .4) for be in (.1, .4)]          # 72 starts
    best, best_obj = None, float("inf")
    for init in inits:
        q = torch.tensor([init], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([q], max_iter=max_iter,
                                line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            a, b, e, al, be = (q[:, i:i+1] for i in range(5))
            pred = torch.logsumexp(torch.stack(
                [a - al*ln, b - be*ld, e.expand(-1, ln.shape[1])]), 0)
            obj = huber(pred - ll, delta).sum()
            obj.backward(); return obj
        opt.step(closure)
        ...   # keep the start with the lowest objective and alpha, beta > 0
```

::: deepdive
[When the three methods disagree]{.box-title}

In the Chinchilla paper, methods 1 and 2 agreed ($a\approx0.50$) and method 3 gave a somewhat
different exponent. A later replication attempt (Besiroglu et al., 2024) reconstructed the data from
the paper's figures and reported that the published method-3 parameters fit the data poorly and
imply implausibly narrow confidence intervals. Their refit moved method 3 into agreement with the
other two. The practical moral: report all the methods you can afford, look at the residuals, and
bootstrap ([@sec:scalinglaws:uncertainty]) before believing a third significant digit of
anything.
:::

## Hyperparameters across scale {#sec:scalinglaws:hparams}

A scaling law describes *well-tuned* models. If every size uses the same learning rate, the small
models are under-stepped or the large ones over-stepped, and the measured curve mixes the scaling of
the architecture with the mis-tuning of the optimizer. Modern recipes handle this in one of two
ways: fit scaling laws for the hyperparameters themselves, or reparameterise the model so that the
optimal hyperparameters stop depending on size.

### Learning rate {#sec:scalinglaws:lr}

For a fixed size and token budget, final loss as a function of $\log(\text{LR})$ is a smooth,
roughly parabolic basin with a cliff on the high side, where training diverges
([@fig:scalinglaws:basin]). Its minimum moves to smaller learning rates as models get
wider. The procedure is direct: sweep the LR at three or four small sizes, take each basin's vertex
(a quadratic in $\log\text{LR}$, ignoring diverged runs), and fit $\text{lr}^*(N)=kN^{a}$.
DeepSeek's LLM report (2024) is a public example of fitting both LR and batch size as power laws in
compute this way, and a large 2025 hyperparameter study from StepFun mapped the LR--batch landscape
by brute-force grids and found it convex with clean minimisers.

![Learning-rate basins on the synthetic ground truth for three model sizes, each trained on 20
tokens per parameter. Vertical lines mark each optimum; it drifts down as the model grows. Curves
end where training diverges. A study that uses one LR for all sizes reads the *wrong* point of each
basin.](figures/ch22_scaling_laws-6.svg){#fig:scalinglaws:basin}

### Batch size and the critical batch {#sec:scalinglaws:batch}

Doubling the batch halves the number of optimizer steps but, past some point, does not halve the
number of steps *needed*. McCandlish et al. (2018) model the trade-off between the optimizer steps
$S$ and examples $E$ needed to reach a target loss as $$\begin{equation}
\Bigl(\frac{S}{S_{\min}}-1\Bigr)\Bigl(\frac{E}{E_{\min}}-1\Bigr) = 1,
\qquad B_{\text{crit}} = \frac{E_{\min}}{S_{\min}},
\label{eq:scalinglaws:critbatch}
\end{equation}$$ a hyperbola ([@fig:scalinglaws:critbatch]). Below $B_{\text{crit}}$,
larger batches are almost free: the same data, fewer steps. Above it, each doubling of the batch
wastes most of the extra data. At $B=B_{\text{crit}}$ you pay exactly twice the minimum of both. The
critical batch is approximately the *gradient noise scale*: the ratio of the trace of the
per-example gradient covariance to the squared norm of the mean gradient. It grows as the loss
falls, so bigger and longer runs can use larger batches. That is why large-scale schedules ramp the
batch size up during training.

![The steps--examples trade-off of [@eq:scalinglaws:critbatch] (analytic). The critical
batch is the knee, where you pay $2\times$ the minimum of each. Systems want to sit at or slightly
above it: fewer, larger steps parallelise across more devices
([@ch:distributed]).](figures/ch22_scaling_laws-7.svg){#fig:scalinglaws:critbatch}

::: systemslens
[Batch size is where the optimizer meets the cluster]{.box-title}

Data parallelism needs a large global batch: with 1,024 GPUs and 4 sequences of 4,096 tokens each,
one step already consumes 16.8M tokens. The critical batch says whether that is efficient. If
$B_{\text{crit}}$ is 4M tokens at your loss, a 16M-token batch wastes most of the extra data, and
the honest choice is fewer GPUs per run or more model parallelism
([@ch:distributed; @ch:sharding]). A scaling study that sweeps batch size at small scale
tells you how far data parallelism can be pushed at large scale.
:::

### $\mu$P: make the optimum stop moving {#sec:scalinglaws:mup}

The maximal-update parameterisation ($\mu$P; Yang et al., 2022) takes the other route. It chooses
per-layer initialisation scales and learning-rate multipliers so that, as width $n$ grows, (A1)
every activation coordinate stays $\Theta(1)$ at initialisation, and (A2) one optimizer step changes
every activation coordinate by $\Theta(1)$, no more and no less. For a hidden weight matrix
$W\in\mathbb{R}^{n\times n}$ the activation condition gives the familiar $\sigma\propto1/\sqrt{n}$
initialisation. The update condition is the new part. With Adam, the update $\Delta W$ has entries
of size about $\eta$, its product with an activation vector of $\Theta(1)$ coordinates sums $n$
correlated terms, and so the output changes by $\Theta(\eta n)$. Keeping that $\Theta(1)$ requires
$\eta\propto 1/n$ for hidden layers, while the input embedding keeps a width-independent rate and
the output layer's logits are scaled down by $1/n$. Under these rules the LR basin of
[@fig:scalinglaws:basin] stays put as width grows, so you tune once on a narrow proxy model
and transfer ("$\mu$Transfer").

::: keyidea
[Two public recipes]{.box-title}

Two well-documented public recipes make a useful contrast. **MiniCPM** (2024) uses $\mu$P to make
hyperparameters width-invariant, fixes the aspect ratio, and fits the loss surface (method 3) with
cheap WSD branches. **DeepSeek LLM** (2024) uses standard parameterisation, fits optimal LR and
batch size as power laws in compute, and sizes the model with IsoFLOP profiles (method 2). Both
work; what they share is that *every* scale-dependent knob is either modelled or parameterised away.
:::

## Making studies cheap: warmup--stable--decay {#sec:scalinglaws:wsd}

Method 2 needs a separate run for every (size, token count) pair, because a cosine schedule's loss
at 50% of training is not the loss a run of half the length would reach: the LR has not decayed yet.
Fitting $m$ sizes at $k$ durations therefore costs $mk$ full runs, and if you want the durations to
cover the same range as the sizes, the cost of a study grows like $n^2$.

The *warmup--stable--decay* (WSD) schedule (used by MiniCPM, and as a multi-step variant by
DeepSeek) removes the coupling. Warm up, hold the LR constant for most of training, and decay
quickly over the last $\sim$`<!-- -->`{=html}10--20% of steps. Because the stable phase does not
depend on the final length, you can save a checkpoint at any point of it and *branch*: run a short
decay from that checkpoint and get, for the price of the decay, a loss that matches a full run of
that length closely ([@fig:scalinglaws:wsd]). One long stable run per size plus a few cheap
decays replaces $k$ separate runs.

::: {#fig:scalinglaws:wsd .figure}
![](figures/ch22_scaling_laws-8.svg) ![](figures/ch22_scaling_laws-9.svg)

[Left: cosine versus WSD learning-rate schedules (analytic). Right: branching decays off a WSD
stable phase gives a loss at several data sizes from one run.]{.figure-caption}
:::

::: napkinmath
[What branching saves]{.box-title}

Suppose you want losses for one model at token counts $D, 2D, 4D, 8D$. With cosine you need four
runs: $15D$ tokens. With WSD and decays of 10% of each length, you need one $8D$ run plus decays of
$0.1D+0.2D+0.4D+0.8D=1.5D$, so $9.5D$ tokens in total, a 37% saving. With many log-spaced durations
the saving approaches 40%, and the stable run doubles as the method-1 curve. The cost is
infrastructure: you must checkpoint the full optimizer state at every branch point, restore it
exactly, and track which branch produced which loss, which is exactly the bookkeeping of
[@sec:pretrain:save] and [@sec:scalinglaws:running].
:::

The synthetic API of this chapter models the cosine effect. Each run reports losses at
$k=1..n_{\text{evals}}$ evenly spaced points, and the early ones sit above the loss of a finished
run of that length by an amount proportional to the not-yet-decayed LR. Reading those intermediate
numbers as completed runs biases the fitted data exponent, which is the Kaplan pitfall in miniature
([@ex:scalinglaws:midrun]).

## Beyond the basic law {#sec:scalinglaws:beyond}

##### Finite data and repetition.

Chinchilla assumes fresh tokens. When unique data runs out, repeated epochs are worth less.
Muennighoff et al. (2023) fit an "effective data" model in which the value of repeated tokens decays
with the number of repetitions. Up to about four epochs the loss is close to what the same number of
unique tokens would give, and beyond that the returns fall off sharply. Data-constrained regimes
push the optimum towards larger models trained on fewer epochs.

##### Data mixtures.

The proportions of web, code, books and so on can themselves be treated as scaling-law inputs.
Fitting mixture laws on small proxies is an active research area and a hard one: small-scale
rankings of mixtures do not always survive scaling, and simply picking the best mixture at small
scale is a strong baseline. [@ch:data] covers how the data itself is built.

##### Inference-aware scaling.

Chinchilla minimises *training* compute. If a model will serve $D_{\text{inf}}$ tokens, total cost
is about $6ND + 2ND_{\text{inf}}$, and the optimum moves to smaller models trained on more data
(Sardana and Frankle, 2023). This is why open models are routinely "over-trained" far past 20 tokens
per parameter ([@sec:pretrain:scaling]).

##### Downstream metrics.

Loss scales smoothly; task accuracy often does not. Some "emergent" jumps are artefacts of
discontinuous metrics such as exact match: a smoothly improving per-token probability crosses a
threshold (Schaeffer et al., 2023). Others are genuinely hard to predict. Practical studies predict
loss with a scaling law and map loss to downstream metrics with a separate, empirical fit (Llama 3
reports such a two-step prediction).

##### Mixture-of-experts.

With MoE ([@sec:modern:moe]) total and active parameters differ. Studies fit laws in active
parameters and treat sparsity (experts per token over total experts) as an extra axis. Several
recent reports size their MoE models with IsoFLOP sweeps over active parameters.

## Designing a budgeted study {#sec:scalinglaws:design}

Now the systems problem. You have a remote training service that accepts a model and recipe, runs it
on one accelerator, and returns validation losses. It charges a wall-clock budget (say 12
device-hours) for fitting, and you must name the configuration and predict the loss for a final run
of 48 device-hours. This is the setting of any team that is allotted a slice of a cluster to decide
what to train next, and of Project 3 ([@ch:proj3]).

### The service model {#sec:scalinglaws:service}

`llmsys.scaling.SyntheticTrainingAPI` implements a service of this kind in-process. Its semantics
follow common shared-cluster policies:

- `submit(cfg, max_runtime_seconds)` validates the config, rejects exact duplicates, and *reserves*
  the full `max_runtime_seconds` against your budget while the run is queued or running.

- Runs execute FIFO on $W$ parallel workers in simulated time. When one finishes, the charge becomes
  its actual runtime, clipped to $[1, \text{max\_runtime}]$, and the unused reservation is refunded.

- A run that needs longer than its reservation *times out*: you are charged the full reservation and
  get back only the losses evaluated before the cut-off.

- An LR far above the optimum diverges: losses end in `NaN`, and you pay for the time until
  divergence is detected.

Behind it sits a hidden loss surface: [@eq:scalinglaws:chin], with $N$ the non-embedding
parameters, data discounted by large batches, a quadratic penalty in
$\log(\text{LR}/\text{lr}^*(N))$, a small aspect-ratio penalty, deterministic per-seed noise, and
the mid-run cosine bump. Its runtime model is the one in the Systems Lens of
[@sec:scalinglaws:accounting], with MFU rising with width and batch.

![A 12-run IsoFLOP grid ($10^{16}$, $4\times10^{16}$, $1.6\times10^{17}$ FLOPs, four sizes each)
executed by the synthetic service on three workers. Light bars are each run's reservation
($1.25\times$ the predicted runtime), dark bars its actual runtime; the difference is refunded at
completion, but while the run is in flight it is unavailable for new submissions. Data: simulated by
`llmsys.scaling`.](figures/ch22_scaling_laws-10.svg){#fig:scalinglaws:gantt}

::: systemslens
[Reservations are a concurrency limit]{.box-title}

Because queued and running jobs hold their full reservation, the number of experiments you can have
*in flight* is bounded by $\text{remaining budget}/\text{reservation}$, not by the cluster.
Over-reserving (a safety factor of $3\times$ "just in case") does not cost budget in the end, but it
serialises your study and stretches it over more calendar time. Under-reserving costs budget: a
timed-out run is charged in full and returns nothing usable. Predict runtimes from your accounting
and a measured MFU, add a modest margin ($1.2$--$1.5\times$), and calibrate the margin on the first
few completed runs.
:::

### A staged plan {#sec:scalinglaws:stages}

A study that spends its budget well proceeds in stages, each small enough to revise the next:

1.  **Dry run and accounting.** Submit the smallest sensible config once. Check that the service
    accepts your configs, that the losses look sane ($\ln V$ at initialisation, a steady fall), and
    that the measured runtime matches your prediction. This costs minutes and catches unit errors
    that would waste hours.

2.  **Hyperparameter transfer.** Sweep the LR (and, if you can afford it, the batch) at three or
    four small sizes; fit $\text{lr}^*(N)$. Fix the aspect ratio by a rule.

3.  **IsoFLOP grid.** Choose 3--5 budgets spaced by $3$--$4\times$, 5--7 sizes each, centred on your
    current best guess, wide enough ($8$--$16\times$ in $N$) to bracket the minimum. Use the fitted
    LR for each size.

4.  **Fit and diagnose.** Fit methods 2 and 3; check residuals; bootstrap the allocation exponent.
    If a budget's minimum is at the edge of its grid, spend more there.

5.  **Verification.** Spend part of the remaining budget on one run at the largest budget you can
    afford, at the predicted optimum. Compare its loss with the prediction. This is the single most
    informative run in the study.

6.  **Final choice.** Choose the final config by searching configurations that *finish* in the
    target wall-clock time (MFU varies with shape) under the fitted law, and report the predicted
    loss with an interval.

::: napkinmath
[Where a geometric grid spends its budget]{.box-title}

With $m$ sizes per budget and budgets $C_{\max}, C_{\max}/r, C_{\max}/r^2,\dots$, the FLOPs spent
are $mC_{\max}(1+1/r+1/r^2+\dots)\approx mC_{\max}\,r/(r-1)$. With $r=4$ the largest budget is
$75\%$ of the grid, and with $r=3$ it is $67\%$. The top budget is what you are really buying, so
choose it first: given 12 hours, a 7-run top budget at $\sim$`<!-- -->`{=html}55% of the study
leaves room for the smaller budgets, the LR sweep and a verification run. Then check the leverage:
your largest budget sits $\log_{10}(C_{\text{final}}/C_{\max})$ decades below the target, and every
decade of extrapolation multiplies exponent error into $N$ error.
:::

``` {#lst:scalinglaws:plan .python caption="Planning an IsoFLOP grid and pricing it in wall-clock seconds (from `llmsys/scaling.py`)."}
def plan_isoflop_grid(budgets, sizes_per_budget=5, span=8.0,
                      hardware=SyntheticHardware(), seconds_available=math.inf,
                      safety=1.25, make_config=config_for_params, n_evals=8):
    plan, spent = [], 0.0
    for c in sorted(budgets):                 # cheapest budgets first
        n_mid = math.sqrt(c / (6 * 20))       # centre on 20 tokens/param
        ns = np.geomspace(n_mid / math.sqrt(span), n_mid * math.sqrt(span),
                          sizes_per_budget)
        rows = []
        for n in ns:
            cfg = make_config(n)              # shape + fitted LR
            cfg = replace(cfg, tokens=tokens_for_budget(cfg, c))
            rows.append((cfg, safety * hardware.runtime(cfg, n_evals)))
        cost = sum(r for _, r in rows)
        if spent + cost > seconds_available:
            break                             # never plan past budget
        plan.extend(rows); spent += cost
    return plan
```

### Choosing the final run when the budget is wall-clock {#sec:scalinglaws:wallchoice}

The closed form ([@eq:scalinglaws:closed]) answers "best $N$ for $C$ FLOPs". A wall-clock
budget answers a different question, because $C$ itself depends on the shape through MFU. The robust
approach is a search. Enumerate candidate sizes and batch sizes, compute how many tokens each can
process in the time limit at its MFU, predict its loss with the fitted surface, and keep the best
that finishes ([@lst:scalinglaws:choose]). [@fig:scalinglaws:wall] shows the
difference on the synthetic hardware.

``` {#lst:scalinglaws:choose .python caption="Wall-clock-aware final choice ([@lab:scalinglaws], Task 9)."}
def choose_final_config(fit, lr_law, seconds, hw, sizes=None,
                        batch_sizes=(128, 256, 512), aspect=64.0):
    sizes = sizes if sizes is not None else np.geomspace(1e7, 3e9, 60)
    k, a = lr_law
    best = None
    for n in sizes:
        for bs in batch_sizes:
            cfg = config_for_params(n, batch_size=bs, aspect=aspect)
            n_ne = cfg.non_embedding_params()
            cfg = replace(cfg, lr=k * n_ne ** a)
            flops = hw.flops_in(cfg, seconds)   # depends on MFU(shape)
            cfg = replace(cfg, tokens=tokens_for_budget(cfg, flops))
            if hw.runtime(cfg) > seconds:
                continue                    # must actually finish
            pred = float(fit.loss(n_ne, cfg.tokens))
            if best is None or pred < best[1]:
                best = (cfg, pred)
    return best
```

![The synthetic hardware model. Left: training FLOPs each model size can actually spend in a 48-hour
slot (blue) versus what a constant, saturated MFU would give (orange, dashed). Right: the MFU curve
behind it. Narrow models lose up to about half their FLOPs to low MFU, so a wall-clock budget
penalises them relative to a FLOP budget. Near this study's optimum (about $5\times10^8$ parameters)
the curve has flattened, and the wall-clock and FLOP optima almost coincide; for smaller budgets
they do not. Simulated, not measured.](figures/ch22_scaling_laws-11.svg){#fig:scalinglaws:wall}

### How sure are we? Bootstrapping the extrapolation {#sec:scalinglaws:uncertainty}

Every number in a scaling study comes from a few dozen noisy runs, and the quantity you care about
(the size at $10^{20}$ FLOPs) is an extrapolation. Quantify the uncertainty by resampling. Resample
runs with replacement *within each budget*, which keeps the IsoFLOP structure; redo the whole fit;
and collect the predictions ([@fig:scalinglaws:fan]). The spread grows with distance from
the data, at a rate set by the exponent's uncertainty.

![A 12-hour synthetic study (four budgets up to $3\times10^{17}$, six sizes each) and a 150-sample
bootstrap. The band is the 5--95% interval of the extrapolated $N_{\text{opt}}$; at $10^{20}$ FLOPs
it spans $4.1\times10^8$--$11.3\times10^8$ parameters, almost a factor of three, from a fit whose
individual points look very clean. Simulated with
`llmsys.scaling`.](figures/ch22_scaling_laws-12.svg){#fig:scalinglaws:fan}

::: napkinmath
[Exponent error becomes size error]{.box-title}

If the fitted exponent is off by $\delta a$, the predicted size at a budget $k$ decades beyond your
data is off by a factor $10^{k\,\delta a}$. With $\delta a=0.03$ (a typical bootstrap standard
deviation for a small study) and $k=2.5$ decades, that is $10^{0.075}\approx1.19$, roughly a 20%
error in $N$. The good news is that IsoFLOP valleys are flat near the bottom: a model 20% off the
optimum loses only a small fraction of a percent of loss. The bad news is that the *loss prediction*
inherits the error in $E$, which is the least well-determined parameter of method 3. Predict the
loss from the fit, then sanity-check it against the trend of your IsoFLOP minima
$L_{\text{opt}}(C)$.
:::

### Reading a study log {#sec:scalinglaws:log}

[@lst:scalinglaws:log] is the output of the reference solution of Lab 15 on the default
synthetic service: 12 simulated hours of fitting budget, a 48-hour target. Read it the way you would
read a real study.

``` {#lst:scalinglaws:log caption="Output of `run.py study` (synthetic service, seed 0; numbers abridged)."}
lr law      : lr*(N) = 0.341 * N^-0.255
fit         : E=1.769 A=513.8 B=493.2 alpha=0.356 beta=0.290
allocation  : N_opt ~ C^0.407   D_opt ~ C^0.749   L_opt ~ C^-0.116
final config: L=23 d=1472 heads=23 d_ff=3904 batch=512 lr=1.98e-3
              tokens=1.46e10  (N_ne = 0.60e9, ~24 tokens/param)
predicted   : 2.7509   (after verification residual)
budget      : used 32,925 s of 43,200 s
score       : true 2.7453   oracle 2.7412   regret 0.0041   pred. error +0.0056
```

Four things to notice. The learning-rate law falls with size (exponent $-0.25$; the hidden truth
uses $-0.30$), so a single LR for all sizes would have been wrong at both ends. The parametric fit
recovers the hidden exponents to within $0.01$ ($0.35$ and $0.29$ in the truth), while the IsoFLOP
allocation exponents do not add to one ($0.41+0.75$). That gap is the accounting effect of
[@sec:scalinglaws:accounting]: $D$ was computed with the full FLOP count, including the LM
head and attention, while $N$ is non-embedding, so the two power laws are fits of different
identities. The choice itself is excellent (regret $0.004$ nats). Finally, before the verification
run the same fit predicted $2.711$, an error of $-0.034$. One run at the largest affordable scale
cut that error by a factor of six, because $E$ was the least constrained parameter of the fit and
the verification residual corrects it directly. A study without a verification run reports a loss it
has never seen.

::: pitfall
[Five ways scaling studies go wrong]{.box-title}

1.  The grid does not bracket the minimum at the largest budget, which is the one that matters most.

2.  One learning rate is used for every size.

3.  Intermediate (mid-schedule) losses are fitted as if they were finished runs.

4.  $and$ use inconsistent definitions across runs or between the fit and the final config.

5.  The final prediction is reported with no interval and no verification.
:::

## Running real sweeps {#sec:scalinglaws:running}

The synthetic service lets you practise the analysis. On real hardware the same study is mostly an
exercise in orchestration. A layout that scales from a laptop to a cluster:

    study/
      configs/          one YAML/JSON per run, generated by a script (never hand-edited)
      ledger.jsonl      append-only: submitted request, job id, result, timestamps
      runs/<run_id>/    checkpoints, metrics.jsonl, stdout, the exact config + git hash
      analysis/         fits, bootstrap draws, plots -- regenerated from ledger.jsonl
      plan.md           the staged plan and every decision, with dates

Three rules make this robust. **Generate configs from code.** The planner of
[@lst:scalinglaws:plan] writes the files, and its output is versioned with the study.
**Make submission idempotent.** Key every run by a hash of its canonical config, and refuse to
resubmit a key that is already in the ledger. A crashed notebook or a re-run cell then costs nothing
(the `Ledger` in `code/projects/p3_scaling/ledger.py` does exactly this). **Derive all analysis from
the ledger**, never from a notebook's in-memory state.

On a Slurm cluster, an IsoFLOP grid is a natural job array: one array per budget, one task per size,
each task reading its config by index. This gives every task the same time limit, which corresponds
to the reservation above.

``` {#lst:scalinglaws:slurm .bash caption="One IsoFLOP budget as a Slurm job array (each task trains one size)."}
#!/bin/bash
#SBATCH --job-name=iso-1e18
#SBATCH --array=0-6                 # seven sizes
#SBATCH --gres=gpu:1 --cpus-per-task=8
#SBATCH --time=02:30:00             # reservation = 1.25 x predicted runtime
#SBATCH --output=runs/%x-%a.out
CFG=$(ls configs/iso-1e18/*.json | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
srun python train.py --config "$CFG" --metrics runs/$(basename "$CFG" .json)/metrics.jsonl
```

Log every run to one experiment tracker project (for example W&B, or the JSONL logger of
[@ch:optim]), grouped by budget, with the config hash as the run name. Then the IsoFLOP plot
is a query, not a manual export. Seed the model initialisation but keep the data order fixed across
runs, so that differences between sizes are not differences between data samples. Finally, record
*achieved throughput* for every run: it is the input to the wall-clock planning of the next study.
[@lab:scalinglaws]'s `run.py cpu-sweep` is a laptop-sized version of this pipeline that trains real byte-level
Qwen3-shaped models and fits what it measures.

## Summary {#sec:scalinglaws:summary}

- Scaling laws are power laws plus an irreducible term; they are measured, not derived, and their
  constants depend on data, architecture, optimizer and accounting choices.

- Account consistently: non-embedding $N$, $C/D\approx6(N_{\text{ne}}+Vd)+6LTd$, and actual (not
  nominal) compute per run.

- Kaplan and Chinchilla disagree mostly because of schedules, accounting and scale; the allocation
  exponent is sensitive, and its errors grow with extrapolation distance.

- Fit allocation laws with IsoFLOP minima (bracketed, min or vertex) and with the parametric surface
  (Huber in log space, many starts); compare them, and bootstrap.

- Model or remove scale dependence in LR and batch size: fit $\text{lr}^*(N)$ and respect the
  critical batch, or use $\mu$P.

- WSD schedules with branched decays make durations cheap, if the checkpointing infrastructure is
  right.

- A study is a budgeted experiment on a cluster: reservations limit concurrency, MFU makes
  wall-clock differ from FLOPs, and ledgers, staged plans and a verification run keep it honest.

## Exercises {#sec:scalinglaws:exercises}

::: {#ex:scalinglaws:lagrange .exercise}
\[Closed-form allocation\] Derive [@eq:scalinglaws:closed] by minimising
[@eq:scalinglaws:chin] subject to $6ND=C$. Show that at the optimum the two reducible terms
are in the fixed ratio $\alpha A N^{-\alpha} : \beta B D^{-\beta} = 1:1$, and interpret that ratio.
:::

::: exercise
[IsoFLOP minima lie on a power law]{.box-title}

Show that the minima of the IsoFLOP profiles of [@eq:scalinglaws:chin] satisfy
$N_{\text{opt}}\propto C^{\beta/(\alpha+\beta)}$ exactly. Then add an embedding term to the FLOP
count, $C=6(N+Vd)D$, and argue qualitatively how the measured exponent changes at small scale.
:::

::: exercise
[Accounting for your own models]{.box-title}

For Qwen3-0.6B ([@ch:modern]) compute $N_{\text{ne}}$, the embedding fraction, and $C/D$ from
[@eq:scalinglaws:fpt] at $T=4096$. How large is the attention term relative to $6N$? Which
of the 12$Ld^2$ assumptions does Qwen3 violate (GQA, $d_{\text{ff}}/d$, head dimension)?
:::

::: exercise
[Quantisation error of the min method]{.box-title}

Sizes are spaced by a factor $r$ on a valley that is locally parabolic in $\log N$ with curvature
$\kappa$. Bound the error in $\log N_{\text{opt}}$ and in $L_{\text{opt}}$ of the min-of-runs
reading, and compare it with the noise level at which a parabola fit stops helping.
:::

::: exercise
[Critical batch arithmetic]{.box-title}

From [@eq:scalinglaws:critbatch], show that training with batch $B$ needs
$S=S_{\min}(1+B_{\text{crit}}/B)$ steps and $E=E_{\min}(1+B/B_{\text{crit}})$ examples. If a step
takes time $t_0 + B/\text{throughput}$, find the batch that minimises wall-clock time to the target
loss, and explain why it is larger than $B_{\text{crit}}$ when $t_0$ is large.
:::

::: {#ex:scalinglaws:midrun .exercise}
\[Mid-run losses are not finished runs\] Using `SyntheticTrainingAPI`, train one size for a long run
with 16 evaluations and treat the intermediate losses as runs of shorter length. Fit the data
exponent $\beta$ from them and compare it with a fit from separate runs of each length. Explain the
bias using the cosine schedule, and say how WSD branching would remove it.
:::

::: exercise
[Cost of a WSD study]{.box-title}

For $m$ sizes and $k$ log-spaced durations (ratio 2), compare the total tokens of a cosine study
(one run per pair) with a WSD study (one longest run per size plus decays of a fraction $f$ of each
length). Give the asymptotic saving as $k\to\infty$.
:::

::: exercise
[$\mu$P in one layer]{.box-title}

For a linear layer $y=Wx$ with $x\in\mathbb{R}^n$, $\|x\|_\infty=\Theta(1)$, show that an Adam
update (entries of magnitude $\approx\eta$) aligned with $x$ changes $y$ by $\Theta(\eta n)$. Derive
the $\eta\propto1/n$ rule, and explain why the same argument gives a different rule for SGD.
:::

::: exercise
[Inference-aware optimum]{.box-title}

Minimise $\hat L(N,D)$ subject to $6ND + 2ND_{\text{inf}} = C_{\text{total}}$ for fixed
$D_{\text{inf}}$. Show that the optimal $N$ is smaller than the Chinchilla optimum, and compute the
optimal $N$ and tokens-per-parameter ratio for the default synthetic truth (`GroundTruth()`) at
$C_{\text{total}}=10^{21}$ and $D_{\text{inf}}=10^{11}$.
:::

::: exercise
[Reservation policy]{.box-title}

Runtime predictions have a log-normal error with $\sigma=0.15$. A timed-out run is charged its
reservation $s\cdot\hat t$ and returns nothing; a completed run is charged its true runtime. Find
the safety factor $s$ that minimises the expected budget per useful run, and compare it with the
factor that minimises calendar time when at most $R/(s\hat t)$ runs can be in flight.
:::

::: exercise
[Replicate the study]{.box-title}

Run [@lab:scalinglaws]'s `run.py replicate --seeds 10`. Report the distribution of regret and prediction error.
Which stage of your plan would you change to reduce the spread, and what would it cost?
:::

## Lab 16: A budgeted scaling study {#sec:scalinglaws:lab}

::: labbox
[A budgeted scaling study against a synthetic training service]{.box-title}

**Code:** `code/labs/lab22_scaling_laws`

**Goals**

Build every piece of a scaling study (accounting, power-law and parametric fits, LR transfer,
uncertainty, wall-clock-aware choice) and run it end to end against a service with a hidden ground
truth, a wall-clock budget, reservations, timeouts and divergence.

**Setup**

``` bash
cd code
LAB_IMPL=starter  python -m pytest labs/lab22_scaling_laws    # your work
LAB_IMPL=solution python -m pytest labs/lab22_scaling_laws    # reference
```

The service is `llmsys.scaling.SyntheticTrainingAPI`; it is provided, not a task.

**Tasks**

1.  `count_params_and_flops`: [@eq:scalinglaws:12ld2; @eq:scalinglaws:fpt]; the test
    checks your count against an actual `Qwen3Model`.

2.  `fit_power_law`: least squares in log-log space.

3.  `isoflop_optima`: min-of-runs and bracketed parabola vertex per budget.

4.  `fit_allocation`: $N_{\text{opt}}(C)$, $D_{\text{opt}}(C)$, $L_{\text{opt}}(C)$.

5.  `fit_chinchilla`: Huber/LSE fit from a grid of starts.

6.  `chinchilla_optimal`: [@eq:scalinglaws:closed].

7.  `fit_lr_optimum`, `fit_lr_scaling`: LR basin vertex and $\text{lr}^*(N)$.

8.  `bootstrap_ci`: percentile intervals for any fitted quantity.

9.  `choose_final_config`: best config that finishes in the time limit.

10. `run_study`: the staged plan of [@sec:scalinglaws:stages], within budget, including a
    verification run whose residual corrects the loss prediction, ending with `api.submit_final`.
    The test requires low regret against the oracle.

**Measurements**

- `run.py study --plot study.png`: your IsoFLOP profiles and allocation fit; compare with
  [@fig:scalinglaws:isoflop].

- `run.py replicate --seeds 10`: regret and prediction-error distributions
  ([@ex:scalinglaws:midrun] and the replication exercise).

- Ablate: a fixed LR for all sizes instead of $\text{lr}^*(N)$; min versus vertex readings; no
  verification run. Which ablation hurts regret most? Which hurts the loss prediction?

- `run.py cpu-sweep`: a real IsoFLOP sweep with tiny models. Expect noisy exponents and explain
  where the noise comes from (data, seeds, the overhead-bound MFU of tiny models).

**Deliverables**

Passing tests; a one-page write-up with your plan, budget usage, both fits, a bootstrap interval for
the final $N$ and loss, and the score from `api.score`.

**Stretch goals**

Add WSD branching to the synthetic service (a `branch(eid, at_fraction)` endpoint that charges only
the decay) and redesign the study around method 1; add a batch-size sweep and fit
$B_{\text{opt}}(C)$.
:::
