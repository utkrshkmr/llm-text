# Optimizers and the Training Loop from First Principles {#ch:optim}

Part II built a model. It cannot do anything until something moves its weights, and that something
is a surprisingly large system: a sampler that pulls windows out of a token file holding billions of
entries, a numerically careful loss, an optimizer that keeps two extra copies of every parameter, a
learning-rate schedule that must know in advance when the run ends, a clipping step that guards
against rare bad batches, a checkpointer that has to survive a crash, and a logger that records
enough for you to work out later what happened. Each piece is a few dozen lines. Getting all of them
right, and fast, separates a training script from a training *system*.

This chapter builds every one of those pieces from first principles: SGD, momentum, AdaGrad,
RMSProp, Adam and AdamW derived and implemented as `torch.optim.Optimizer` subclasses; Lion and Muon
as modern alternatives; warmup, cosine and warmup--stable--decay schedules; global-norm clipping; a
resumable memory-mapped batch sampler; atomic, bit-exact checkpoints; and a small
experiment-management layer. All of it lives in `llmsys/optim.py` and `llmsys/explog.py`.
[@ch:pretrain] then puts these pieces into the GPT-2 pretraining loop and adds mixed
precision, activation checkpointing and MFU. Throughout, we treat the optimizer step as what it is
on a GPU: an elementwise, memory-bound kernel whose cost we can predict to within a few percent.

![Roadmap of this chapter. Stages 1--4 are the mathematics and cost of the update rule; stages 5--8
are the plumbing that turns an update rule into a training run you can
trust.](figures/ch13_optimizers-1.svg){#fig:optim:roadmap}

## The training loop as a system {#sec:optim:loop}

Every training step does the same seven things ([@fig:optim:loop]): fetch a batch, compute
logits, reduce them to a scalar loss, backpropagate, possibly clip the gradient, update the
parameters (and the optimizer's state), and occasionally evaluate, log and checkpoint. In code:

``` {#lst:optim:skeleton .python caption="The skeleton every training loop in this book follows."}
for step in range(start_step, total_steps):
    set_lr(opt, schedule(step))               # 1. schedule
    x, y = next(batches)                      # 2. data (B, T) int64 on device
    logits = model(x)                         # 3. forward  (B, T, V)
    loss = cross_entropy(logits, y)           # 4. loss     scalar
    opt.zero_grad(set_to_none=True)
    loss.backward()                           # 5. backward
    gnorm = clip_grad_norm_(model.parameters(), 1.0)   # 6. clip
    opt.step()                                # 7. update p and its state
    if step % log_every == 0:
        logger.log(step, loss=loss.item(), grad_norm=gnorm)
    if (step + 1) % ckpt_every == 0:
        ckpt.save(step + 1, model, opt, batches)
```

Two properties of this loop matter more than any single line in it. First, it must be *resumable*: a
job on a shared cluster will be preempted, so the state that defines "where we are" (weights,
optimizer moments, step, data position, random-number generators) must be saved and restored
completely. Second, it must be *observable*: when a run goes wrong after twelve hours you need the
gradient norm, learning rate and loss at every step. [@sec:optim:ckpt; @sec:optim:explog]
build both properties.

![Dataflow of one training step. Solid arrows run every step; dashed arrows run every few hundred
steps. The checkpoint must capture everything on the left of the dashed line, including the
sampler's random state, or a resumed run will not reproduce the
original.](figures/ch13_optimizers-2.svg){#fig:optim:loop}

::: systemslens
[Where the time goes in the loop]{.box-title}

On a GPU the forward and backward passes are matrix multiplies and dominate the step: roughly $6N$
FLOPs per token for an $N$-parameter model ([@ch:gpt]). Everything else in
[@lst:optim:skeleton] is elementwise or host-side work and should cost a few percent:

- the optimizer step reads and writes every parameter, gradient and moment once (memory-bound,
  [@sec:optim:cost]);

- clipping needs one reduction over all gradients and a host synchronisation if you turn the norm
  into a Python float;

- `loss.item()` and `float(gnorm)` each force the CPU to wait for the GPU, so call them only on
  logging steps or you serialise the asynchronous pipeline of [@ch:pytorch];

- batch sampling runs on the host and must overlap with GPU work ([@sec:optim:data]).

If a profile shows any of these at more than 5--10% of step time, it is a bug, not a fundamental
cost.
:::

## The objective, numerically {#sec:optim:loss}

The language-modelling loss is the average negative log-likelihood of the next token,
$$\begin{equation}
\ell(\theta) = -\frac{1}{BT}\sum_{b,t}\log \operatorname{softmax}(\boldsymbol{o}_{b,t})_{y_{b,t}}
= \frac{1}{BT}\sum_{b,t}\Bigl(\operatorname{logsumexp}(\boldsymbol{o}_{b,t}) - o_{b,t,y_{b,t}}\Bigr),
\label{eq:optim:ce}
\end{equation}$$ where $\boldsymbol{o}_{b,t}\in\mathbb{R}^{V}$ are logits.
[@sec:pretrain:loss] derives this and gives a from-scratch implementation
([@lst:pretrain:ce]). Three numerical rules make it safe:

1.  **Never form $\exp(o)$ of raw logits.** With bf16 logits of magnitude 90,
    $e^{90}\approx 10^{39}$ overflows fp32. Subtract the row maximum first:
    $\operatorname{logsumexp}(\boldsymbol{o}) = m + \log\sum_u e^{o_u - m}$ with $m=\max_u o_u$. The
    same trick makes softmax stable, and it is the identity behind FlashAttention's online softmax
    ([@sec:attention:flash]).

2.  **Cancel $\log$ against $\exp$ symbolically.** Computing $\operatorname{softmax}$ then $\log$
    loses precision for tiny probabilities (and gives $\log 0=-\infty$); the right-hand form of
    [@eq:optim:ce] never materialises a probability.

3.  **Do the reduction in fp32.** The $(B,T,V)$ logits tensor is the largest activation in a small
    model, and averaging $BT$ terms in bf16 loses about three significant digits. Upcast the logits
    (or let a fused kernel do it) before the logsumexp.

::: keyidea
[Know your loss at initialisation]{.box-title}

A freshly initialised model with small output weights predicts an almost uniform distribution over
$V$ tokens, so the first loss should be close to $\ln V$: about 6.24 for $V=512$ and 10.83 for
$V=50{,}257$. A first loss of 30 means the logits are too large (initialisation or a missing final
norm); a first loss of exactly $\ln V$ to six digits often means the output is constant (dead
network, zeroed head). Perplexity $e^{\ell}$ starts near $V$ and should fall quickly.
:::

The vocabulary dimension also makes the loss a *memory* problem: the logits for $B\cdot T=16{,}384$
tokens and $V=151{,}936$ (Qwen3) are 2.5 G elements, or 10 GB in fp32. Production code therefore
computes the loss in chunks over the sequence, or uses fused "linear + cross-entropy" kernels that
never store the full logits. [@ch:distill] implements chunked cross-entropy for long
reasoning traces.

## Writing an optimizer in PyTorch {#sec:optim:api}

PyTorch's `torch.optim.Optimizer` base class does three jobs for us. It stores *parameter groups*
(lists of tensors that share hyperparameters), it keeps a per-tensor *state* dictionary
(`self.state[p]`), and it serialises both through `state_dict()`/`load_state_dict()`. A subclass
only has to implement `__init__` (pass the parameters and a `defaults` dictionary to `super()`) and
`step`. [@lst:optim:sgd] is SGD with momentum, written the way PyTorch defines it so that
our test can compare against `torch.optim.SGD` to six digits.

``` {#lst:optim:sgd .python caption="SGD with heavy-ball or Nesterov momentum (from `llmsys/optim.py`)."}
class SGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0, nesterov=False, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay))

    @torch.no_grad()                        # the update itself is not part of the graph
    def step(self, closure=None):
        loss = None
        if closure is not None:             # API contract: optional loss re-evaluation
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:     # each group may have its own lr / wd
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:          # frozen or unused parameter
                    continue
                g = p.grad if wd == 0 else p.grad.add(p, alpha=wd)
                if mu != 0:
                    st = self.state[p]      # persistent, saved by state_dict()
                    if "momentum_buffer" not in st:
                        st["momentum_buffer"] = g.clone()
                    else:
                        st["momentum_buffer"].mul_(mu).add_(g)
                    buf = st["momentum_buffer"]
                    g = g.add(buf, alpha=mu) if group["nesterov"] else buf
                p.add_(g, alpha=-lr)        # in-place: no new tensor, no graph
        return loss
```

Several details in this listing are easy to get wrong:

- **In-place updates under `no_grad`.** `p.add_` mutates the parameter's storage, so every reference
  to it (including one inside a tied module) sees the change. Writing `p = p - lr*g` would only
  rebind a local name. Some older code writes to `p.data`, which also works but bypasses autograd's
  version counter; the decorator is the modern idiom.

- **State lives in `self.state[p]`, keyed by the tensor.** That is what makes `state_dict()` work:
  PyTorch maps tensors to integer indices in group order, so a checkpoint can be loaded into a
  freshly built model *if parameters are created in the same order*. Reordering modules between save
  and load scrambles optimizer state silently.

- **Parameter groups are how you vary hyperparameters.** The usual LLM recipe puts matrices in a
  group with weight decay and gains/biases in a group without it (`decay_param_groups` in the
  library), and schedules write `group["lr"]` on every step ([@sec:optim:sched]).

- **`zero_grad(set_to_none=True)`** frees the gradient tensors rather than filling them with zeros.
  That saves a memset kernel per parameter, and peak memory between steps.

::: pitfall
[A learning rate that is "too small" can look like a bug]{.box-title}

On a toy quadratic you can watch three regimes as you scale the learning rate up by factors of 10:
the loss falls slowly, falls fast, oscillates with growing amplitude, and finally diverges to `inf`
within a few steps. For gradient descent on $\tfrac12\lambda x^2$ the boundary is exactly
$\eta\lambda = 2$: the update multiplies $x$ by $(1-\eta\lambda)$, whose magnitude exceeds 1 beyond
that point. A real network has a whole spectrum of curvatures $\lambda_i$, and the largest one (the
"sharpness") sets the stability limit; this is the origin of the folk wisdom that the best learning
rate sits just below the "edge of stability" ([@sec:optim:debug]).
:::

## From SGD to Adam and AdamW {#sec:optim:adam}

Each classic optimizer adds one idea to SGD. Writing $g_t$ for the gradient at step $t$:

::: description
$b_t = \mu b_{t-1} + g_t$, $\theta_t = \theta_{t-1} - \eta b_t$. Averaging gradients over roughly
$1/(1-\mu)$ steps damps oscillation across steep directions and accumulates speed along shallow
ones. Nesterov's variant evaluates the step "ahead" by using $g_t + \mu b_t$.

Keep $s_t = s_{t-1} + g_t^2$ (elementwise) and step $-\eta\, g_t/\sqrt{s_t+\epsilon}$. Each
coordinate gets its own learning rate, inversely proportional to its accumulated gradient size.
Because $s_t$ only grows, the effective step size decays to zero, which is good for convex problems
and bad for long non-convex training.

Replace the sum with an exponential moving average, $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$, so
the per-coordinate scale tracks the *recent* gradient magnitude.

(Kingma and Ba, 2015) is RMSProp with momentum on the numerator,
$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$, plus a bias correction.
:::

##### Why Adam needs bias correction.

With $m_0=0$, unrolling the recursion gives $m_t = (1-\beta_1)\sum_{k=1}^{t}\beta_1^{t-k} g_k$. If
the gradients have a stationary mean $\bar g$, then
$\mathbb{E}[m_t] = (1-\beta_1)\bar g \sum_{k=1}^{t}\beta_1^{t-k} = (1-\beta_1^t)\,\bar g$. The
estimate is biased towards zero early in training, by a factor of 10 at $t=1$ when $\beta_1=0.9$.
Dividing by $(1-\beta_1^t)$ removes the bias, and likewise $(1-\beta_2^t)$ for $v_t$, where the
effect is larger: with $\beta_2=0.999$ the uncorrected second moment would be 1000 times too small
at step 1, and the first steps would be enormous. The full update is $$\begin{equation}
\hat m_t = \frac{m_t}{1-\beta_1^t},\qquad \hat v_t = \frac{v_t}{1-\beta_2^t},\qquad
\theta_t = \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\label{eq:optim:adam}
\end{equation}$$ At $t=1$, $\hat m_1 = g_1$ and $\hat v_1 = g_1^2$, so the first update is
$-\eta\,\mathrm{sign}(g_1)$ in every coordinate whose gradient is much larger than $\epsilon$.
Adam's step size is bounded by roughly $\eta$ per coordinate regardless of the gradient's scale.
That property is the main reason it tolerates the badly scaled gradients of transformers, whose
embedding, attention and MLP gradients differ by orders of magnitude.

##### Weight decay: coupled versus decoupled.

Classic L2 regularisation adds $\tfrac{\lambda}{2}\|\theta\|^2$ to the loss,
i.e. $g_t\leftarrow g_t+\lambda\theta$. For SGD that is identical to shrinking the weights by
$(1-\eta\lambda)$ each step. For Adam it is not: the decay term passes through the $1/\sqrt{\hat v}$
normalisation, so parameters with large gradients are barely decayed and parameters with tiny
gradients are decayed hard. Loshchilov and Hutter (2019) proposed *decoupling* the decay from the
adaptive update: $$\begin{equation}
\theta_t = (1-\eta\lambda)\,\theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\label{eq:optim:adamw}
\end{equation}$$ This is AdamW, the default optimizer for LLM pretraining. Note that the decay is
multiplied by the *scheduled* learning rate $\eta$, so it also anneals when the learning rate does;
when you read a config that says "wd = 0.1", the per-step shrink factor at peak learning rate
$3\times10^{-4}$ is $3\times10^{-5}$.

``` {#lst:optim:adamw .python caption="The AdamW update for one parameter tensor (loop path of `llmsys.optim.AdamW`). It matches `torch.optim.AdamW` exactly, including the placement of $\\epsilon$ after the bias-corrected square root."}
st["step"] += 1
t = st["step"]
if wd != 0:
    if group["decoupled"]:
        p.mul_(1 - lr * wd)                    # AdamW: shrink, ignores g
    else:
        g = g.add(p, alpha=wd)                 # Adam + L2: decay via g
m, v = st["exp_avg"], st["exp_avg_sq"]
m.lerp_(g, 1 - b1)                             # m = b1*m + (1-b1)*g
v.mul_(b2).addcmul_(g, g, value=1 - b2)        # v = b2*v + (1-b2)*g*g
denom = (v.sqrt() / math.sqrt(1 - b2 ** t)).add_(eps)
p.addcdiv_(m, denom, value=-lr / (1 - b1 ** t))
```

![Twenty-five steps of three optimizers on $f(\theta)=\tfrac12(\theta_1^2+25\theta_2^2)$ starting
from $(-4,1.5)$, computed with `llmsys.optim`. SGD's learning rate is capped by the steep direction
($\eta\lambda_{\max}=1.9$), so it zig-zags and crawls along the shallow one. Momentum accumulates
speed along $\theta_1$. Adam normalises each coordinate, so its first steps move diagonally at speed
$\eta$ in both directions; it overshoots and spirals
in.](figures/ch13_optimizers-3.svg){#fig:optim:traj}

##### Hyperparameters that matter for LLMs.

Typical pretraining settings are $\beta_1=0.9$, $\beta_2=0.95$, $\epsilon=10^{-8}$, weight decay
$0.1$, and a peak learning rate between $10^{-4}$ and $10^{-3}$ for 0.1--10B-parameter models
(smaller models tolerate larger rates). Lowering $\beta_2$ from 0.999 to 0.95 shortens the
second-moment memory from about 1000 steps to 20. When a rare large gradient arrives, $v$ catches up
within a few steps instead of leaving the effective learning rate wrong for hundreds; in practice
this is one of the cheapest stability fixes for large runs. $\epsilon$ matters only for coordinates
whose gradients are tiny; if you keep $v$ in bf16, $\epsilon=10^{-8}$ underflows relative to
$\sqrt{v}$ and behaves like zero.

## What an optimizer step costs {#sec:optim:cost}

The forward and backward passes are compute-bound matrix multiplies. The optimizer step is the
opposite: a long chain of elementwise operations with an arithmetic intensity below one FLOP per
byte. Its time is set entirely by how many bytes it moves.

::: napkinmath
[AdamW step time for a 1.1B-parameter model]{.box-title}

Take a Llama-style model with $d=2048$, $L=22$, GQA with 32 query and 4 key/value heads,
$d_{\text{ff}}=5632$, $V=32{,}000$ and untied embeddings: $N\approx1.10\times10^9$ parameters
(`llmsys.flops.params`). In fp32, one AdamW step reads $p, g, m, v$ (16 bytes per parameter) and
writes $p, m, v$ (12 bytes): **28 bytes per parameter**. Standard mixed precision moves the same
amount: read bf16 $g$ (2), fp32 master, $m$, $v$ (12); write fp32 master, $m$, $v$ (12) and the bf16
working copy (2). That is $$28 \times 1.10\times10^{9} \approx 30.8\ \text{GB per step}
\;\Rightarrow\; \frac{30.8\ \text{GB}}{3.35\ \text{TB/s}} \approx 9.2\ \text{ms on one H100}.$$ The
arithmetic is about 18 FLOPs per parameter, $2\times10^{10}$ FLOPs in total, which takes 0.02 ms at
the H100's peak. A training step on a batch of, say, 0.5M tokens costs
$6N\cdot 5\times10^5 = 3.3\times10^{15}$ FLOPs, about 8 s at 40% MFU, so the optimizer is around
0.1% of the step. If your profile shows a much larger share, the update is not one memory pass: it
is hundreds of small kernels.
:::

That last case is the usual one for a naive implementation. The loop in [@lst:optim:adamw]
launches about ten kernels per parameter tensor, and a transformer has hundreds of tensors (a
28-layer model has around 300). Many are small (norm gains have $d$ elements), so the step can
become *launch-bound*: thousands of kernel launches at a few microseconds each. PyTorch solves this
in two steps:

- **Multi-tensor ("foreach") kernels** take a *list* of tensors and process all of them in one
  launch, chunking the work across thread blocks. Our `AdamW(foreach=True)` path
  ([@lst:optim:foreach]) issues about ten launches per parameter *group* instead of per
  tensor, and the test checks it matches the loop to floating-point precision.

- **Fused kernels** (`torch.optim.AdamW(fused=True)` on CUDA) do the whole update for a chunk of
  parameters in registers: one read and one write of each tensor, the 28 bytes per parameter of the
  napkin estimate.

``` {#lst:optim:foreach .python caption="The same AdamW update over a whole parameter group with multi-tensor kernels."}
torch._foreach_mul_(params, 1 - lr * wd)                 # decoupled decay
torch._foreach_lerp_(ms, grads, 1 - b1)                  # m
torch._foreach_mul_(vs, b2)
torch._foreach_addcmul_(vs, grads, grads, value=1 - b2)  # v
denoms = torch._foreach_sqrt(vs)
torch._foreach_div_(denoms, [math.sqrt(1 - b2 ** t) for t in steps])
torch._foreach_add_(denoms, eps)
upd = torch._foreach_div(ms, denoms)
torch._foreach_mul_(upd, [-lr / (1 - b1 ** t) for t in steps])
torch._foreach_add_(params, upd)
```

![Bytes of model state per parameter under common precision recipes (activations excluded). Both
fp32 and standard mixed precision cost 16 B/param; pure-bf16 states halve that but lose precision in
$v$ and in small weight updates; 8-bit optimizers (Dettmers et al., 2022) quantise $m$ and $v$
block-wise. [@ch:sharding] removes the redundancy across GPUs
instead.](figures/ch13_optimizers-4.svg){#fig:optim:statebytes}

::: systemslens
[Optimizer state is the largest tensor you own]{.box-title}

For a model trained with AdamW, $m$ and $v$ together are twice the size of the fp32 weights, so the
optimizer state is the largest consumer of memory in the job. This drives three design decisions you
will meet later: sharding optimizer state across data-parallel ranks (ZeRO-1,
[@ch:sharding]), offloading it to CPU memory, and lower-precision or factored states (8-bit
Adam; Adafactor's row/column factorisation of $v$, [@sec:optim:beyond]). It also explains
why checkpoints are about three times the size of the released model: a 7B-parameter model releases
14 GB of bf16 weights but writes 112 GB of training checkpoint.
:::

## Accounting for memory and compute in a training run {#sec:optim:account}

Before launching a run you should be able to predict whether it fits in memory, how long a step
takes, and how many hours the whole run needs. The method has four parts, each a sum over named
tensors or matrix multiplies.

##### 1. Parameters, gradients, optimizer state.

Count parameters by listing every weight matrix ([@sec:gpt:accounting] does it for GPT-2,
[@ch:modern] for Qwen3). Then multiply by bytes per parameter for each copy
([@fig:optim:statebytes]). For our 1.1B example: $1.10\times10^9\times16 = 17.6$ GB of
model state.

##### 2. Activations.

Walk through one block and list every tensor that autograd saves for the backward pass, with its
shape. For a pre-norm SwiGLU block with naive attention, on a batch of $B$ sequences of length $T$:

::: center
| saved tensor                                        | elements           |
|:----------------------------------------------------|:-------------------|
| inputs to the two RMSNorms                          | $2BTd$             |
| normalised input to $W_Q,W_K,W_V$                   | $BTd$              |
| $Q$, $K$, $V$ (GQA)                                 | $BT(H+2G)d_{h}$    |
| attention scores and probabilities                  | $2BHT^2$           |
| attention output (input to $W_O$)                   | $BTHd_{h}$         |
| normalised input to the MLP                         | $BTd$              |
| gate and up projections; SiLU output; their product | $4BTd_{\text{ff}}$ |
:::

Add the final norm, the logits ($BTV$) and the softmax needed by the loss's backward. Which tensors
are actually saved depends on the implementation: fused kernels and FlashAttention save fewer.
[@ch:perf] shows how to measure them with saved-tensor hooks.

##### 3. FLOPs.

$2mnp$ for every $(m\times n)(n\times p)$ product, summed over the forward pass; the backward pass
costs about twice the forward, giving the familiar $6N$ per token plus the attention term
$\approx 12 L H d_{h}T$ per token (forward + backward) that grows with context. The optimizer adds
$\approx18N$ FLOPs per step, independent of batch size.

##### 4. Time.

Divide total FLOPs by achievable throughput, i.e. peak FLOP/s times the model-FLOPs utilisation
(MFU) you expect (35--50% is good for dense transformers).

::: napkinmath
[Fitting and timing the 1.1B example]{.box-title}

With $T=2048$ in bf16 and naive attention, the table gives about $3.4\times10^8$ elements, or
0.68 GB, per layer per sequence, and 15.5 GB per sequence for all 22 layers plus logits. The
$2BHT^2$ attention matrices are 79% of that. With FlashAttention (which saves only the output and a
log-sum-exp per row) it falls to about 0.145 GB per layer, 3.7 GB per sequence. On an 80 GB GPU with
17.6 GB of model state, the activation budget allows about 16 sequences of 2048 tokens per
micro-batch with FlashAttention, but only 4 with naive attention. Either way the peak memory is
linear in batch size, $\text{mem}(B) = a\cdot B + b$ with $b$ the model state, which is the form you
should always derive.

Compute: $6N + 12LHd_{h}T \approx 6.6 + 0.7 = 7.3$ GFLOPs per token at $T=2048$. Training on
$10^{11}$ tokens costs $\approx 7.3\times10^{20}$ FLOPs; on 8 H100s at 40% MFU
($8\times 989\times10^{12}\times0.4 = 3.2\times10^{15}$ FLOP/s) that is about 64 hours.
:::

::: pitfall
["Float32" peak numbers]{.box-title}

Vendor tables often quote a tensor-core number for "FP32" that is really TF32: 10-bit mantissa
inputs with fp32 accumulation. On an H100 SXM that figure is about half the bf16 dense peak, and
true IEEE fp32 matrix multiplies run at a much lower rate. When you compute MFU, use the peak that
matches the arithmetic you actually run and say which one it is. With
`torch.set_float32_matmul_precision("high")` PyTorch uses TF32 for fp32 matmuls on Ampere and later
GPUs.
:::

## Learning-rate schedules {#sec:optim:sched}

A schedule is a function $\eta(t)$ evaluated every step and written into `group["lr"]`. Three shapes
cover almost all LLM training ([@fig:optim:sched]).

::: description
Ramp from near zero to $\eta_{\max}$ over $T_w$ steps (typically 0.1--2% of training, often
1000--2000 steps). Adam's second-moment estimate is noisy for the first $\sim 1/(1-\beta_2)$ steps,
and at initialisation the residual stream and attention logits are far from their trained scales.
Large early steps at full learning rate can push the model into a regime it never recovers from
(loss spikes, dead attention heads).

$\eta(t) = \eta_{\min} + \tfrac12(\eta_{\max}-\eta_{\min})
    \bigl(1+\cos(\pi\frac{t-T_w}{T_c-T_w})\bigr)$ for $T_w\le t\le T_c$, then constant
$\eta_{\min}$. It is the default since GPT-3 and Chinchilla. The run length $T_c$ is baked in: if
you stop early you get a model trained at a high learning rate, and if you continue past $T_c$ you
train at $\eta_{\min}$ forever. When you change the step budget (for example to run on a smaller
machine), set $T_c$ to the new final step.

Hold $\eta_{\max}$ constant for most of training and decay over the last 10--20%, linearly or as
$1-\sqrt{x}$ (Hu et al., 2024; Hägele et al., 2024). Because the long stable phase does not depend
on the final length, you can branch decay phases off a single run at several token budgets.
[@ch:scalinglaws] uses this to make scaling-law sweeps several times cheaper.
:::

![The schedules implemented in `llmsys.optim`, for a 1000-step run with 50 warmup steps and a floor
of $0.1\,\eta_{\max}$.](figures/ch13_optimizers-5.svg){#fig:optim:sched}

``` {#lst:optim:sched .python caption="Schedules are plain callables, so they are trivial to test, plot and checkpoint."}
@dataclass
class WarmupCosine:
    peak: float; total: int; warmup: int = 0; floor: float = 0.0
    def __call__(self, step):
        if self.warmup > 0 and step < self.warmup:
            return self.peak * (step + 1) / self.warmup      # step 0 already moves
        if step >= self.total:
            return self.floor
        prog = (step - self.warmup) / max(1, self.total - self.warmup)
        return self.floor + 0.5 * (self.peak - self.floor) * (1 + math.cos(math.pi * prog))

sched = WarmupCosine(peak=3e-4, total=20_000, warmup=500, floor=3e-5)
for step in range(start, 20_000):
    set_lr(opt, sched(step))      # writes every param group (times an optional per-group scale)
```

Two conventions differ between codebases and cause off-by-one disagreements. Is the first step 0 or
1? Does warmup reach $\eta_{\max}$ at step $T_w-1$ or $T_w$? Ours is 0-based and reaches the peak at
step $T_w-1$. Any convention is fine if your tests pin it down, and if a resumed run calls the
schedule with the restored step rather than a step counter that restarted at zero.

We measured the WSD branching idea on CPU with Lab 12's `run.py wsd` command
([@fig:optim:wsd]).

![One WSD trunk with three decay branches versus three separate cosine runs, at budgets of 200, 400
and 600 steps (same model and data as [@fig:optim:lrsweep], $\eta=3\times10^{-3}$; measured
on CPU, one seed). The branches end at 1.579, 1.505 and 1.488, against 1.638, 1.512 and 1.486 for
the cosine runs, so they match or beat them. They cost 720 optimizer steps in total (480 trunk +
40 + 80 + 120 decay) instead of 1200. The loss falls steeply as each branch decays: much of the
benefit usually credited to a schedule's shape comes from annealing at the
end.](figures/ch13_optimizers-6.svg){#fig:optim:wsd}

## Gradient clipping {#sec:optim:clip}

Occasionally a batch produces a gradient far larger than usual: a rare token, a pathological
document, or the model entering a sharp region. One such step can undo hours of progress.
Global-norm clipping rescales the *entire* gradient vector when its norm exceeds $M$:
$$\begin{equation}
\boldsymbol{g} \leftarrow \boldsymbol{g}\cdot\min\Bigl(1, \frac{M}{\|\boldsymbol{g}\|_2+\epsilon}\Bigr),
\qquad \|\boldsymbol{g}\|_2 = \Bigl(\sum_{\text{tensors }i}\|\boldsymbol{g}_i\|_2^2\Bigr)^{1/2}.
\label{eq:optim:clip}
\end{equation}$$ Clipping the global norm preserves the gradient's *direction*; clipping each tensor
(or each element) separately would change it ([@fig:optim:clip]). $M=1.0$ is the
near-universal LLM default. With Adam, clipping matters less for the direction (Adam normalises per
coordinate anyway) than for the moments: a huge $g$ would inflate $v$ for the next
$\sim 1/(1-\beta_2)$ steps.

![Global-norm clipping (green) shrinks the gradient onto the ball of radius $M$ and keeps its
direction. Clipping each coordinate (or each tensor) to $[-M,M]$ (dashed) changes the direction,
here from $31^\circ$ to $45^\circ$.](figures/ch13_optimizers-7.svg){#fig:optim:clip}

``` {#lst:optim:clipcode .python caption="Global-norm clipping with multi-tensor kernels. It returns the *pre-clip* norm, the single most useful scalar to log."}
@torch.no_grad()
def clip_grad_norm_(parameters, max_norm, eps=1e-6, foreach=True):
    grads = [p.grad for p in parameters if p.grad is not None]
    norms = torch._foreach_norm([g.float() for g in grads])   # fp32 norms
    total = torch.linalg.vector_norm(torch.stack(norms))      # global ||g||
    scale = max_norm / (total + eps)
    if scale < 1:                     # NOTE: syncs GPU -> CPU
        torch._foreach_mul_(grads, scale.to(grads[0].dtype))
    return float(total)
```

::: systemslens
[Clipping is a global reduction]{.box-title}

[@eq:optim:clip] needs a norm over *all* parameters before any of them can be scaled. On
one GPU that is one reduction kernel, but turning the result into a Python `bool` or `float` forces
a host--device synchronisation. PyTorch's own implementation avoids the branch by always multiplying
by `clamp(scale, max=1)` on the device. When parameters are sharded across GPUs (FSDP, tensor
parallelism, [@ch:sharding; @ch:distributed]), each rank holds only part of $\boldsymbol{g}$,
so clipping needs an all-reduce of the squared partial norms. That is one more collective per step,
and the place where many custom parallel trainers get the semantics subtly wrong by clipping each
shard separately. With fp16 loss scaling ([@sec:pretrain:mixed]), always *unscale*
gradients before computing the norm.
:::

## Beyond AdamW: Lion, Adafactor, Shampoo and Muon {#sec:optim:beyond}

AdamW has been the default for LLMs since about 2019, but it is not the only option, and recent
speed-running efforts on small transformers have favoured alternatives. A short survey, from
simplest to most involved:

::: description
(Chen et al., 2023) was found by program search: take the *sign* of an interpolation between
momentum and gradient, $\theta \leftarrow \theta -
    \eta\,\mathrm{sign}(\beta_1 m + (1-\beta_1) g)$, then update $m$ with a second rate $\beta_2$.
It needs one state tensor instead of two (12 vs 16 bytes/param in mixed precision) and a learning
rate 3--10$\times$ smaller than AdamW's, because every coordinate moves by exactly $\eta$.

(Shazeer and Stern, 2018) stores the second moment of an $n\times m$ matrix as a rank-1
factorisation (row and column sums, $n+m$ numbers instead of $nm$), with optional update clipping
and relative step sizes. It was the memory-saving default for T5 and PaLM.

keep a left and a right preconditioner per weight matrix,
$\theta\leftarrow\theta-\eta\,L^{-1/4} G R^{-1/4}$ with $L=\sum GG^\top$ and $R=\sum G^\top G$. This
approximates full-matrix AdaGrad for the matrix's Kronecker structure. The inverse roots are
expensive and are recomputed every few steps, often on separate hardware; SOAP (Vyas et al., 2024)
runs Adam in Shampoo's eigenbasis.

(Jordan et al., 2024) keeps a momentum buffer for each hidden weight matrix and replaces the update
by its nearest semi-orthogonal matrix: if the momentum is $U\Sigma V^\top$, the update is $UV^\top$,
which equalises all singular directions. The orthogonalisation uses a few Newton--Schulz iterations
of an odd polynomial ([@fig:optim:ns]), five matrix multiplies on the smaller Gram matrix,
so it runs in bf16 on tensor cores. Embeddings, the LM head and all 1-D parameters stay on AdamW.
:::

![The Newton--Schulz polynomial $p(\sigma)=a\sigma+b\sigma^3+c\sigma^5$ with Muon's coefficients
$(3.4445,-4.7750,2.0315)$, iterated. It acts on each singular value independently because
$X\mapsto aX+b(XX^\top)X+c(XX^\top)^2X$ is an odd matrix polynomial. After five iterations every
singular value above $\sim$`<!-- -->`{=html}0.005 lands in the shaded band $[0.68,1.2]$: not exactly
1, but close enough, and the coefficients were tuned for speed of reaching the band rather than
exact convergence. Our test measures $[0.68,1.13]$ on a random $32\times16$
matrix.](figures/ch13_optimizers-8.svg){#fig:optim:ns}

``` {#lst:optim:ns .python caption="Newton--Schulz orthogonalisation, the core of Muon."}
def newton_schulz_orthogonalize(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    if X.shape[0] > X.shape[1]:            # work with the smaller Gram matrix
        X = X.T
    X = X / (X.norm() + eps)               # ||X||_F >= sigma_max, so all sigma <= 1
    for _ in range(steps):
        A = X @ X.T                        # (r, r), r = min(m, n)
        X = a * X + (b * A + c * A @ A) @ X
    return (X.T if G.shape[0] > G.shape[1] else X).to(G.dtype)
```

::: napkinmath
[What Muon's orthogonalisation costs]{.box-title}

For a $d\times d$ weight, one iteration costs $X X^\top$ ($2d^3$), $A^2$ ($2d^3$) and the product
with $X$ ($2d^3$), so $6d^3$ FLOPs, or $30d^3$ for five iterations. Per optimizer step, over all
hidden matrices of a model with $N$ parameters arranged as square $d\times d$ blocks, that is
$30 d\cdot N$ FLOPs. For $d=2048$ that is $6\times10^4 N$, the same as the forward and backward pass
over $10^4$ tokens. With a batch of $5\times10^5$ tokens it adds about 2% to the step, and it is
compute-bound tensor-core work, unlike AdamW's memory-bound update. The real systems cost is
elsewhere: the update needs the *whole* matrix, so under FSDP or tensor parallelism
([@ch:sharding]) each matrix has to be gathered or the orthogonalisation distributed.
:::

[@fig:optim:compare] compares the four optimizers on a small task we can train on a CPU.
Take it as a demonstration that the implementations work, not as a ranking: each optimizer received
one hand-picked learning rate, and the model has two layers.

![Four optimizers from `llmsys.optim` on the task of [@fig:optim:lrsweep], each with one
learning rate picked by hand (measured on CPU, one seed). All four reach within 0.06 nats of each
other, and plain SGD with Nesterov momentum comes out ahead. On a two-layer model with a stationary
data distribution, the gradient-scale problems that motivate adaptive methods barely arise. Do not
read rankings from toy runs; the lab asks you to tune each optimizer before
comparing.](figures/ch13_optimizers-9.svg){#fig:optim:compare}

## Feeding the loop: batches from a memory-mapped token array {#sec:optim:data}

Pretraining data is tokenised once, offline, into one long array of token ids with an end-of-text
separator between documents, and stored as raw little-endian integers
([@sec:text:pipeline]). With a vocabulary under 65,536, `uint16` halves the file size
relative to `int32`; a 10B-token corpus is 20 GB. The training loop then samples random windows from
it ([@fig:optim:window]):

``` {#lst:optim:batcher .python caption="Random windows from a memory-mapped token file (`RandomWindowBatcher.__next__`)."}
tokens = np.memmap("train.bin", dtype=np.uint16, mode="r")   # nothing read yet
starts = rng.integers(0, len(tokens) - T, size=B)            # B random offsets
idx = starts[:, None] + np.arange(T + 1)[None, :]            # (B, T+1) indices
chunk = torch.from_numpy(tokens[idx].astype(np.int64))       # reads ~B*(T+1)*2 bytes
chunk = chunk.pin_memory().to("cuda", non_blocking=True)     # async H2D copy
x, y = chunk[:, :-1], chunk[:, 1:]                           # targets = inputs shifted by 1
```

![Sampling windows from a flat token file. Each batch row is $T+1$ contiguous tokens starting at a
uniformly random offset; inputs and targets are the two shifted views. The memmap reads only the
touched pages.](figures/ch13_optimizers-10.svg){#fig:optim:window}

Why random windows rather than the epoch-based sliding-window dataset of [@ch:text]? At
pretraining scale you rarely see a token more than once, so "epochs" lose their meaning. Sampling
with replacement is simple, needs no shuffle buffer, and makes every offset equally likely. It also
makes the sampler's state tiny: a random generator. Three engineering details matter:

- **Validate the memmap.** A memmap with the wrong `dtype` or a header you forgot about silently
  yields garbage. Assert that `tokens.max() < vocab_size` on a sample and decode a few windows back
  to text before training.

- **Make the sampler resumable.** Use a private generator (`np.random.default_rng(seed)`) and save
  its `bit_generator.state` in the checkpoint. A global `np.random` call elsewhere in your code
  would otherwise change which batches a resumed run sees.

- **Overlap host work with the GPU.** Fancy indexing a memmap, converting to `int64` and copying to
  the device takes milliseconds of host time per batch. Pinned memory plus `non_blocking=True` makes
  the copy asynchronous, and a background thread (`PrefetchBatcher`) samples the next batch while
  the GPU computes the current one.

::: napkinmath
[Does the sampler keep up?]{.box-title}

A batch of $B=64$ windows of $T=1024$ `uint16` tokens is 131 KB read from the page cache (or a fast
NVMe drive). The model-side work for that batch on a 125M-parameter model is
$6\times1.25\times10^8\times65{,}536 \approx 4.9\times10^{13}$ FLOPs, about 0.1 s on an H100 at 50%
MFU. The data side is three orders of magnitude cheaper, so a single prefetch thread is plenty. The
picture changes only for tiny models or when you tokenise on the fly; tokenise offline.
:::

## Checkpointing everything {#sec:optim:ckpt}

A checkpoint is correct when *loading it and continuing produces the same run as never stopping*.
That requires more than the weights ([@fig:optim:ckpt]):

![Anatomy of a resumable checkpoint and the write protocol. Writing to a temporary name and renaming
means a crash during step 1 leaves the previous checkpoint intact; readers never see a half-written
file.](figures/ch13_optimizers-11.svg){#fig:optim:ckpt}

::: description
AdamW's $m$, $v$ and step count. Restarting with fresh moments re-runs the warmup transient: Adam's
first steps are sign-like ([@sec:optim:adam]) and too large for a trained model, and the
bias-correction counter also restarts.

The learning rate is a function of the global step.

For our sampler, its generator state; for a sequential loader, the byte offset and shuffle seed.
Without it the resumed run revisits some data and skips other data.

Dropout masks and any sampling inside the loop draw from torch's generators. For bit-exactness,
restore python, numpy, torch-CPU and every CUDA device's generator.
:::

``` {#lst:optim:ckptcode .python caption="Saving and restoring with `CheckpointManager`."}
ckpt = CheckpointManager("runs/ts-small/ckpt", keep_last=3)
if ckpt.latest() is not None:                       # resume if possible
    start = ckpt.restore(model, opt, data)["step"]
...
if (step + 1) % ckpt_every == 0:                    # step+1 = number of completed steps
    ckpt.save(step + 1, model, opt, data, extra=dict(config=asdict(cfg)))

# inside save(): write-then-rename, so a crash never corrupts the last good file
torch.save(obj, final + ".tmp")
os.replace(final + ".tmp", final)
```

The lab's acceptance test is the strongest specification of "correct": train 16 steps straight
through; separately, train with a simulated crash at step 11, rebuild the model from scratch in a
"new process", resume from the step-8 checkpoint, and require the resumed losses and final weights
to be *bit-identical* to the uninterrupted run. That test catches every item in the list above, and
also loaders that are not seeded.

::: systemslens
[Checkpoint I/O at scale]{.box-title}

A checkpoint of a model with $N$ parameters trained with AdamW is about $16N$ bytes: 1.1 TB for a
70B model. Writing it from one process at 2 GB/s takes nine minutes, during which the GPUs idle.
Large systems therefore (i) shard the write so every rank saves its own slice in parallel (PyTorch
Distributed Checkpoint), (ii) copy state to host memory and write asynchronously while training
continues, and (iii) save more often to fast local disk and less often to remote object storage. The
checkpoint interval trades lost work against I/O overhead: with mean time between failures $M$ and
checkpoint cost $C$, Young's approximation gives an optimal interval of about $\sqrt{2CM}$.
[@sec:pretrain:efficiency] returns to async checkpointing in the GPT-2 loop.
:::

## Experiment management {#sec:optim:explog}

You will run dozens of variants of every experiment, so the infrastructure for launching and
comparing them pays for itself within a day. `llmsys/explog.py` implements the minimum we have found
useful:

- **Typed configs with command-line overrides.** A `@dataclass` holds every hyperparameter;
  `config_from_cli(TrainConfig, sys.argv[1:])` applies `--key=value` overrides (and an optional
  `--config=file.json`) and *rejects unknown keys*, so a typo in a sweep script fails immediately
  instead of silently running the default.

- **One directory per run, named by content.** `RunLogger.create(root, name, cfg)` appends a short
  hash of the config to the name and writes `config.json` and `meta.json` (command line, git
  revision, PyTorch version, GPU). Two runs with the same name but different settings cannot
  overwrite each other.

- **Append-only JSON-lines metrics, with step *and* wall-clock time on every record.** JSONL
  survives crashes (at worst the last line is truncated, and the reader skips it), can be tailed
  live, and replotted against steps, tokens or seconds later. Plotting against wall-clock time is
  what tells you whether a change that improves loss per step is worth its cost per step.

- **Optional mirroring to Weights & Biases.** If you pass `wandb_project`, the same records are sent
  to W&B; the local files remain the source of truth.

- **Sweeps as data.** `grid(lr=log_spaced(1e-4, 1e-2, 5), batch=[32, 64])` produces points, and
  `sweep_commands(...)` turns them into shell lines you can pipe into a Slurm array job or a cloud
  launcher.

``` {#lst:optim:explog .python caption="A training script's experiment plumbing."}
@dataclass
class TrainConfig:
    d_model: int = 512; n_layers: int = 4; lr: float = 3e-3; batch: int = 64
    steps: int = 5000; warmup: int = 200; run_name: str = "base"

cfg = config_from_cli(TrainConfig, sys.argv[1:])          # --lr=1e-3 --batch=128
run = RunLogger.create("runs", cfg.run_name, cfg, wandb_project=None)
...
run.log(step, split="train", loss=loss.item(), lr=lr, grad_norm=gn,
        tokens=(step + 1) * cfg.batch * T, tok_per_s=tps)
if step % eval_every == 0:
    run.log(step, split="val", loss=val_loss)
run.summary(final_val=val_loss, diverged=False); run.close()
# later:  plot_runs(glob("runs/lr*"), y="loss", x="wall", where={"split": "val"})
```

##### What to log.

Every step: training loss, learning rate, pre-clip gradient norm, tokens seen, throughput (tokens/s)
and, on GPU, `torch.cuda.max_memory_allocated()`. Every few hundred steps: validation loss on a
*fixed* set of batches (the same batches each time, so curves are comparable), a few generated
samples, and per-layer weight and update norms. Keep a human-readable *experiment log* next to the
runs: what you changed, why, what you expected, and what happened. Six weeks later the plots alone
will not tell you why a run exists.

## Debugging the loop and tuning the learning rate {#sec:optim:debug}

When a model does not learn, work from cheapest to most expensive check:

1.  **Initial loss $\approx \ln V$** ([@sec:optim:loss]).

2.  **Overfit one batch.** Train on a single fixed batch for a few hundred steps. The loss should
    approach zero. If it does not, the bug is in the model, loss or optimizer, not the data or
    hyperparameters. Typical culprits: a causal mask that leaks the future (loss goes to zero *too*
    fast on the real data too), targets not shifted, parameters not registered (a plain Python list
    instead of `nn.ModuleList`), `optimizer.zero_grad` never called.

3.  **Check gradients against finite differences** in float64 on a tiny model whenever you write a
    custom autograd function; PyTorch's `gradcheck` utility automates this.

4.  **Watch norms.** Log per-layer activation RMS, weight norms and the update-to-weight ratio
    $\|\Delta\theta\|/\|\theta\|$ (`llmsys.optim.update_ratio`). For AdamW at a healthy learning
    rate it is around $10^{-3}$ per step for most matrices. Ratios near $10^{-1}$ mean the learning
    rate is far too high; ratios near $10^{-5}$ mean those weights barely move.

5.  **Sweep the learning rate on a log grid**, and judge the result by validation loss at a fixed
    token budget, with the schedule's $T_c$ set to that budget.

[@fig:optim:lrsweep] shows such a sweep, measured on CPU in this environment with Lab 12's
`run.py sweep-lr` on a synthetic first-order Markov language whose entropy rate (the lowest
achievable loss, 1.443 nats/token) is known exactly.

![Learning-rate sweep for a 2-layer, $d=64$ Qwen3-shaped model trained for 400 steps ($16\times64$
tokens per step, warmup + cosine to $\eta/10$, AdamW, clip 1.0) on a synthetic Markov language with
a known entropy rate of 1.443 nats/token. Measured on CPU in this environment (single seed, so
differences under $\sim$`<!-- -->`{=html}0.01 are noise). The optimum is broad, spanning $10^{-2}$
to $3\times10^{-2}$. At $\eta=1$ the training loss spiked above its initial value (7.3 vs 5.8) and
recovered only because clipping and QK-norm kept the run from diverging. The best learning rate sits
within roughly one decade of the largest one that still trains, as the edge-of-stability folklore
predicts.](figures/ch13_optimizers-12.svg){#fig:optim:lrsweep}

##### Learning rate and batch size.

The two interact. Below a *critical batch size* (McCandlish et al., 2018), doubling the batch halves
the gradient noise and lets you raise the learning rate (roughly linearly for SGD, roughly as
$\sqrt{B}$ for Adam) to reach the same loss in half the steps. Above it, extra batch size buys
almost nothing per token. The critical batch size grows as the loss falls, which is one reason large
runs ramp the batch size up during training. On one GPU, the practical recipe is to use the largest
batch that fits and still sits below the critical size, re-tune the learning rate whenever you
change it, and compare runs at equal *tokens*, not equal steps. [@ch:scalinglaws] gives the
scaling rules, including how $\mu$P (Yang et al., 2022) makes the optimal learning rate transfer
across model widths.

## Summary {#sec:optim:summary}

- A training loop is a small system: sampler, numerically careful loss, optimizer with state,
  schedule, clipping, checkpointing and logging. It must be resumable and observable.

- Compute cross-entropy with logsumexp in fp32 and check that the initial loss is $\ln V$.

- Adam is RMSProp plus momentum plus bias correction. Its per-coordinate normalisation bounds the
  step by about $\eta$. AdamW decouples weight decay from that normalisation; for LLMs use
  $\beta_2\approx0.95$.

- The optimizer step is memory-bound: about 28 bytes per parameter per step for AdamW, a few
  milliseconds per billion parameters on an H100, as long as it runs as one fused or multi-tensor
  pass rather than hundreds of small kernels.

- AdamW model state is 16 bytes per parameter. Activations are linear in batch size, and attention
  scores dominate them without FlashAttention. Derive $\text{mem}(B)=aB+b$ before every run.

- Warmup protects Adam's early steps; cosine bakes the run length into the schedule; WSD does not,
  which allows branching decays.

- Clip the *global* gradient norm, log the pre-clip norm, and remember that clipping is a global
  reduction under sharding.

- Lion, Adafactor, Shampoo/SOAP and Muon trade memory or extra compute for better conditioning. Muon
  orthogonalises momentum with a few tensor-core matmuls.

- Sample random windows from a validated `uint16` memmap; checkpoint weights, optimizer state, step,
  sampler state and RNG atomically; test resumption bit for bit.

- Log step and wall-clock time on every record, keep configs typed and runs uniquely named, and
  debug from the cheapest check (initial loss) to the most expensive (LR sweeps).

## Exercises {#sec:optim:exercises}

::: exercise
[Bias correction]{.box-title}

Show that without bias correction, Adam's first update with $\beta_1=0.9,\beta_2=0.999$ has
magnitude $\eta(1-\beta_1)/\sqrt{1-\beta_2}\approx3.16\,\eta$ per coordinate. What happens with
$\beta_2=0.95$? Why is the second-moment correction the more important one?
:::

::: exercise
[Coupled versus decoupled decay]{.box-title}

Consider a single scalar parameter with zero loss gradient, trained with Adam + L2 ($\lambda$) and
with AdamW ($\lambda$), both at learning rate $\eta$. Write the update for each. Show that Adam + L2
shrinks the weight by $\approx\eta\,\mathrm{sign}(\theta)$ per step (independent of $\lambda$, once
$v$ is dominated by the decay term) while AdamW shrinks it geometrically.
:::

::: exercise
[Optimizer bytes]{.box-title}

Count bytes read and written per parameter for one step of (a) SGD with momentum in fp32, (b) Lion
in mixed precision with fp32 master weights, (c) AdamW with bf16 $m$ and $v$ and fp32 master
weights. Convert each into milliseconds per billion parameters on an H100 (3.35 TB/s).
:::

::: exercise
[Launch-bound updates]{.box-title}

A 28-layer Qwen3-0.6B has about 310 parameter tensors. If each elementwise kernel launch costs
5 $\mu$s of CPU time and the loop implementation issues 10 kernels per tensor, how long does the
optimizer step take on the CPU side? Compare with the memory-bound time for 0.6B parameters. Measure
the loop and `foreach` paths of `llmsys.optim.AdamW` on a GPU if you have one.
:::

::: exercise
[Accounting method]{.box-title}

Using the method of [@sec:optim:account], derive $\text{mem}(B)=aB+b$ for Qwen3-0.6B
([@ch:modern]) trained in mixed precision with AdamW at $T=4096$, (i) with naive attention
and (ii) with FlashAttention. What micro-batch size fits on a 24 GB GPU in each case?
:::

::: exercise
[Schedules and resumption]{.box-title}

A colleague resumes a cosine run from a checkpoint, but their script reconstructs the schedule with
`total` equal to the *remaining* steps and starts counting at zero. Sketch the learning rate they
actually used. What would the loss curve look like?
:::

::: exercise
[Clipping semantics under sharding]{.box-title}

Two tensor-parallel ranks each hold half of every gradient. Rank 0 has local norm 3 and rank 1 local
norm 4; $M=1$. What scale should each rank apply? What happens if each rank clips its own shard to
norm 1 independently? Write the extra collective needed.
:::

::: exercise
[Muon geometry]{.box-title}

Show that $p(X)=aX+b(XX^\top)X+c(XX^\top)^2X$ maps $U\Sigma V^\top$ to $Up(\Sigma)V^\top$. Why must
the input be scaled so that $\sigma_{\max}\le1$? Why does Muon apply only to 2-D hidden matrices and
not to embeddings?
:::

::: exercise
[Resumable data]{.box-title}

Our sampler draws with replacement. Design a sampler that visits every window of a token file
exactly once per epoch in random order without materialising the permutation (hint: a random affine
bijection $i\mapsto (ai+c)\bmod n$ with $\gcd(a,n)=1$). What must its `state_dict` contain?
:::

::: exercise
[Edge of stability]{.box-title}

Reproduce [@fig:optim:lrsweep] with `run.py sweep-lr`, extend the grid until a run diverges
without clipping (pass `--clip=1e9`), and plot the final loss against the learning rate. How far
below the divergence point is the best learning rate? Repeat after removing QK-norm from the model.
What changes?
:::

## Lab 13: Optimizers and a resumable training loop {#sec:optim:lab}

::: labbox
[Optimizers and a resumable training loop]{.box-title}

**Code:** `code/labs/lab13_optim`

**Goals**

Implement the update rules and the plumbing of this chapter, verify them against PyTorch, and prove
that your loop can be killed and resumed without changing a single bit of the result.

**Setup**

``` bash
cd code
LAB_IMPL=starter  python -m pytest labs/lab13_optim -q     # your implementation
LAB_IMPL=solution python -m pytest labs/lab13_optim -q     # reference
python labs/lab13_optim/run.py --help                       # experiments
```

**Tasks**

1.  `SGD.step`: momentum and Nesterov, matching `torch.optim.SGD` (4 tests).

2.  `AdamW._step_loop`: decoupled and coupled decay, bias correction, $\epsilon$ placement; matches
    `torch.optim.AdamW` and `torch.optim.Adam`.

3.  `AdamW._step_foreach`: the same numbers with `torch._foreach_*` kernels.

4.  `WarmupCosine` and `WSD` schedules (0-based steps, floor after the end).

5.  `clip_grad_norm_`: global norm in fp32, in-place scaling, pre-clip norm returned; matches
    `torch.nn.utils.clip_grad_norm_`.

6.  `RandomWindowBatcher`: vectorised window gather from a memmap, shifted targets,
    `state_dict`/`load_state_dict` for exact replay.

7.  `newton_schulz_orthogonalize`: singular values pulled into $[0.5,1.3]$.

8.  `optimizer_state_bytes` and `training_state_bytes`.

9.  `RunLogger.log` and `load_jsonl` (tolerating a truncated last line).

10. `CheckpointManager.save/restore`: atomic write, RNG and data state, keep-last-$k$.

11. `train_loop`: wire everything together; the bit-exact resume test must pass.

**Measurements**

- `run.py make-markov` then `run.py sweep-lr`: reproduce [@fig:optim:lrsweep]; extend the
  grid to find the divergence point with and without clipping.

- `run.py compare`: tune each optimizer's learning rate with a three-point sweep before comparing.
  Does the ranking survive tuning?

- `run.py wsd`: compare cosine runs and WSD branches at three budgets ([@fig:optim:wsd]).
  Count total optimizer steps for both approaches.

- On a GPU: time `AdamW(foreach=False)`, `AdamW(foreach=True)` and `torch.optim.AdamW(fused=True)`
  for Qwen3-0.6B's parameters; compare with the 28-bytes-per-parameter bound.

**Deliverables**

Passing tests; the three plots above with one paragraph each; a table of measured optimizer-step
times against the bandwidth bound.

**Stretch goals**

Implement 8-bit block-wise quantised Adam states and measure the loss difference after 1000 steps.
Add a `PrefetchBatcher` to `train_loop` and measure the step-time change on a GPU. Implement
Adafactor's factored second moment and compare its memory.
:::
