# RL Training Systems {#ch:rlsys}

Chapters [-@ch:grpo] and [-@ch:grpo2] implemented reinforcement learning with verifiable
rewards (RLVR) as a single Python loop: sample a group of answers, grade them, compute advantages,
take a gradient step, repeat. On one GPU with a 0.6B-parameter model that loop is easy to read. It
is also hopeless at scale. The models that made RL-trained reasoning famous were trained with
thousands of accelerators for weeks, and almost none of that engineering effort went into the loss
function. It went into the *system* that wraps the loss: an inference engine that produces millions
of tokens per step, a training engine that consumes them, a fleet of reward workers that execute
untrusted code, and the plumbing that moves weights, tokens and log-probabilities between them
without leaving expensive hardware idle.

This chapter is about that plumbing. It is written as a graduate systems lecture: we derive cost
models, build a simulator, and study the design space spanned by today's open frameworks (OpenRLHF,
veRL/HybridFlow, TRL, NeMo-Aligner and several newer asynchronous systems). The ecosystem moves
quickly; treat the framework descriptions as a snapshot of designs that were public as of 2025, and
the cost models as the part that will age well.

![Roadmap of this chapter. We start from the dataflow of a single RL step, then add the systems
concerns one at a time, ending with the reward infrastructure. Every stage has a counterpart in the
lab.](figures/ch46_rl_systems-1.svg){#fig:rlsys:roadmap}

## The anatomy of an RL step {#sec:rlsys:anatomy}

Strip GRPO ([@ch:grpo]) down to its dataflow and five kinds of work remain. They are
summarised in [@fig:rlsys:dataflow].

1.  **Rollout (generation).** For a batch of $B$ prompts, sample $G$ responses each from the current
    policy $\pi_{\theta}$. This is pure inference: a prefill per prompt followed by thousands of
    memory-bound decode steps ([@ch:inference]).

2.  **Reward.** Score each of the $BG$ responses. For RLVR this means extracting an answer and
    checking it with a math verifier ([@ch:eval]) or executing generated code against unit
    tests; for RLHF it means a forward pass of a learned reward model.

3.  **Scoring (log-prob) passes.** Compute per-token log-probabilities of every response under the
    policy that generated it ($\pi_{\theta_{\text{old}}}$) and, if a KL penalty is used, under a
    frozen reference policy $\pi_{\text{ref}}$. These are prefill-only forward passes.

4.  **Advantage computation.** Cheap arithmetic: group means and standard deviations (GRPO), or
    generalised advantage estimation with a critic (PPO).

5.  **Training.** One or more optimizer steps on the clipped surrogate objective, each a forward and
    backward pass over all response tokens.

After the update, the inference engine must receive the new weights before the next rollout can be
on-policy. That arrow---*weight synchronisation*---is what makes RL different from both pretraining
(no inference engine) and serving (no weight updates).

![Dataflow of one RLVR step. Orange = inference workloads (memory-bandwidth bound), blue =
training-style workloads (compute bound), green = reward computation (often CPU-bound, sometimes
another model). The purple edge is the weight synchronisation that closes the
loop.](figures/ch46_rl_systems-2.svg){#fig:rlsys:dataflow}

### Where does the time go?

The single most important fact about RL systems for LLMs is that the stages have *different
bottlenecks*. Training and scoring passes process all tokens of a sequence in parallel and are
compute-bound; they want large matrix multiplies and achieve a respectable fraction of peak FLOP/s
([@ch:pretrain]). Generation emits one token per sequence per step, rereads all weights and
the KV cache at every step, and is memory-bandwidth-bound ([@ch:inference]). A configuration
that is ideal for one stage is usually wasteful for the other: training wants the model sharded with
FSDP or tensor+pipeline parallelism across many GPUs and a large activation budget; generation wants
as few shards as possible (to avoid communication on every decode step) and as much free memory as
possible for the KV cache.

::: napkinmath
[One RL step for an 8B model on 64 H100s]{.box-title}

Take Qwen3-8B ($N\approx 8.19\times10^9$ parameters, 36 layers, 8 KV heads of dimension 128),
$B=512$ prompts, $G=8$ samples each, prompts of 200 tokens and responses of 4,000 tokens on average.
That is $BG = 4096$ sequences and $\approx 17.2$M tokens per step.

- *Training:* $6N\cdot 17.2\text{M} \approx 8.5\times10^{17}$ FLOPs. At 40% MFU on 64 H100s
  ($64 \times 989$ TFLOP/s $\times 0.4$) this is about **33 s**.

- *Scoring:* two forward passes ($\pi_{\theta_{\text{old}}}$ and $\pi_{\text{ref}}$),
  $2\times 2N\cdot 17.2$M $\approx 5.6\times10^{17}$ FLOPs, about **22 s**.

- *Generation:* the KV cache costs $2\cdot 36\cdot 8\cdot 128\cdot 2 \approx 147$ KB per token. With
  16.4 GB of bf16 weights and $\sim$`<!-- -->`{=html}8 GB of runtime overhead, about 55 GB of an
  80 GB card is left for KV; at an average live context of 2,200 tokens that is $\approx 170$
  concurrent sequences. Each decode step streams $\approx 71$ GB at 3.35 TB/s,
  $\approx$`<!-- -->`{=html}21 ms, so one GPU emits $\approx 8{,}000$ tokens/s and 64 GPUs need
  $\approx$**32 s** for 16.4M generated tokens---*if every slot stayed busy*.

- *The tail:* if the longest response in the batch is 16,000 tokens, the last sequence decodes alone
  at $\approx 5$--6 ms per token (weights dominate), i.e. another **80--85 s** during which almost
  the whole generation fleet is idle.

The idealised generation time already matches training, and the tail more than doubles it. This is
why practitioners report that rollout generation typically dominates RL wall-clock time, and why
most of this chapter is about keeping the generator busy.
:::

::: keyidea
[RL is a pipeline of heterogeneous stages]{.box-title}

An RL step chains an inference workload (bandwidth-bound, latency-sensitive, variable length) with a
training workload (compute-bound, fixed shapes) through a weight-update dependency. Every design in
this chapter (colocation, asynchrony, partial rollouts, packing) is an answer to the question "how
do we keep both kinds of hardware busy despite that dependency?"
:::

## Models, memory, and placement {#sec:rlsys:placement}

### Which models must be resident?

Classic RLHF with PPO juggles four networks: the *actor* (the policy being trained), the *critic* (a
value model, usually initialised from the reward model), a frozen *reward model*, and a frozen
*reference model* used for the KL penalty. GRPO removes the critic by estimating the baseline from
the group of samples; RLVR replaces the reward model with a program; and recent recipes (DAPO,
Dr. GRPO; [@ch:grpo2]) often drop the reference model by setting the KL coefficient to zero.
A modern RLVR run may therefore hold only the actor---but the actor exists in two *forms*: as
training state and as an inference engine.

::: {#tab:rlsys:memory .table-float}
| Component | Bytes/param | 8B model | Notes |
|:---|---:|---:|:---|
| Actor training state (bf16 w+g, fp32 master, Adam $m,v$) | 16 | 131 GB | sharded by FSDP/ZeRO-3 |
| Actor activations (per micro-batch) | -- | 10s of GB | checkpointing trades for compute |
| Reference model (bf16, inference only) | 2 | 16.4 GB | can be offloaded or dropped |
| Critic (PPO only) | 16 | 131 GB | absent in GRPO |
| Rollout engine weights (bf16) | 2 | 16.4 GB | per engine replica |
| Rollout KV cache | -- | rest of HBM | sets decode concurrency |

: Memory accounting for the models of an RL run (8B example). Training state is sharded across the
trainer's GPUs; inference weights are replicated once per rollout engine (or sharded by its
tensor-parallel degree).
:::

### Colocated versus disaggregated placement

There are two basic ways to place the rollout engine and the trainer on a cluster of $G$ GPUs
([@fig:rlsys:placement]).

##### Colocated (time-shared).

Every GPU hosts both a training shard and an inference engine, and the two *take turns*. During
generation the optimizer state and gradients are offloaded to host memory (or freed), and the
engine's KV cache gets most of HBM; for training, the engine releases its KV cache (vLLM's "sleep
mode" and SGLang's memory release APIs exist for exactly this), and the trainer reclaims memory.
veRL's *3D-HybridEngine* (Sheng et al., HybridFlow, 2025) and OpenRLHF's hybrid-engine mode follow
this pattern. Its virtue is that *no GPU is ever dedicated to a stage that is not running*. Its
costs are the switch overhead (offload/onload, resharding) and the fact that the two stages are
strictly serialised.

##### Disaggregated (split pools).

$G_{\text{gen}}$ GPUs run inference engines and $G_{\text{train}} = G - G_{\text{gen}}$ run the
trainer. OpenRLHF's original Ray-based design placed actor, critic, reward and reference models (and
vLLM engines) on separate groups of GPUs; most asynchronous systems use split pools. Each pool can
use the parallelism layout that suits it, and the pools can run *concurrently*---but only if the
algorithm tolerates data generated by slightly older weights ([@sec:rlsys:async]). In a
strictly synchronous schedule each pool idles while the other works.

![Two placements. (a) Colocation multiplexes both workloads on the same devices, in time. (b)
Disaggregation dedicates devices to each workload, which permits different parallel layouts and
concurrent execution, at the price of cross-pool weight transfers and potential
idleness.](figures/ch46_rl_systems-3.svg){#fig:rlsys:placement}

### A cost model for the two placements

Let $W_g$ be the generation work of one step and $W_t$ the training+scoring work, both measured in
GPU-seconds at perfect scaling. Ignore the tail for now (we return to it in
[@sec:rlsys:tail]). With $G$ GPUs: $$\begin{align}
T_{\text{coloc}} &= \frac{W_g}{G} + \frac{W_t}{G} + T_{\text{switch}},\label{eq:rlsys:coloc}\\
T_{\text{sync-split}} &= \frac{W_g}{G_{\text{gen}}} + \frac{W_t}{G - G_{\text{gen}}} + T_{\text{sync}},\label{eq:rlsys:sync}\\
T_{\text{async-split}} &= \max\!\left(\frac{W_g}{G_{\text{gen}}} + T_{\text{sync}},\ \frac{W_t}{G - G_{\text{gen}}}\right).\label{eq:rlsys:async}
\end{align}$$ Minimising [@eq:rlsys:sync] over $G_{\text{gen}}$ gives
$G_{\text{gen}}^\star/G = \sqrt{W_g}/(\sqrt{W_g}+\sqrt{W_t})$ and
$T^\star = (\sqrt{W_g}+\sqrt{W_t})^2/G + T_{\text{sync}} \ge (W_g+W_t)/G$: a synchronous split is
*never* faster than ideal colocation, because each pool idles while the other works.
[@eq:rlsys:async] is minimised by balancing the two terms,
$G_{\text{gen}}^\star/G = W_g/(W_g+W_t)$, which yields $T^\star \approx (W_g+W_t)/G$---the same as
colocation without $T_{\text{switch}}$, but with one step of policy lag. So the trade-off is:

::: systemslens
[What disaggregation really buys]{.box-title}

In the idealised model, splitting pools only helps if (i) the algorithm can run the pools
concurrently (asynchrony), or (ii) the stages scale *sub-linearly* on shared hardware so that
specialised layouts recover efficiency (e.g. generation with TP$=1$ engines has no per-token
communication, whereas training needs FSDP across many GPUs), or (iii) the switch cost
$T_{\text{switch}}$ is large (offloading 131 GB of optimizer state over PCIe links that deliver a
few tens of GB/s per GPU, and share host memory bandwidth across 8 GPUs per node, takes seconds per
step), or (iv) the hardware is heterogeneous---older or inference-optimised accelerators can
generate while training GPUs train. In practice, all four effects matter, which is why both designs
survive.
:::

## Weight synchronisation and resharding {#sec:rlsys:sync}

After every optimizer step the inference engines must receive $\theta_{k+1}$. Two problems hide in
that sentence: *moving* the bytes and *reshaping* them.

### Moving the bytes

::: napkinmath
[How long does a weight update take?]{.box-title}

An 8B model in bf16 is 16.4 GB.

- Colocated, same GPU: no transfer at all if the engine can be pointed at the trainer's tensors
  (e.g. via CUDA IPC handles); otherwise a device-to-device copy at HBM speed, a few ms.

- Over NVLink within a node ($\sim$`<!-- -->`{=html}450 GB/s per direction on H100):
  $\approx 36$ ms.

- Over one 400 Gb/s InfiniBand NIC (50 GB/s): $\approx 0.33$ s. With 8 NICs per node and each
  trainer rank sending its own shard in parallel, $\approx 0.04$ s.

- Over 100 Gb/s Ethernet (12.5 GB/s): $\approx 1.3$ s; for a 32B model, $\approx 5$ s.

A naive point-to-point fan-out to $R$ engines multiplies the sender's time by $R$; a pipelined
(chunked) broadcast tree costs roughly one payload transfer regardless of $R$
([@ch:distributed]). Compared with the tens of seconds of a step these numbers are small
inside a datacenter---but they are *on the critical path* of a synchronous loop, and they grow
quickly for geo-distributed or heterogeneous-cluster RL.
:::

Two common optimisations reduce the payload: send only the *LoRA adapters* when the policy is
trained with LoRA ([@ch:instruct]), so the update is a few hundred MB; and send *bucketed*
chunks so that transfer overlaps with the engine loading the previous chunk.

### Reshaping the bytes

The trainer and the engine rarely agree on layout. A trainer using FSDP holds each parameter
flattened and split across $D$ data-parallel ranks; a Megatron trainer splits attention and MLP
weights by tensor-parallel degree $t_{\text{train}}$ and layers by pipeline stage. An inference
engine typically wants whole layers with tensor parallelism $t_{\text{gen}}$, often
$t_{\text{gen}} < t_{\text{train}}$, and fused tensors (e.g. a single `qkv_proj` instead of separate
$W_Q, W_K, W_V$, or gate and up projections concatenated). Weight sync is therefore a *resharding*
operation: an all-gather (or partial all-gather) along the training sharding, a re-split along the
inference sharding, plus name and layout translation ([@fig:rlsys:reshard]).

![Weight synchronisation is a resharding problem: parameters sharded for training must be gathered,
renamed, fused and re-split for the inference engine. veRL's 3D-HybridEngine organises the colocated
case so that each GPU only gathers the slice its inference shard needs, avoiding a full replica in
memory.](figures/ch46_rl_systems-4.svg){#fig:rlsys:reshard}

::: pitfall
[Silent layout bugs]{.box-title}

A transposed projection, a swapped `gate`/`up` pair, or a RoPE layout mismatch (interleaved
vs. rotate-half, [@ch:modern]) produces an engine that still emits fluent text---just from
the *wrong* model. Rewards then drift for reasons unrelated to learning. Always check, after each
sync during bring-up, that engine log-probs on a fixed prompt match the trainer's to within
numerical tolerance ([@sec:rlsys:mismatch]).
:::

## Synchronous versus asynchronous RL {#sec:rlsys:async}

### On-policy in theory, slightly off-policy in practice

The GRPO objective of [@ch:grpo] is an expectation over responses sampled from
$\pi_{\theta_{\text{old}}}$, the policy at the start of the step. A *synchronous* system enforces
this literally: rollouts for step $k+1$ start only after the weights of step $k$ are in the engine.
An *asynchronous* system lets the generator run ahead. If a sample used in the update to
$\theta_{k+1}$ was produced by $\theta_{k-s}$, we say it has *staleness* $s$.

Off-policy data is not a catastrophe; PPO itself performs several minibatch updates per batch, so
all but the first minibatch is already slightly off-policy, and the clipped ratio
$\pi_\theta/\pi_{\theta_{\text{old}}}$ exists precisely to bound how far one batch can push the
policy. What asynchrony changes is that the *behaviour policy* that produced a sample, call it
$\mu = \pi_{\theta_{k-s}}$, is now distinct from the proximal policy around which we clip. Several
systems therefore use a *decoupled* objective (AReaL, Fu et al., 2025, gives a clear derivation):
importance-weight by $\pi_{\text{prox}}/\mu$ to correct for staleness, and clip
$\pi_\theta/\pi_{\text{prox}}$ to bound the update, $$\begin{equation}
\mathcal{L}(\theta) = -\mathbb{E}_{y\sim\mu}\left[\frac{\pi_{\text{prox}}(y)}{\mu(y)}\,
\min\!\Big(r_\theta A,\ \operatorname{clip}(r_\theta,1-\epsilon,1+\epsilon)A\Big)\right],\qquad
r_\theta = \frac{\pi_\theta(y)}{\pi_{\text{prox}}(y)}.
\label{eq:rlsys:decoupled}
\end{equation}$$ With $s=0$, $\mu=\pi_{\text{prox}}=\pi_{\theta_{\text{old}}}$ and
[@eq:rlsys:decoupled] reduces to the standard clipped objective. The same structure
reappears in [@sec:rlsys:mismatch], where the behaviour policy differs from
$\pi_{\theta_{\text{old}}}$ not because of staleness but because of numerics.

### Three schedules

[@fig:rlsys:timelines] draws the three schedules that bracket the design space.

![Schedules for RL. (a) Synchronous: the generator idles during training (and vice versa in split
pools). (b) One-step asynchronous: batch $k{+}1$ is generated with $\theta_{k-1}$ or $\theta_k$
while batch $k$ trains; staleness $\le 1$. (c) Partial rollouts: each generation phase stops once
enough responses have finished; long responses continue in the next phase under newer weights.
Purple ticks are weight syncs.](figures/ch46_rl_systems-5.svg){#fig:rlsys:timelines}

Synchronous.

: Simple and exactly on-policy. Throughput is limited by $T_{\text{gen}}+T_{\text{train}}$, and the
  generator's long tail is fully exposed.

One-step (bounded-staleness) asynchronous.

: Generation of batch $k+1$ overlaps training on batch $k$. In steady state the step time is
  $\max(T_{\text{gen}},T_{\text{train}})$ and the staleness is bounded by a pipeline depth $s$ (the
  generator may not run more than $s$ versions ahead). This is the schedule of several open systems
  and of our simulator's `async` policy. Fully asynchronous variants (AReaL; PipelineRL, which even
  swaps weights *in flight* between decode steps of an ongoing sequence) push this further.

Partial rollouts.

: Keep more responses in flight than one batch needs; end each generation phase once $B$ responses
  have finished and resume the unfinished ones in the next phase. The tail no longer blocks a step,
  but some samples are produced by *several* policies, and the batches over-represent short
  responses until the long ones catch up. The Kimi k1.5 report (2025) describes using partial
  rollouts for long-context RL.

## The long-tail problem {#sec:rlsys:tail}

Reasoning-model response lengths are heavy-tailed: most answers to a batch of math problems end
after a few hundred to a few thousand tokens, but a handful run to the length cap. With continuous
batching ([@ch:serving]) a decode engine admits new work whenever a slot frees up, but a
*synchronous* RL step has no new work to admit---it needs *these* $BG$ responses before it can
train. The engine's occupancy therefore decays as short responses finish, and the last few sequences
decode almost alone.

We can quantify this with a toy model. Suppose response lengths are log-normal with mean $\bar L$
and shape $\sigma$, all $n$ responses start together on $n$ slots, and each slot decodes at a
constant $r$ tokens/s. The step's generation time is $\max_i L_i/r$ while the useful work is
$\sum_i L_i/r$, so the generator's utilisation is $$\begin{equation}
U = \frac{\sum_i L_i}{n\,\max_i L_i} \;\approx\; \frac{\bar L}{\mathbb{E}[\max_i L_i]}.
\end{equation}$$ For log-normal lengths $\mathbb{E}[\max_i L_i]$ grows roughly like
$e^{\mu + \sigma\sqrt{2\ln n}}$, so utilisation falls quickly with $\sigma$.
[@fig:rlsys:occupancy] shows the decay of occupancy computed by the simulator from the lab
(`generation_occupancy`).

![Occupancy of 128 decode slots during one synchronous generation phase with log-normal response
lengths (simulated with `llmsys.rlsys.sim.generation_occupancy`). The area above each curve is
wasted generator capacity: utilisation is 56%, 33% and 13% for $\sigma=0.25$, $0.5$ and $1$; with
$\sigma=1$ almost 87% of slot-time is idle because a few responses run far longer than the
rest.](figures/ch46_rl_systems-6.svg){#fig:rlsys:occupancy}

Mitigations, roughly in order of how much they change the algorithm:

1.  **Length caps and length-aware rewards.** A hard cap bounds $\max_i L_i$; DAPO (Yu et al., 2025)
    adds a soft overlong penalty so the policy learns to avoid truncation. Capping changes the task
    (truncated responses score zero) and should be monitored as the "clip ratio" of truncated
    samples.

2.  **Bigger batches.** More concurrent responses amortise the tail across more useful work, until
    KV memory runs out.

3.  **Over-provisioning with dropping.** Launch more prompts than needed and train on the first $B$
    groups to finish. This is *biased*: it systematically discards long, often hard, responses. Use
    with care.

4.  **Partial rollouts.** Resume, do not drop, the stragglers ([@fig:rlsys:timelines]c).

5.  **Asynchrony.** Let the next batch's generation fill the idle slots.

::: systemslens
[Dynamic sampling interacts with the tail]{.box-title}

DAPO's *dynamic sampling* ([@ch:grpo2]) discards groups whose $G$ rewards are all equal,
since their advantages are zero, and keeps sampling until the batch is full of informative groups.
From a systems standpoint this is an over-provisioning policy with a data-dependent acceptance rate:
early in training most groups fail and late in training most succeed, so the number of rollouts per
optimizer step varies severalfold over a run. An asynchronous generator that streams groups into a
buffer handles this gracefully; a synchronous one has to guess how many prompts to launch.
:::

### Simulating the schedules

The lab's simulator (`llmsys/rlsys/sim.py`) implements the four policies of
[@fig:rlsys:timelines] on top of an event-driven model of continuous batching: `n_slots`
decode slots at `tokens_per_sec` each, log-normal lengths, training time proportional to tokens, and
fixed costs for switching and weight sync. [@lst:rlsys:rungen] shows its core.

``` {#lst:rlsys:rungen .python caption="Continuous-batching generation with an early-stop for partial rollouts (from `llmsys/rlsys/sim.py`)."}
def run_generation(pending, n_slots, tps, stop_after, version):
    queue = list(pending)
    heap = []                  # (finish_time, seq, rollout, start_time)
    t, busy, seq = 0.0, 0.0, 0
    finished = []

    def admit(now):
        nonlocal seq
        while queue and len(heap) < n_slots:
            r = queue.pop(0)
            if not r.versions or r.versions[-1] != version:
                r.versions.append(version)          # this policy touched r
            heapq.heappush(heap, (now + r.remaining / tps, seq, r, now))
            seq += 1

    admit(0.0)
    while heap:
        t, _, r, start = heapq.heappop(heap)
        busy += t - start
        r.remaining = 0.0
        finished.append(r)
        if stop_after is not None and len(finished) >= stop_after:
            break                                   # partial rollouts: stop early
        admit(t)
    unfinished = []
    for (ft, _, r, start) in heap:                  # account partial progress
        busy += t - start
        r.remaining = max(0.0, (ft - t) * tps)
        unfinished.append(r)
    return t, finished, unfinished + queue, busy
```

[@fig:rlsys:simresults] reports the throughput of each policy as the tail gets heavier. Two
lessons stand out. First, one-step asynchrony helps a lot when the stages are balanced and the tail
is light ($\sigma=0.25$: $+64\%$), but almost nothing when generation is dominated by a few
stragglers---overlapping training with a mostly idle generator does not make the generator busier.
Second, partial rollouts attack the tail directly and keep throughput roughly constant as $\sigma$
grows, at the price of 45--85% of samples containing tokens from more than one policy version and of
a length-biased batch composition.

![Throughput of four RL schedules in the lab simulator (128 slots, batch 128, mean length 1,000
tokens, default costs; `python labs/lab46_rlsys/bench.py sim`). These are outputs of a model, not
measurements of a real cluster; their value is in the
trends.](figures/ch46_rl_systems-7.svg){#fig:rlsys:simresults}

::: pitfall
[Partial rollouts bias the batch toward short answers]{.box-title}

In each phase, the responses that finish first are the short ones. Early in a run, the gradient
therefore over-weights short responses, and long responses reach the trainer one or more versions
late. If the reward correlates with length (it usually does for reasoning), this is a systematic
bias. Log the length distribution of *trained* samples, not only of generated ones.
:::

## Packing the scoring and training passes {#sec:rlsys:packing}

Once rollouts exist, the trainer runs prefill-style passes over them: the
$\pi_{\theta_{\text{old}}}$ and $\pi_{\text{ref}}$ log-prob passes and the forward/backward of the
update. Their cost is linear in tokens (plus a quadratic attention term), so every padding token is
wasted work. With heavy-tailed lengths, padding to the longest sequence in a micro-batch can waste
most of the compute ([@fig:rlsys:packing]).

![Padding vs. packing for four responses of lengths 18, 5, 9 and 3. Packing concatenates sequences
into rows of a fixed token budget; a block-diagonal causal mask (or a varlen kernel driven by
`cu_seqlens`) prevents attention across sequence boundaries, and position ids restart at 0 for every
segment.](figures/ch46_rl_systems-8.svg){#fig:rlsys:packing}

Three ingredients make packing correct:

1.  **Bin packing.** Group sequences into rows under a token budget. First-fit decreasing (FFD) is
    simple and provably uses at most $\tfrac{11}{9}\mathrm{OPT}+\tfrac{6}{9}$ bins (the tight bound
    is due to Dósa, 2007).

2.  **Segment-aware attention.** Each token may attend only to earlier tokens of its own sequence.
    FlashAttention's `varlen` interface takes cumulative sequence lengths (`cu_seqlens`) and never
    materialises the mask; we build the mask explicitly.

3.  **Per-segment positions.** RoPE angles (or GPT-2 position embeddings) must restart at 0 for each
    sequence; otherwise the second sequence in a row sees shifted positions.

``` {#lst:rlsys:packing .python caption="Block-diagonal causal mask and a packed log-prob pass (`llmsys/rlsys/packing.py`, abridged)."}
def block_causal_mask(cu_seqlens):
    N = int(cu_seqlens[-1])
    seg = torch.zeros(N, dtype=torch.long)
    for k in range(len(cu_seqlens) - 1):
        seg[int(cu_seqlens[k]):int(cu_seqlens[k + 1])] = k
    same = seg[:, None] == seg[None, :]
    causal = torch.ones(N, N, dtype=torch.bool).tril()
    return ~(same & causal)                     # True = forbidden

def packed_completion_logprobs(model, seqs, prompt_lens, max_tokens):
    out = [0.0] * len(seqs)
    for bin_ in ffd_pack([len(s) for s in seqs], max_tokens):
        ids, pos, cu = build_packed_row([seqs[i] for i in bin_])
        logp = torch.log_softmax(varlen_forward(model, ids, pos, cu)[0].float(), -1)
        for j, i in enumerate(bin_):
            s, e = int(cu[j]), int(cu[j + 1])
            tok_lp = logp[s:e - 1].gather(-1, ids[0, s + 1:e, None]).squeeze(-1)
            out[i] = float(tok_lp[prompt_lens[i] - 1:].sum())
    return out
```

The function `varlen_forward` reuses the *weights* of the core `Qwen3Model` or `GPTModel` but runs
its own attention with the packed mask and per-token RoPE positions. The lab's test checks that
every packed log-prob equals the unpacked `sequence_logprob` of [@ch:refine] to within
$10^{-3}$. On a CPU with a tiny model and 64 log-normal-length sequences, the packed pass took 17 s
against 42 s for padded batches of 8 in our runs, with a real-token fraction of 96% vs. 39%.

::: pitfall
[Loss normalisation across micro-batches and ranks]{.box-title}

Token-level losses (DAPO, [@ch:grpo2]) are defined as a sum over *all* tokens in the global
batch divided by the global token count. If each micro-batch computes its own token-mean and
gradients are then averaged over micro-batches and data-parallel ranks, micro-batches with few
tokens are silently up-weighted, and the result depends on how the packer happened to group
sequences. The fix is to divide each micro-batch's token *sum* by the global token count (an
all-reduce of a scalar before the backward passes). Several frameworks have had bugs of exactly this
kind.
:::

::: systemslens
[Dynamic batch size by token budget]{.box-title}

Fixed-size micro-batches of variable-length sequences produce wildly varying memory peaks.
Production trainers instead form micro-batches by a *token budget* (veRL calls this dynamic batch
size; the idea is general), which makes activation memory roughly constant and lets the budget be
tuned to the largest value that fits. With packing, the token budget is literally the row length.
:::

## Train/inference mismatch {#sec:rlsys:mismatch}

A subtle problem appeared once RL systems began using high-throughput inference engines: even with
*identical* weights, the engine's token probabilities $\mu(y_t)$ differ from those the trainer
computes, $\pi_{\theta_{\text{old}}}(y_t)$. Engines use different kernels (fused attention, paged
KV, different GEMM tilings), different reduction orders, batch-size-dependent algorithms, sometimes
FP8 or quantised KV caches, and sampling-time logit processing. All of these are harmless for
serving and all of them change the low bits of the logits. Over thousands of tokens and a vocabulary
of 150k entries, low-bit differences become occasional large ratio differences on low-probability
tokens---exactly the tokens a sampler sometimes picks.

![Train/inference mismatch: the samples come from the engine's distribution $\mu$, while the
policy-gradient estimator assumes they come from
$\pi_{\theta_{\text{old}}}$.](figures/ch46_rl_systems-9.svg){#fig:rlsys:mismatch}

Because $y\sim\mu$, the unbiased policy gradient is $$\begin{equation}
\nabla_\theta J = \mathbb{E}_{y\sim\mu}\left[\sum_t \frac{\pi_{\theta_{\text{old}}}(y_t\mid y_{<t})}{\mu(y_t\mid y_{<t})}
\nabla_\theta \min\!\big(r_t A,\ \operatorname{clip}(r_t,1\pm\epsilon) A\big)\right],
\end{equation}$$ where we have used a per-token approximation of the sequence-level ratio, as PPO
does. Ignoring $\rho_t = \pi_{\theta_{\text{old}}}/\mu$ (i.e. setting it to 1) introduces a bias
that has been reported to destabilise long runs. Since $\rho_t$ is occasionally huge, the naive
correction has high variance. Two practical estimators, both implemented in
`llmsys/rlsys/mismatch.py`:

- **Truncated importance sampling (TIS):** $w_t = \min(\rho_t, C)$ with $C\approx 2$. This was
  popularised for LLM RL in a 2025 analysis by Yao et al. of the mismatch in vLLM+FSDP systems; it
  is the same truncation idea as in off-policy actor--critic methods.

- **Masked importance sampling (MIS):** $w_t = \mathbf{1}[1/C\le\rho_t\le C]$, which drops outlier
  tokens instead of down-weighting them; variants operate at the sequence level.

The weights are *detached*---they rescale gradients and are never optimised.

``` {#lst:rlsys:tis .python caption="Clipped surrogate with truncated importance-sampling correction."}
def is_weights(logp_train, logp_rollout, cap=2.0, mode="truncate"):
    ratio = torch.exp((logp_train - logp_rollout).detach())
    if mode == "truncate":
        return ratio.clamp(max=cap)
    if mode == "mask":
        return ((ratio >= 1.0 / cap) & (ratio <= cap)).to(ratio.dtype)
    return torch.ones_like(ratio)

def clipped_pg_loss(logp_new, logp_old, adv, mask, clip_eps=0.2,
                    logp_rollout=None, tis_cap=None, tis_mode="truncate"):
    adv = adv[:, None] if adv.dim() == 1 else adv
    ratio = torch.exp(logp_new - logp_old.detach())
    surr = torch.minimum(ratio * adv,
                         ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
    if logp_rollout is not None and tis_cap is not None:
        surr = surr * is_weights(logp_old, logp_rollout, tis_cap, tis_mode)
    m = mask.to(surr.dtype)
    return -(surr * m).sum() / m.sum().clamp(min=1)
```

Correction is one approach; *elimination* is the other. Making inference kernels
*batch-invariant*---so the result for a sequence does not depend on what else is in the
batch---removes a major source of nondeterminism, and with matching kernels in the trainer it is
possible to run bitwise on-policy RL (Thinking Machines Lab, "Defeating nondeterminism in LLM
inference", 2025). Others have argued for higher-precision formats (e.g. fp16 with its larger
mantissa instead of bf16) to shrink the mismatch. These remedies cost inference throughput; TIS
costs almost nothing. A reasonable default is: always log the mismatch, always recompute
$\pi_{\theta_{\text{old}}}$ in the trainer rather than trusting engine log-probs for the ratio, and
enable TIS or MIS.

::: napkinmath
[How big is the mismatch from precision alone?]{.box-title}

The lab evaluates the same tiny Qwen3 model in fp32 and in bf16 on random token sequences
(`token_logprobs` with `dtype=torch.bfloat16`). On our CPU runs the mean absolute log-prob
difference was $\approx 0.09$ nats, the per-token ratio ranged over $[0.70,\,1.37]$, and the $k_3$
KL estimate was $\approx 7\times10^{-3}$. A random tiny model has nearly uniform outputs and is not
representative of a trained model; run `bench.py mismatch --qwen3_dir` on real weights to see the
long tail on rare tokens that motivates TIS.
:::

::: keyidea
[Monitor these every step]{.box-title}

Mean $|\log\pi_{\theta_{\text{old}}}-\log\mu|$; the $k_3$ estimate of
$\operatorname{KL}(\mu\,\|\,\pi_{\theta_{\text{old}}})$; the fraction of tokens with
$\rho_t\notin[0.5,2]$; the maximum $\rho_t$. A sudden increase is often the first sign of a
weight-sync bug, a kernel change, or a numerically unstable model state.
:::

## Configuring the rollout engine for RL {#sec:rlsys:engine}

A serving engine tuned for chat traffic is not automatically a good rollout engine. Several settings
that are harmless (or desirable) for serving change the *distribution* that RL assumes it is
sampling from, or waste the structure of RL workloads.

##### Sampling parameters define $\mu$.

The policy-gradient estimator assumes samples come from the policy whose log-probabilities appear in
the ratio. If the engine samples with temperature $\tau\neq 1$, top-$p$ or top-$k$
([@ch:scaling]), it samples from a *transformed* distribution. Either sample at $\tau=1$
without truncation, or compute the training-side log-probabilities under the same
transformation---including the truncation mask: renormalise over exactly the token set that was
available at sampling time (one of the GRPO modifications listed in [@ch:grpo2]). Repetition
penalties, logit biases and grammar-constrained decoding all fall under the same rule.

##### Exploit the shared prompt.

GRPO samples $G$ responses per prompt. An engine that supports parallel sampling (`n`$=G$ in vLLM)
or prefix caching ([@ch:serving]) prefills each prompt once and forks its KV blocks
copy-on-write for the $G$ continuations ([@ch:scaling]). For long prompts (agentic tasks with
large contexts, or long system prompts) this removes up to a factor $G$ of prefill work and of KV
memory for the prompt.

##### Return token ids, not text.

The trainer must score exactly the token sequence the engine sampled. Decoding to text and
re-tokenising can change the sequence (merges across the prompt/response boundary, normalisation of
whitespace or special tokens), which silently breaks the ratio. Pass token ids end to end; this is
sometimes called a "token-in, token-out" interface.

##### CUDA graphs, cache size and preemption.

Decode steps of small models are dominated by kernel-launch overhead unless captured in CUDA graphs
([@ch:inference]); engines capture graphs for a set of batch sizes at start-up, which costs
time and memory each time an engine is woken in a colocated setup. The KV cache budget determines
concurrency; when it is exhausted the engine preempts sequences (swap or recompute), which for RL
does not change the result but can multiply the tail.

::: systemslens
[Rollout engines are throughput servers with no SLOs]{.box-title}

Serving engines trade throughput for latency SLOs (TTFT, inter-token latency). An RL rollout has no
per-request latency target at all: only the time until the *whole batch* (or the first $B$ groups)
is done matters. Scheduling should therefore maximise concurrency and prioritise requests that are
likely to be long (start them first, a longest-processing-time heuristic) rather than serve requests
in arrival order. Length predictors trained on previous steps' rollouts have been proposed for
exactly this purpose.
:::

## Multi-turn and agentic rollouts {#sec:rlsys:agentic}

The RLVR loop in this book generates one response per prompt. Agentic RL---training models to use
tools, browse, or edit a code repository---turns each rollout into a *conversation with an
environment*: the model emits a tool call, generation pauses, the environment executes the call (a
shell command, a search query, a unit test), and the observation is appended to the context before
generation resumes ([@fig:rlsys:agentic]).

![Agentic rollouts interleave decoding (orange) with environment steps (green) whose latency can
exceed the decoding time. An engine that blocks a slot while a tool runs wastes the slot;
asynchronous rollout workers release it and re-admit the sequence when the observation arrives,
ideally with its KV cache still resident (prefix
caching).](figures/ch46_rl_systems-10.svg){#fig:rlsys:agentic}

This changes the systems picture in four ways:

1.  **Rollouts become state machines.** A rollout worker drives the loop generate $\rightarrow$
    parse tool call $\rightarrow$ execute $\rightarrow$ append observation $\rightarrow$ generate,
    typically with `asyncio` over many concurrent episodes so that environment latency overlaps
    other episodes' decoding.

2.  **The KV cache must survive pauses.** Re-prefilling the whole context after every tool call is
    quadratic in the number of turns; prefix caching (or explicitly pinned sessions) keeps the
    conversation's KV resident between turns.

3.  **Loss masks get more complex.** Only tokens the model generated should receive gradient; tool
    outputs and environment observations are context, so they are masked out of the loss exactly
    like prompt tokens in SFT ([@ch:instruct]). The token-in/token-out rule matters even more
    because observations are spliced between generated segments.

4.  **Environments are infrastructure.** Each episode may need its own container (a repository
    checkout, a browser, a database), with start-up costs of seconds; environment pools,
    snapshotting, and failure handling become as important as the GPU schedule.

## Reward infrastructure {#sec:rlsys:reward}

In RLVR the reward is a program. That sounds cheap until one counts: 4,096 responses per step, each
requiring answer extraction, normalisation and a symbolic-equivalence check ([@ch:eval]), or
the execution of model-written code against a test suite. Reward computation is therefore a
*service* with its own throughput, latency, and isolation requirements
([@fig:rlsys:reward]).

![Reward computation as a service. Samples are scored as soon as they finish, so reward latency
hides behind the generation tail. Different reward types run on different resources; multi-reward
training normalises each component before combining them
([@ch:grpo2]).](figures/ch46_rl_systems-11.svg){#fig:rlsys:reward}

##### Streaming.

Because responses finish at very different times, the reward of the first finished response can be
computed while the last ones are still decoding. A reward service that consumes a stream of
completions hides almost all of its latency behind the generation tail.

##### Timeouts everywhere.

Symbolic equivalence checking can take unbounded time on adversarial expressions, and generated code
can loop forever. Every reward call needs a hard wall-clock timeout, a defined reward on timeout
(usually 0), and a metric counting timeouts.

##### Isolation.

Executing model-written code is executing untrusted code. A policy under optimisation pressure will
discover anything that raises its reward: deleting or monkeypatching the tests, reading the expected
outputs from disk, exiting with status 0 before the assertions run, or exhausting resources to crash
the grader into a default score. Production sandboxes use containers with seccomp filters,
user-space kernels (gVisor), micro-VMs (Firecracker), or jail tools (nsjail), with no network,
read-only base images, cgroup CPU/memory limits, and a pool of pre-warmed workers to amortise
start-up cost.

``` {#lst:rlsys:sandbox .python caption="A teaching sandbox: a fresh interpreter, empty environment, resource limits and a timeout. Not a security boundary."}
def _limits(cpu_sec, mem_mb):
    def fn():
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_sec, cpu_sec))
        mem = mem_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
        os.setsid()
    return fn

def run_python_reward(code, tests, timeout=2.0, mem_mb=512):
    program = code + "\n\n" + tests + "\n"
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "prog.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            p = subprocess.run([sys.executable, "-I", "-S", path], cwd=tmp,
                               env={}, capture_output=True, text=True,
                               timeout=timeout,
                               preexec_fn=_limits(int(timeout) + 1, mem_mb))
        except subprocess.TimeoutExpired:
            return SandboxResult("timeout", 0.0, time.perf_counter() - t0)
    ok = p.returncode == 0
    status = "pass" if ok else ("fail" if "AssertionError" in p.stderr else "error")
    return SandboxResult(status, float(ok), time.perf_counter() - t0, p.stderr[-500:])
```

::: pitfall
[Reward hacking through the grader]{.box-title}

[@lst:rlsys:sandbox] concatenates the tests after the code, so a response that calls
`os._exit(0)` at import time "passes" every test. Real graders run tests in a separate process or
namespace from the candidate code, check that the expected number of tests actually ran, and treat
any deviation from the protocol as failure. Audit samples with suspiciously high reward.
:::

::: napkinmath
[Sandbox capacity]{.box-title}

A cold Python process costs tens of milliseconds to start; a container, hundreds of milliseconds; a
micro-VM from a snapshot, on the order of 100 ms. If 4,096 code responses per step each run a 0.5 s
test suite and the generation phase lasts 60 s, we need at least $4096\times 0.5/60\approx 34$
concurrent workers just to keep up with the stream---and several times that to keep reward latency
off the critical path when completions arrive in a burst at the end of a synchronous phase.
:::

## A tour of open RL frameworks {#sec:rlsys:frameworks}

With the vocabulary of the previous sections we can place the open frameworks in a common design
space ([@tab:rlsys:frameworks]). The table describes designs as they were documented around
2025; all of these projects evolve quickly, often converging on each other's best ideas, so consult
their current documentation before relying on a detail.

::: {#tab:rlsys:frameworks .table-float}
| Framework | Orchestration | Training / rollout engines | Placement & notable ideas |
|:---|:---|:---|:---|
| TRL (Hugging Face) | single program on top of `accelerate` | Transformers + DeepSpeed/FSDP; vLLM for generation | vLLM either as a separate server process or colocated in the training process; easiest entry point, HF-ecosystem native |
| OpenRLHF (Hu et al., 2024) | Ray actors | DeepSpeed ZeRO; vLLM | originally separate GPU groups per model (actor, critic, reward, reference, vLLM); later a hybrid-engine colocation mode and asynchronous options |
| veRL / HybridFlow (Sheng et al., 2025) | hybrid: single controller (Ray driver) for the RL dataflow, SPMD workers inside each model | FSDP or Megatron-LM; vLLM or SGLang | 3D-HybridEngine for colocated, low-redundancy resharding between training and generation layouts; token-budget dynamic batching |
| NeMo-Aligner (Shen et al., 2024) | Megatron-based distributed jobs | Megatron-LM; TensorRT-LLM-accelerated generation | large-scale PPO/RLHF in the NVIDIA stack; continued in NVIDIA's newer NeMo-RL |
| AReaL (Fu et al., 2025) | asynchronous generation and training services | SGLang rollouts; FSDP/Megatron training | fully asynchronous with bounded staleness and a decoupled PPO objective; interruptible rollouts |

: Design choices of several open RL frameworks for LLMs (snapshot of public designs around 2025).
Other actively developed systems include slime, ROLL, SkyRL, PipelineRL and prime-rl; the space
changes month to month.
:::

### Single controller versus multi-controller

The HybridFlow paper frames a useful distinction. In a *multi-controller* (SPMD) program, every GPU
runs the same script and coordination is implicit in collectives---the natural style for training a
single model ([@ch:distributed]). In a *single-controller* program, one driver issues remote
calls to groups of workers---the natural style for a dataflow of several models, because the RL
algorithm reads like the loop of [@ch:grpo]:

``` {#lst:rlsys:controller .python caption="The RL loop as a single-controller program (pseudocode in the style of Ray-based frameworks). Each call fans out to an SPMD worker group."}
for step in range(num_steps):
    batch = dataloader.next()
    rollouts = rollout_group.generate(batch, n=G)           # vLLM/SGLang workers
    rewards = reward_service.score(rollouts)                # CPU pool / sandboxes
    old_lp = actor_group.compute_logprobs(rollouts)         # FSDP workers, packed
    ref_lp = ref_group.compute_logprobs(rollouts) if beta > 0 else None
    adv = group_normalise(rewards, G)                       # on the driver
    metrics = actor_group.update(rollouts, adv, old_lp, ref_lp)
    rollout_group.load_weights(actor_group.export_weights())  # weight sync
```

The driver handles only small metadata (or references to data held by workers, as Ray object
references), so it does not become a bandwidth bottleneck, while each `*_group` call is an SPMD
collective program internally. HybridFlow argues that this hybrid gives the flexibility of
single-controller dataflow with the efficiency of SPMD execution.

## Checkpointing and fault tolerance {#sec:rlsys:ft}

An RL run's state is larger than a pretraining run's: actor weights and optimizer state, critic
state (PPO), the frozen reference (re-loadable), data-loader position and RNG state, *and* the
in-flight rollout state of an asynchronous pipeline---the replay buffer of finished but untrained
samples, partially generated responses, and the policy version that produced each. Two pragmatic
rules:

- **Checkpoint at step boundaries and drop in-flight rollouts on restart.** The lost work is at most
  one step of generation, and restart logic stays simple. Record the policy version in every sample
  so that resumed async pipelines can enforce the staleness bound.

- **Make rollout engines stateless and restartable.** Engines crash (OOMs from unexpectedly long
  sequences, driver errors). A controller that can re-launch an engine, reload the latest weights
  and re-issue its unfinished prompts turns a crash into a slowdown.

Sharded, asynchronous checkpointing of the trainer ([@ch:pretrain]) applies unchanged.

## Putting it together: a design worksheet {#sec:rlsys:design}

::: napkinmath
[Sizing an 8B RLVR run on 64 GPUs]{.box-title}

Using the numbers from [@sec:rlsys:anatomy]: $W_t \approx 64\times(33+22)$ GPU-s
$\approx 3{,}500$ GPU-s of training and scoring per step, and
$W_g \approx 64 \times 32 \approx 2{,}050$ GPU-s of generation if slots stay busy.

- *Colocated, synchronous:* $(W_g+W_t)/64 \approx 87$ s plus the tail ($\sim$`<!-- -->`{=html}80 s)
  plus switch overhead: $\sim$`<!-- -->`{=html}170--180 s per step.

- *Split, one-step async:* balance $G_{\text{gen}}/G = W_g/(W_g+W_t)\approx 0.37$, i.e. 24
  generation GPUs and 40 training GPUs; the per-step time approaches
  $\max(W_g/24, W_t/40)\approx 87$ s, but the tail now sits inside the generation term: unless it is
  mitigated (partial rollouts, caps), generation takes $\approx 85+80 = 165$ s and the trainer idles
  half the time.

- *Split, async + partial rollouts:* with the tail absorbed into the next phase, step time
  approaches the balanced $\approx 90$ s, a $\sim$`<!-- -->`{=html}2$\times$ improvement over the
  synchronous baseline, at the cost of staleness $\le 1$--2 and mixed-policy samples.

These figures are idealised, but they show where to look first: the tail, then the balance between
pools, then the constant overheads.
:::

A checklist for a new RL system:

1.  Measure the response-length distribution before choosing a schedule; it determines whether
    asynchrony or tail mitigation matters more.

2.  Recompute $\pi_{\theta_{\text{old}}}$ in the trainer and log mismatch metrics; enable TIS/MIS.

3.  Pack the scoring and training passes; normalise token-level losses by the global token count.

4.  Bound staleness explicitly, record policy versions per sample, and correct with importance
    weights where staleness is non-zero.

5.  Stream rewards; put timeouts and isolation around every reward call.

6.  Verify weight sync with a log-prob comparison during bring-up.

## Summary {#sec:rlsys:summary}

- An RL step chains bandwidth-bound generation, CPU-bound or GPU-bound rewards, and compute-bound
  scoring/training through a weight-update dependency. Generation usually dominates wall-clock time,
  mostly because of the long tail of response lengths.

- Colocation time-shares devices (no idle pools, switch overhead); disaggregation dedicates devices
  (specialised layouts, concurrency) but only beats colocation when combined with asynchrony,
  heterogeneous hardware, or high switch costs.

- Weight sync is a resharding problem as much as a bandwidth problem; validate it by comparing
  log-probs.

- Asynchronous RL bounds staleness and corrects for it with importance weights (decoupled
  objectives); partial rollouts tame the tail at the cost of mixed-policy, length-biased batches.

- Packing with block-diagonal masks and per-segment positions removes padding waste in scoring and
  training passes; loss normalisation must use global token counts.

- Inference engines sample from $\mu\neq\pi_{\theta_{\text{old}}}$ even with identical weights;
  truncated or masked importance sampling corrects this cheaply, batch-invariant kernels remove it.

- Rewards are a service: stream them, time them out, and isolate code execution; assume the policy
  will find any loophole in the grader.

## Exercises {#sec:rlsys:exercises}

::: exercise
[Optimal split]{.box-title}

Derive the optimal $G_{\text{gen}}$ for [@eq:rlsys:sync] and show that
$T^\star_{\text{sync-split}} \ge T_{\text{coloc}} - T_{\text{switch}}$ with equality iff one stage
has zero work. For which values of $T_{\text{switch}}$ does the synchronous split win?
:::

::: exercise
[KV budget]{.box-title}

Repeat the generation napkin math of [@sec:rlsys:anatomy] for Qwen3-4B (36 layers, 8 KV
heads, head dimension 128) on an 80 GB GPU with average live context 6,000 tokens. How many
concurrent sequences fit, and what is the upper bound on decode tokens/s per GPU?
:::

::: exercise
[Staleness bound]{.box-title}

In the `async` policy of the simulator, generation of batch $k$ may start only after training step
$k-s-1$ has finished. Prove that every sample of batch $k$ has staleness at most $s$, and show with
a timeline that $s=0$ recovers the synchronous split schedule.
:::

::: exercise
[Loss normalisation bug]{.box-title}

Two micro-batches contain 100 and 1,000 response tokens with mean token losses $\ell_1$ and
$\ell_2$. Compute the effective loss when (a) micro-batch means are averaged, and (b) the
token-level definition is used. Write the one-line fix for a gradient-accumulation loop.
:::

::: exercise
[TIS bias and variance]{.box-title}

For a single token with $\rho\sim$ a log-normal distribution with $\mathbb{E}_\mu[\rho]=1$, compute
the bias $\mathbb{E}_\mu[\min(\rho,C)]-1$ as a function of $C$ and the log-normal variance. Why is a
small bias acceptable here?
:::

::: exercise
[Packing efficiency]{.box-title}

Show that FFD never uses more than $2\,\mathrm{OPT}$ bins with a short argument (the full
$\tfrac{11}{9}$ bound is harder). Then use `efficiency_report` to compare padding, length-sorted
padding and packing on log-normal lengths with $\sigma\in\{0.25,0.5,1\}$.
:::

::: exercise
[Partial-rollout bias]{.box-title}

Using the simulator, record the mean length of samples consumed in each step under `partial` and
compare with the generated length distribution. How many steps does it take for the consumed-length
distribution to match the generated one? How does `oversample` change this?
:::

::: exercise
[Weight sync across sites]{.box-title}

A 32B policy is trained in one datacenter and rolled out in another over a 100 Gb/s link. Estimate
the sync time in bf16, fp8 and with LoRA rank 64 on all linear layers. Which schedule from
[@sec:rlsys:async] would you use?
:::

::: exercise
[Attacking the grader]{.box-title}

List five ways a policy could obtain reward from [@lst:rlsys:sandbox] without solving the
task, and give a mitigation for each. Implement two of them as tests for your own sandbox.
:::

::: exercise
[Measurement]{.box-title}

Run `bench.py sim` and add a policy `async+partial` to the simulator (partial rollouts in a
generator pool that runs concurrently with training). Plot throughput and mean staleness against
`oversample`.
:::

::: exercise
[Design]{.box-title}

You have 32 H100s and a 14B model. Propose a placement, schedule and parallel layout for an RLVR run
with $B=256$, $G=16$, mean response length 6,000 and a cap of 32,000 tokens. Justify each choice
with a napkin estimate.
:::

## Lab 46: Building blocks of an RL training system {#sec:rlsys:lab}

::: labbox
[Building blocks of an RL training system]{.box-title}

**Code:** `code/labs/lab46_rlsys`

**Goals**

Implement and measure the pieces that surround the GRPO update in a production RL framework: packed
scoring passes, mismatch correction, schedule simulation and sandboxed rewards.

**Setup**

``` bash
cd code
pytest -q labs/lab46_rlsys                    # grades starter.py
LAB_IMPL=solution pytest -q labs/lab46_rlsys  # reference implementation
```

Everything runs on a CPU in under a minute with tiny models; `bench.py` accepts a real Qwen3
checkpoint for the mismatch measurement.

**Tasks**

1.  **Packing.** Implement `ffd_pack` and `efficiency_report` (padded, length-sorted padded, packed
    real-token fractions).

2.  **Varlen log-probs.** Implement `block_causal_mask` and `packed_completion_logprobs` on top of
    the provided `varlen_forward`. The tests compare against unpacked `sequence_logprob` for both
    Qwen3- and GPT-shaped models.

3.  **Mismatch.** Implement `is_weights` (truncate/mask/none), `mismatch_metrics` (including the
    $k_3$ KL estimate) and `clipped_pg_loss` with an optional TIS factor. Check that the gradient at
    $\pi_\theta=\pi_{\theta_{\text{old}}}$ is $-A/n_{\text{tokens}}$.

4.  **Pipeline simulation.** Implement `run_generation` (continuous batching with an early stop) and
    `weight_sync_seconds`; the provided `simulate` drives your code through the colocated, sync,
    async and partial policies.

5.  **Sandboxed reward.** Implement `run_python_reward` and `score_batch` with timeouts, resource
    limits and an empty environment.

**Measurements**

- `bench.py sim`: throughput, utilisation, staleness and mixed-policy fraction for each policy as
  $\sigma$ varies. Expect async to help most for light tails and partial rollouts to dominate for
  heavy tails ([@fig:rlsys:simresults]).

- `bench.py packing`: wall time of padded vs. packed log-prob passes and the real-token fractions.
  Expect a large gap whenever lengths are heavy-tailed.

- `bench.py mismatch`: bf16-vs-fp32 mismatch metrics, ideally on real Qwen3-0.6B weights; look at
  the maximum ratio and the fraction outside $[0.5, 2]$.

**Deliverables**

Passing tests; a one-page note with the simulator sweep (a plot like
[@fig:rlsys:simresults]), the packing timings, and the mismatch metrics, each with a
sentence of interpretation.

**Stretch goals**

- Add an `async+partial` policy and a staleness-aware sampler that drops samples older than $s$
  versions; plot throughput vs. staleness.

- Wire `clipped_pg_loss` with TIS into your GRPO loop from [@ch:grpo2], generate rollouts
  with a bf16 copy of the model and train in fp32; compare reward curves with and without TIS.

- Replace the dense mask in `varlen_forward` with a loop over segments that calls
  `F.scaled_dot_product_attention` per segment, and measure the speedup.
:::
