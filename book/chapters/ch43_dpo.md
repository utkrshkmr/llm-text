# Direct Preference Optimization and Its Family {#ch:dpo}

[@ch:rlhf] built the classical RLHF pipeline: collect human comparisons, fit a reward model,
then run PPO with four networks in memory (policy, reference, reward, value) and a generation engine
in the loop. It works, but it is the most complex training system in this book outside large-scale
pretraining. In 2023, Rafailov and colleagues showed that for the KL-regularised objective RLHF
optimises, the reward-modelling stage and the RL stage can be *collapsed into one supervised loss on
the preference pairs themselves*. The result, *Direct Preference Optimization* (DPO), needs no
reward model, no value function, no sampling during training and no PPO machinery. A DPO step is two
forward passes over (prompt, response) pairs and one backward pass. That is the cost profile of
supervised fine-tuning, which makes DPO easy to run, to scale and to debug.

For a systems person this chapter has two halves. The first derives DPO carefully, because every
design choice in an implementation follows from where the terms in the loss come from: summing over
response tokens only, caching the reference, interpreting the logged metrics. The second half is
about running it: data formats, the pair-batch layout, the reference-model cache, LoRA-DPO, launch
recipes, what healthy logs look like, the failure modes, the large family of variants (cDPO, IPO,
KTO, SimPO, ORPO, CPO, online DPO), and what a DPO step costs in memory and FLOPs.

![Roadmap of this chapter. Steps 1--4 derive DPO; steps 5--8 implement and run it; steps 9--12 place
it among its relatives and cost it out.](figures/ch43_dpo-1.svg){#fig:dpo:roadmap}

## Why skip the reward model? {#sec:dpo:why}

Recall the three-stage pipeline of [@ch:rlhf] ([@fig:dpo:pipelines], top). After
supervised fine-tuning (SFT) produces $\pi_{\text{SFT}}$, annotators compare pairs of responses
$(y_w, y_l)$ to a prompt $x$, with $y_w$ the *winner* ("chosen") and $y_l$ the *loser* ("rejected").
A reward model $r_\phi$ is fitted with the Bradley--Terry (BT) likelihood $$\begin{equation}
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x,y_w,y_l)}\big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\big],
\label{eq:dpo:bt}
\end{equation}$$ and the policy is then trained online against it, with a penalty for drifting from
the reference $\pi_{\text{ref}}=\pi_{\text{SFT}}$: $$\begin{equation}
\max_{\pi_\theta}\ \mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}\big[r_\phi(x,y)\big]
 \;-\;\beta\,\operatorname{KL}\big(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big).
\label{eq:dpo:rlobj}
\end{equation}$$ The RL stage is where the systems pain lives. Every step needs fresh samples from
the current policy, so a generation engine has to sit in the loop and receive weight updates
([@ch:rlsys]). A critic as large as the policy has to be trained, and the reward model must
stay in memory. PPO's many hyperparameters (clip range, GAE $\lambda$, value-loss coefficient, KL
controller) all interact.

DPO keeps the *objective* ([-@eq:dpo:rlobj]) and the *data* (the same comparison pairs) but
removes both the explicit reward model and the RL loop ([@fig:dpo:pipelines], bottom). The
trick is that ([-@eq:dpo:rlobj]) has a closed-form solution, and that solution lets us
express the reward in terms of the policy.

![RLHF and DPO optimise the same KL-regularised objective from the same data. DPO replaces the
reward model and the online PPO loop with a classification-style loss whose cost profile is that of
supervised fine-tuning.](figures/ch43_dpo-2.svg){#fig:dpo:pipelines}

## Deriving DPO {#sec:dpo:derive}

### The optimal policy of the KL-regularised objective

Fix a prompt $x$ and drop it from the notation. For a reward $r(y)$, we want the distribution $\pi$
over responses that maximises
$\mathbb{E}_{\pi}[r(y)] - \beta\,\operatorname{KL}(\pi\,\|\,\pi_{\text{ref}})$.

::: {#prop:dpo:gibbs .proposition}
\[Gibbs policy\] The unique maximiser is $$\begin{equation}
\pi^*(y) = \frac{1}{Z}\,\pi_{\text{ref}}(y)\exp\!\Big(\frac{r(y)}{\beta}\Big),\qquad
Z = \sum_{y}\pi_{\text{ref}}(y)\exp\!\Big(\frac{r(y)}{\beta}\Big).
\label{eq:dpo:gibbs}
\end{equation}$$
:::

::: proof
*Proof.* Write the objective as
$J(\pi)=\sum_y \pi(y)\big[r(y)-\beta\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}\big]$. Substituting
$r(y)=\beta\log\frac{\pi^*(y)}{\pi_{\text{ref}}(y)}+\beta\log Z$ from ([-@eq:dpo:gibbs])
gives $$J(\pi)=\sum_y\pi(y)\Big[\beta\log\frac{\pi^*(y)}{\pi(y)}\Big]+\beta\log Z
      =-\beta\,\operatorname{KL}(\pi\,\|\,\pi^*)+\beta\log Z .$$ $\beta\log Z$ does not depend on
$\pi$, and the KL divergence is non-negative and zero only at $\pi=\pi^*$. ◻
:::

The proof shows something useful in practice too: maximising the RLHF objective is the same as
minimising the reverse KL to an exponentially tilted reference. Small $\beta$ means a sharp tilt
(the policy chases reward); large $\beta$ keeps $\pi^*$ close to $\pi_{\text{ref}}$.

### Inverting for the reward

[@eq:dpo:gibbs] expresses the policy in terms of the reward. Take logs and solve for the
reward instead (restoring $x$): $$\begin{equation}
r(x,y) = \beta\log\frac{\pi^*(y\mid x)}{\pi_{\text{ref}}(y\mid x)} + \beta\log Z(x).
\label{eq:dpo:reparam}
\end{equation}$$ Any reward function corresponds to some optimal policy, and
([-@eq:dpo:reparam]) turns that around: *the log-ratio of a policy to the reference, scaled
by $\beta$, is a reward for which that policy is optimal*, up to a prompt-dependent constant
$\beta\log Z(x)$. Computing $Z(x)$ exactly is hopeless, since it sums over all possible responses.
But BT only ever looks at reward *differences* between two responses to the same prompt, and in that
difference $Z(x)$ cancels.

### Plugging into Bradley--Terry

Parameterise the policy as $\pi_\theta$ and substitute ([-@eq:dpo:reparam]) into the BT
likelihood ([-@eq:dpo:bt]): $$\begin{equation}
\boxed{\;
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\Big[\log\sigma\Big(
\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}
-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big].\;}
\label{eq:dpo:loss}
\end{equation}$$ That is the whole method. Define the *implicit reward*
$\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ and the margin
$h=\hat r_\theta(x,y_w)-\hat r_\theta(x,y_l)$. Then
$\mathcal{L}_{\text{DPO}}=-\mathbb{E}[\log\sigma(h)]$: DPO is the BT reward-model loss, with the
reward model *being* the policy's log-ratio. Fitting the implicit reward to the preferences moves
the policy directly to the optimum of ([-@eq:dpo:rlobj]) for that reward. This is the sense
of the DPO paper's subtitle, "your language model is secretly a reward model".

::: keyidea
[What DPO assumes]{.box-title}

DPO inherits two modelling assumptions. (1) Preferences follow Bradley--Terry: the probability that
$y_w\succ y_l$ is a logistic function of a reward difference. (2) We want the optimum of the
*KL-regularised* objective with reference $\pi_{\text{ref}}$. The optimisation is offline: the
expectation in ([-@eq:dpo:loss]) is over a fixed dataset, not over samples from $\pi_\theta$.
That last point is the source of both DPO's simplicity and most of its failure modes
([@sec:dpo:failures]).
:::

### How DPO uses Bradley--Terry implicitly

Compare DPO with explicit reward modelling term by term. A reward model is an LM trunk with a scalar
head ([@ch:rlhf]), trained by ([-@eq:dpo:bt]), after which the head is thrown away and
only its scores are used in PPO. DPO uses the same likelihood but with a different hypothesis class
for $r$: functions of the form $\beta\log\pi_\theta/\pi_{\text{ref}}$. Two consequences follow.

- **The reward is only defined up to $\beta\log Z(x)$.** Implicit rewards of responses to the *same*
  prompt are comparable; rewards across *different* prompts are not. Do not rank prompts by implicit
  reward, and do not use the DPO model as a general-purpose reward model for best-of-$N$ across
  prompts without care.

- **The implicit reward is a sum over tokens.** Because
  $\log\pi(y\mid x)=\sum_t\log\pi(y_t\mid x,y_{<t})$, the implicit reward decomposes into per-token
  log-ratios. DPO can therefore be read as learning token-level credit assignment, and later work
  interprets the per-token log-ratios as an optimal advantage-like quantity (Rafailov et al., 2024).
  In practice, plotting per-token log-ratios over a response is an excellent debugging tool: you can
  see *which* tokens the update promoted or suppressed.

### The gradient: a weighted likelihood update {#sec:dpo:grad}

Differentiate ([-@eq:dpo:loss]). With $\frac{d}{dh}[-\log\sigma(h)]=-\sigma(-h)$,
$$\begin{equation}
\nabla_\theta\mathcal{L}_{\text{DPO}} = -\beta\,\mathbb{E}\Big[\underbrace{\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)}_{\text{weight }w\in(0,1)}
\Big(\underbrace{\nabla_\theta\log\pi_\theta(y_w\mid x)}_{\text{raise winner}}
-\underbrace{\nabla_\theta\log\pi_\theta(y_l\mid x)}_{\text{lower loser}}\Big)\Big].
\label{eq:dpo:grad}
\end{equation}$$ Each pair contributes "maximum likelihood on the winner, unlikelihood on the
loser", weighted by how *wrong* the current implicit reward is about that pair. When the implicit
reward already ranks $y_w$ far above $y_l$ the weight goes to zero, and the pair stops contributing
([@fig:dpo:grad]). The weight is what distinguishes DPO from naive "unlikelihood" training,
which keeps pushing the loser's probability down forever and degenerates; the original DPO paper
reports exactly this degeneration for the unweighted version.

![The DPO gradient weight $\sigma(-h)$ as a function of the implicit-reward margin (analytic).
Mis-ranked pairs get a full likelihood/unlikelihood update; well-ranked pairs fade out. At
initialisation ($\pi_\theta=\pi_{\text{ref}}$) every margin is $0$ and every weight is
$1/2$.](figures/ch43_dpo-3.svg){#fig:dpo:grad}

::: pitfall
[The prompt cancels, so truncate it freely but consistently]{.box-title}

$\log\pi(x,y_w)-\log\pi(x,y_l)=\log\pi(y_w\mid x)-\log\pi(y_l\mid x)$: the prompt's own
log-probability cancels inside each ratio-difference, both for $\pi_\theta$ and for
$\pi_{\text{ref}}$. So you may sum log-probs over the whole concatenated sequence and get the same
loss. But you *must* then use exactly the same prompt tokens for winner, loser, policy and
reference. If the reference cache was built with one chat template or truncation policy and training
uses another, the prompt terms no longer cancel and the loss is silently wrong. Summing over
response tokens only (as we do) makes the code robust to this.
:::

## Preference data in practice {#sec:dpo:data}

A DPO dataset is a list of triples. On disk, the most portable format is JSON Lines with one record
per comparison:

    {"prompt": "How do I boil an egg?", "chosen": "Place the egg in ...", "rejected": "Eggs are ...",
     "source": "helpful", "chosen_score": 8, "rejected_score": 3}

Real datasets arrive in other shapes, and the first job is a converter into this one.

- **Paired conversations.** Some corpora store two complete transcripts that share a prefix and
  differ in the final assistant turn, for example Anthropic's HH-RLHF, whose records hold a `chosen`
  and a `rejected` dialogue written as alternating "`Human:`"/"`Assistant:`" turns. The prompt is
  everything up to the last assistant turn; the responses are the two final turns. Decide explicitly
  what to do with multi-turn dialogues whose earlier turns differ between the two sides: dropping
  them is the simplest correct choice.

- **Rated completions.** Datasets such as UltraFeedback attach several completions with scalar
  ratings to each prompt. Turn them into pairs by taking the best against a random lower-rated one
  (or the worst), drop ties, and optionally keep the score gap as metadata to filter or weight
  pairs.

- **Self-generated (on-policy) pairs.** Sample $k$ responses from $\pi_{\text{ref}}$, score them
  with a reward model, a verifier ([@ch:eval]) or an LLM judge, and keep (best, worst). Data
  drawn from the reference policy itself matches DPO's assumptions better than responses written by
  other models ([@sec:dpo:failures]).

``` {#lst:dpo:tokenize .python caption="Loading JSONL pairs and tokenising them with a chat template. Prompts are truncated from the left (keep the latest turns), responses from the right; EOS is appended to both responses."}
def tokenize_pair(rec, encode, eos_id, template=lambda p: p,
                  max_prompt=512, max_response=512):
    if rec["chosen"] == rec["rejected"]:
        return None                        # no preference signal
    p = encode(template(rec["prompt"]))[-max_prompt:]
    c = encode(rec["chosen"])[: max_response - 1] + [eos_id]
    r = encode(rec["rejected"])[: max_response - 1] + [eos_id]
    return PreferencePair(p, c, r, {k: v for k, v in rec.items()
                                    if k not in ("prompt", "chosen", "rejected")})

# e.g. with Qwen3's ChatML template from the Qwen3 chapter:
tok = Qwen3Tokenizer("ckpt/qwen3-0.6b/tokenizer.json")
pairs = [tokenize_pair(r, tok.encode, tok.eos_token_id,
                       template=tok.apply_chat_template)
         for r in load_preference_jsonl("data/prefs.jsonl")]
pairs = [p for p in pairs if p is not None]
```

::: pitfall
[Append EOS, and use the template the SFT model was trained with]{.box-title}

If responses lack EOS, the policy is never taught where a preferred answer *ends*, and length
drifts. If the template differs from the one used to train $\pi_{\text{ref}}$
([@ch:instruct]), the reference log-probs are computed on out-of-distribution inputs, the "KL
anchor" points somewhere odd, and step-0 metrics look strange (see the checklist in
[@sec:dpo:recipe]).
:::

##### Look at the data before training.

For a few dozen random pairs, read the prompt and both responses and ask: do you agree with the
label? Is the chosen response systematically longer? (Compute the mean length difference; on many
public datasets it is clearly positive, which DPO will happily learn as "longer is better".) Are
there near-duplicates or identical pairs? Is the rejected response sometimes truncated mid-sentence
(a giveaway artefact the model can key on)? Ten minutes here saves days of confusing runs.

### Building pairs on-policy {#sec:dpo:onpolicy}

When no suitable dataset exists, or when you want pairs that match the policy's own distribution,
generate them. One round is: sample $k$ responses per prompt from the current model with
temperature, score each with whatever judge you trust, and keep the best and the worst if their
scores differ by enough to be a reliable preference ([@lst:dpo:onpolicy]). With a verifier
([@ch:eval]) the scores are exact; with a reward model or LLM judge, the gap threshold
filters out pairs where the judge is guessing.

``` {#lst:dpo:onpolicy .python caption="One round of on-policy pair construction (`llmsys.prefopt.build_onpolicy_pairs`, abridged). Generation uses the batched, left-padded, KV-cached path from [@ch:serving]."}
@torch.no_grad()
def build_onpolicy_pairs(model, prompts, score_fn, pad_id, eos_id, k=4,
                         max_new_tokens=64, temperature=1.0, min_gap=0.0):
    flat = [list(p) for p in prompts for _ in range(k)]   # k copies each
    outs = generate_batch(model, flat, max_new_tokens, pad_id=pad_id,
                          eos_id=eos_id, temperature=temperature)
    pairs = []
    for i, p in enumerate(prompts):
        cands = sorted((score_fn(p, outs[i * k + j]), outs[i * k + j])
                       for j in range(k))
        (lo, worst), (hi, best) = cands[0], cands[-1]
        if hi - lo > min_gap and best != worst:
            pairs.append(PreferencePair(list(p), best, worst, {"gap": hi - lo}))
    return pairs
```

::: systemslens
[Pair generation is an inference job: run it like one]{.box-title}

The $k$ samples of a prompt share their prefix, so a serving engine with prefix caching
([@ch:serving]) prefills the prompt once and decodes $k$ branches. At scale, generation runs
offline on an inference fleet (vLLM, SGLang or the mini engine of [@ch:serving]), writes
JSONL shards with the sampling parameters and the generating checkpoint's hash, and a separate
scoring job (a reward model is a prefill-only workload; a judge is another generation job) adds
scores. Keep generation, scoring and training as separate, restartable stages that communicate
through files, just as in the data pipelines of [@ch:data]. That makes each stage
independently scalable and debuggable, and lets you re-score without re-generating.
:::

## Implementing DPO {#sec:dpo:impl}

### The pair batch

The workhorse is one forward pass over a $(2B,T)$ batch: rows $0..B{-}1$ hold prompt+chosen and rows
$B..2B{-}1$ prompt+rejected, in the same order ([@fig:dpo:batch]). A boolean `loss_mask`
marks response tokens. Padding goes on the right: under a causal mask real tokens never attend to
later padding, so no attention mask is needed for training (contrast with left padding for
generation, [@ch:serving]).

![The DPO pair batch: one $(2B,T)$ tensor, chosen half first, same pair order in both halves,
right-padded. Log-probabilities are summed only where the mask is 1, giving $\log\pi(y\mid x)$ for
each row; splitting the result at $B$ yields winner and loser
terms.](figures/ch43_dpo-4.svg){#fig:dpo:batch}

``` {#lst:dpo:batch .python caption="Building the pair batch (from `llmsys/prefopt.py`)."}
def build_pair_batch(pairs, pad_id):
    rows, masks = [], []
    for side in ("chosen", "rejected"):          # chosen half first
        for pr in pairs:
            resp = getattr(pr, side)
            rows.append(list(pr.prompt) + list(resp))
            masks.append([False] * len(pr.prompt) + [True] * len(resp))
    ids, loss_mask = _pad_rows(rows, masks, pad_id)   # right padding
    return {"input_ids": ids, "loss_mask": loss_mask}
```

### Sequence log-probabilities

``` {#lst:dpo:logps .python caption="Response-only sequence log-probabilities. The logit at position $t$ scores token $t{+}1$, so both targets and mask shift by one. `gather` minus `logsumexp` avoids materialising a $(2B,T,V)$ log-softmax tensor."}
def sequence_logps(model, input_ids, loss_mask):
    logits = model(input_ids)[:, :-1].float()        # (2B, T-1, V)
    targets = input_ids[:, 1:]
    mask = loss_mask[:, 1:].to(logits.dtype)
    tok = (logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
           - torch.logsumexp(logits, dim=-1))         # log p(target)
    return (tok * mask).sum(-1), mask.sum(-1)         # sum, #tokens
```

::: pitfall
[Sum, don't average, for DPO]{.box-title}

[@eq:dpo:loss] uses $\log\pi(y\mid x)$, the *sum* over response tokens. Averaging per token
silently changes the objective: it removes the dependence on length from both terms and corresponds
to a different (length-normalised) implicit reward. That is a legitimate design, but it is SimPO's
([@sec:dpo:family]), not DPO's, and it changes the scale on which $\beta$ operates. Pick
one deliberately and log which one you used.
:::

### The loss

``` {#lst:dpo:loss .python caption="DPO with optional conservative label smoothing (cDPO, [@sec:dpo:family]). The rewards are returned detached for logging."}
def dpo_loss(pol_c, pol_r, ref_c, ref_r, beta=0.1, label_smoothing=0.0):
    r_c = beta * (pol_c - ref_c)              # implicit reward of y_w
    r_r = beta * (pol_r - ref_r)              # implicit reward of y_l
    h = r_c - r_r
    loss = (-(1 - label_smoothing) * F.logsigmoid(h)
            - label_smoothing * F.logsigmoid(-h))
    return loss, r_c.detach(), r_r.detach()
```

Use `F.logsigmoid`, never `torch.log(torch.sigmoid(h))`: for large negative $h$ the latter
underflows to $\log 0$.

### The reference model: cache it, or get it for free {#sec:dpo:ref}

$\pi_{\text{ref}}$ is frozen, so $\log\pi_{\text{ref}}(y_w\mid x)$ and
$\log\pi_{\text{ref}}(y_l\mid x)$ are constants of the dataset. There are three ways to provide
them.

1.  **A second model on the device.** Simplest to write; costs a full copy of the weights in memory
    and an extra forward per step. A common two-GPU recipe puts the policy on one device and the
    reference on another.

2.  **Precompute and cache** ([@lst:dpo:cache]). One forward pass over the dataset before
    training stores two floats per pair. Training then never loads the reference. For multiple
    epochs this also saves compute; for one epoch it costs the same FLOPs, but they are spent
    *before* training, so every GPU byte can go to the policy during training.

3.  **LoRA-DPO.** If the policy is $\pi_{\text{ref}}$ plus LoRA adapters
    ([@sec:instruct:lora]), the reference *is* the frozen base: zero the adapters' scaling
    and run the same model ([@lst:dpo:lora]). No second copy, no cache file, and the
    reference stays consistent with the policy's exact numerics.

``` {#lst:dpo:cache .python caption="Caching reference log-probs (abridged). The optional third column stores KTO's mismatched-pair term ([@sec:dpo:family])."}
@torch.no_grad()
def precompute_reference_logps(ref_model, pairs, pad_id, batch_size=16):
    ref_model.eval()
    out = []
    for s in range(0, len(pairs), batch_size):
        b = build_pair_batch(pairs[s: s + batch_size], pad_id)
        lp, _ = sequence_logps(ref_model, b["input_ids"], b["loss_mask"])
        B = lp.shape[0] // 2
        out.append(torch.stack([lp[:B], lp[B:]], dim=1).float())
    return torch.cat(out)                       # (N, 2): [chosen, rejected]

torch.save({"ref_logps": cache, "ref_ckpt": "ckpt/sft-step-4000",
            "template": "chatml-v1", "n_pairs": len(pairs)}, "cache/ref.pt")
```

``` {#lst:dpo:lora .python caption="LoRA-DPO: the base weights are the reference model."}
@contextmanager
def disable_lora(model):
    mods = [m for m in model.modules() if isinstance(m, LoRALinear)]
    saved = [m.scaling for m in mods]
    for m in mods:
        m.scaling = 0.0                   # y = base(x) + 0 * B(A(x))
    try:
        yield model
    finally:
        for m, s in zip(mods, saved):
            m.scaling = s

policy = apply_lora(load_sft_model(), rank=16, alpha=32)
with disable_lora(policy):
    ref_logps = precompute_reference_logps(policy, pairs, pad_id)
```

::: pitfall
[Cache hygiene]{.box-title}

A reference cache is only valid for the exact (reference checkpoint, tokenizer, template, truncation
lengths, data order) it was built with. Store these alongside the tensor (as in
[@lst:dpo:cache]) and assert them at load time. Index the cache by *example id*, not by
batch position, so shuffling cannot misalign it. One cheap end-to-end check: at step 0 the DPO loss
must equal $\ln 2\approx 0.6931$ to three decimals (next section). If it does not, the cache and the
policy disagree.
:::

### The training step

``` {#lst:dpo:step .python caption="One optimizer step of DPO with gradient accumulation (the lab's `dpo_train_loop`, abridged)."}
for step in range(steps):
    for _ in range(grad_accum):
        idx = next_indices(batch_size)                       # shuffled
        b = build_pair_batch([pairs[i] for i in idx], pad_id)
        lp, _ = sequence_logps(policy, b["input_ids"], b["loss_mask"])
        B = lp.shape[0] // 2
        loss, rc, rr = dpo_loss(lp[:B], lp[B:],
                                ref_logps[idx, 0], ref_logps[idx, 1], beta)
        (loss.mean() / grad_accum).backward()
        log_metrics(loss, rc, rr, lp[:B], lp[B:])
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
```

## Running DPO: a recipe {#sec:dpo:recipe}

### Hyperparameters

::: {#tab:dpo:hparams .table-float}
| **Knob** | **What it does** | **Typical starting point** |
|:---|:---|:---|
| $\beta$ | strength of the KL anchor; scale of implicit rewards | $0.1$ (sweep $0.01$--$0.5$) |
| learning rate | full fine-tuning is very sensitive | $\sim 5\times10^{-7}$--$10^{-6}$ (full), $\sim10^{-5}$--$10^{-4}$ (LoRA) |
| effective batch | pairs per optimizer step | 32--128 pairs |
| epochs | passes over fixed data; overfitting comes fast | 1--3 |
| warmup | avoids a large early step | a few % of steps, linear |
| optimizer | the DPO paper used RMSprop; AdamW is common | AdamW, no weight decay |
| max lengths | prompt/response truncation | cover $\geq95\%$ of pairs; log what is cut |

: DPO hyperparameters and commonly used starting values. These are conventions, not laws: the right
learning rate in particular depends on the model size and on full vs LoRA training.
:::

### Project layout and launch

    dpo/
      configs/qwen3-0.6b-dpo.yaml   # model, data, beta, lr, batch, lengths, seeds
      data/prefs.train.jsonl         # {"prompt","chosen","rejected", ...}
      data/prefs.val.jsonl           # ~200-1000 held-out pairs
      cache/ref.train.pt             # reference log-probs + metadata
      runs/<name>/metrics.jsonl      # one line per step
      runs/<name>/ckpt-best/         # best val reward-accuracy checkpoint

``` bash
# 1) reference cache (once per dataset x reference checkpoint)
python -m dpo.cache_ref --config configs/qwen3-0.6b-dpo.yaml
# 2a) single GPU (LoRA) or one GPU each for policy/reference
python -m dpo.train --config configs/qwen3-0.6b-dpo.yaml
# 2b) full fine-tuning of a larger model: shard the policy (see the distributed-training chapters)
torchrun --nproc_per_node 8 -m dpo.train --config configs/qwen3-8b-dpo.yaml --fsdp
```

The modules named here are the pattern to build, not files in this repository; [@lab:dpo]'s
`llmsys/prefopt.py` implements their core (`precompute_reference_logps`, `train_preference`). For
multi-GPU full fine-tuning, the policy is wrapped with FSDP exactly as in [@ch:sharding]; the
cached reference log-probs are just another per-example field that the data loader returns.

### What healthy logs look like

Log, every step: the loss; **reward accuracy** (fraction of pairs with $\hat r_w>\hat r_l$); the
**margin** $\mathbb{E}[h]$; the mean implicit rewards of chosen and rejected; the mean **log-probs**
$\log\pi_\theta(y_w\mid x)$ and $\log\pi_\theta(y_l\mid x)$; and the gradient norm.

- **Step 0**: $\pi_\theta=\pi_{\text{ref}}$, so every implicit reward is exactly 0: loss $=\ln 2$,
  margin $=0$, and reward accuracy is $0.5$ if ties count half (0 if you use a strict $>$). Anything
  else is a bug (cache mismatch, template mismatch, dropout left on in one pass, a different dtype
  for reference and policy).

- **Early training**: the margin rises, accuracy climbs fast, and the rejected reward goes negative
  much faster than the chosen reward rises.

- **Watch the chosen log-prob.** It often *decreases* even though its implicit reward increases
  relative to the loser ([@sec:dpo:failures]). A small drop is normal; a collapse of both
  log-probs by tens of nats signals over-optimisation.

- **Validation reward accuracy** on held-out pairs is the cheapest model-selection signal. Keep the
  checkpoint with the best value, then confirm with generation-based evaluation.

::: {#tab:dpo:debug .table-float}
| **Symptom** | **Likely cause** | **Check** |
|:---|:---|:---|
| step-0 loss $\neq\ln 2$ | reference cache built with a different template, truncation, dtype or checkpoint; dropout on | recompute 8 pairs on the fly and diff against the cache |
| reward accuracy stuck at 0.5 | loss mask covers prompt only or is shifted wrongly; learning rate far too small | assert mask sums equal response lengths; overfit 16 pairs |
| train accuracy $\to1$ in a few hundred steps, val flat | memorising a small or duplicated set | dedupe, hold out by prompt, fewer epochs, larger $\beta$ |
| both log-probs fall by tens of nats | over-optimisation / likelihood displacement | lower lr or step count, raise $\beta$, add NLL on $y_w$ |
| generations get longer every checkpoint | chosen responses are longer on average | length stats of the data; LC win rate; SimPO/R-DPO |
| NaN loss | $\log\sigma$ computed as $\log(\sigma)$; fp16 logits | use `F.logsigmoid`; fp32 log-softmax |
| OOM at the LM head | $(2B,T,V)$ logits materialised | gather$-$logsumexp, chunking, response slicing |

: A DPO debugging checklist.
:::

When something looks wrong, [@tab:dpo:debug] lists the usual suspects in the order to check
them.

### Evaluating a DPO model

Implicit-reward accuracy measures how well the model *ranks* the dataset's pairs, not how good its
*generations* are. For the latter, generate on held-out prompts and evaluate as in
[@ch:eval]: pairwise win rate against the SFT model with an LLM judge
([@sec:eval:judge]), with position swapping to remove order bias, and a length-controlled
variant that regresses out the effect of response length, since judges favour longer answers. Also
run capability benchmarks (multiple-choice knowledge, math with a verifier) and safety probes
([@ch:alignment]) to detect an *alignment tax*, where preference tuning costs capability.

## DPO on a toy task you can run on a laptop {#sec:dpo:toy}

[@lab:dpo]'s task makes the preferred behaviour *exactly measurable*. A prompt is `Q d1 …dk >`; a
response is a style token followed by the reversed digits and EOS. The reference policy is a tiny
Qwen3-shaped model (two layers, width 64) trained by SFT on demonstrations that use the polite style
`sure` only 25% of the time (otherwise `nope`). Preference pairs come in two kinds: *style* pairs
(70%: same correct digits, `sure` $\succ$ `nope`) and *quality* pairs (30%: both polite, correct
reversal $\succ$ one corrupted digit). Two metrics are computed exactly: $P(\texttt{sure})$, the
probability of the polite first token averaged over prompts, and greedy exact-match accuracy of the
reversal.

![DPO on the toy task, measured on CPU in this environment (one seed; accuracy curves use 50 prompts
per point, so they are noisy). Left: the preferred behaviour appears within 25 steps from the
reference's $0.24$. Middle: the margin grows almost entirely by pushing the *rejected* responses
down, by up to 78 nats for $\beta=0.05$; the chosen log-prob also drifts down. Right: the reversal
skill the pairs do not ask about degrades and fluctuates, most for the smaller $\beta$ values: an
alignment tax.](figures/ch43_dpo-5.svg){#fig:dpo:toycurves}

::: {#tab:dpo:toyresults .table-float}
| **Method (150 steps, lr $10^{-3}$)** | $P(\texttt{sure})$ | **greedy acc.** | **note**                     |
|:-------------------------------------|:------------------:|:---------------:|:-----------------------------|
| reference (SFT)                      |       0.239        |      1.00       | starting point               |
| DPO, $\beta=0.05$                    |       0.998        |      0.85       |                              |
| DPO, $\beta=0.1$                     |       0.981        |      0.52       |                              |
| DPO, $\beta=0.5$                     |       0.947        |      0.95       | stronger anchor              |
| cDPO, $\beta=0.1$, $\varepsilon=0.1$ |       0.428        |      0.58       | finite target margin         |
| IPO, $\tau=0.1$                      |       0.977        |      1.00       | finite target margin         |
| KTO, $\beta=0.1$                     |       0.999        |      0.98       | pairs split into good/bad    |
| SimPO, $\beta=2$, $\gamma=0.5$       |       0.838        |      0.98       | no reference                 |
| ORPO, $\lambda=0.5$                  |       0.999        |      1.00       | no reference, + NLL          |
| CPO, $\beta=0.1$                     |       0.999        |      0.98       | no reference, + NLL          |
| LoRA-DPO, $r=4$, lr $5\times10^{-3}$ |       1.000        |      0.48       | 8516 of 84804 params trained |

: Final metrics on the toy task (measured on CPU, one seed; accuracy on 100 prompts). Every method
moves the style; methods with an NLL term (ORPO, CPO) or a bounded margin (IPO) keep the reversal
skill best in this run, while plain DPO with small $\beta$ and an aggressive learning rate pays the
largest alignment tax.
:::

![Per-token log-ratios $\log\pi_\theta(y_t\mid\cdot)-\log\pi_{\text{ref}}(y_t\mid\cdot)$ after 150
DPO steps ($\beta=0.1$) on the toy task, for one prompt (measured on CPU in this environment;
implicit reward $=\beta\times$ row sum). The winner gains only at the style token
($+1.39\approx\ln(0.99/0.25)$). In the style loser, the model learned more than "don't say `nope`":
every digit *after* `nope` became 15--18 nats less likely, mass that went elsewhere. In the quality
loser, the corrupted digit `9` is penalised, and so is the token after
it.](figures/ch43_dpo-6.svg){#fig:dpo:tokens}

[@fig:dpo:toycurves; @tab:dpo:toyresults; @fig:dpo:tokens] show the three behaviours
you will see at scale. (1) DPO moves exactly the behaviour the pairs encode: $P(\texttt{sure})$ goes
from the reference's $0.24$ to $0.95$--$1.0$ within a few dozen steps. (2) The margin grows mostly
by pushing the *rejected* responses down (tens of nats), while the chosen log-prob barely moves or
drifts down. [@fig:dpo:tokens] shows where: the only token whose probability rises is the
style token, and the digits that follow `nope` are suppressed far more than the style decision alone
requires. (3) Capability that the pairs do not mention can regress. The reference reverses digits
perfectly, but plain DPO at $\beta=0.1$ and LoRA-DPO with a large learning rate end at about 50%
greedy accuracy, while the variants with an NLL term (ORPO, CPO) or a bounded margin (IPO) keep it.

cDPO's row is instructive. Its loss is minimised at a finite margin ($\log 9$ in reward units, or 22
nats of log-ratio for $\beta=0.1$), and the cheapest way to reach that margin is to push the losers
down further, *not* to make `sure` more likely. The objective is satisfied while the behaviour we
care about only moved from $0.24$ to $0.43$. The implicit-reward accuracy and margin alone would not
have shown this, which is why you should always measure the behaviour itself. With one seed and a
toy model, treat the ranking of methods in [@tab:dpo:toyresults] as an illustration of
mechanisms, not as a benchmark.

## Failure modes {#sec:dpo:failures}

##### Likelihood displacement.

[@eq:dpo:grad] only constrains the *difference* of log-ratios, so both log-probabilities
may fall while the margin grows. Probability mass freed from both responses goes *somewhere else*:
sometimes to paraphrases of the winner (harmless), sometimes to responses unlike either (harmful).
Displacement is worse when $y_w$ and $y_l$ are similar (they share most tokens, so pushing $y_l$
down drags $y_w$ along); Razin et al. (2024) analyse when this happens. Mitigations: an auxiliary
SFT/NLL term on the chosen response (as in iterative RPO, Pang et al., 2024, and in CPO), larger
$\beta$, fewer epochs, and on-policy pairs.

##### Overfitting deterministic preferences.

If a pair is always labelled the same way, the BT-optimal margin is infinite; DPO keeps pushing
$h\to\infty$, so the implicit reward and the policy can move arbitrarily far from $\pi_{\text{ref}}$
despite the KL term. The KL regularisation in ([-@eq:dpo:rlobj]) acts through the *optimal
policy*, but with finite data the empirical optimum is not regularised. IPO
([@sec:dpo:family]) was designed for exactly this.

##### Length exploitation.

If chosen responses are longer on average, DPO learns "longer is better". A summed log-prob implicit
reward also scales with length. Watch the mean generation length per checkpoint; use
length-controlled evaluation; consider length-normalised variants (SimPO), a length penalty in the
margin (R-DPO, Park et al., 2024), or length-balanced data.

##### Off-policy data.

DPO's gradient only touches the responses in the dataset. If those were written by a different
(stronger) model, the policy is pushed towards sequences it would rarely sample itself, and what it
actually generates is never directly corrected. Generating pairs from $\pi_{\text{ref}}$ (or
iteratively from the current policy; *online DPO*) closes this gap at the cost of reintroducing a
generation engine.

##### A mis-specified reference.

DPO anchors to whatever $\pi_{\text{ref}}$ is. A reference that is not the SFT model the data was
collected around (wrong checkpoint, wrong template) makes the KL anchor meaningless. This is why the
cache metadata matters.

::: systemslens
[Why DPO is easy to scale, and where it isn't]{.box-title}

A DPO step is SFT on $2\times$ the sequences plus (optionally) a no-grad forward of the reference.
Everything from [@ch:pretrain; @ch:distributed; @ch:sharding] applies unchanged: FSDP,
activation checkpointing, mixed precision, sequence packing. There is no generation, no weight sync
to an inference engine, and no long-tailed rollouts ([@ch:rlsys]), so throughput is
predictable and GPU utilisation can approach SFT levels. The price is off-policy data. The moment
you go online (sample new pairs every round), you inherit the RL-systems problems of
[@ch:rlsys], but in a milder, batched form: generate a whole dataset, label it, train,
repeat.
:::

## The family: cDPO, IPO, KTO, SimPO, ORPO, CPO and online variants {#sec:dpo:family}

DPO kicked off a large literature of "direct alignment" losses. Nearly all of them are functions of
the same few per-pair quantities: summed or averaged log-probs of the winner and loser under the
policy, optionally minus the same under a reference. [@fig:dpo:losses] plots several as
functions of the margin, and [@tab:dpo:family] summarises them.

##### Conservative DPO (cDPO) and robust variants.

If a fraction $\varepsilon$ of labels are flipped, model the label as noisy:
$\mathcal{L}=-(1-\varepsilon)\log\sigma(h)-\varepsilon\log\sigma(-h)$. The optimum is now at a
finite margin, $h^*=\log\frac{1-\varepsilon}{\varepsilon}$, which also tames overfitting.

##### IPO.

Azar et al. (2023) cast RLHF and DPO as instances of a general objective ($\Psi$PO) and showed that
DPO's use of the BT logit is what makes deterministic preferences blow up. *Identity preference
optimisation* replaces the logistic loss with a squared regression of the log-ratio margin onto a
finite target: $$\begin{equation}
\mathcal{L}_{\text{IPO}}=\mathbb{E}\Big[\Big(\log\frac{\pi_\theta(y_w\mid x)\,\pi_{\text{ref}}(y_l\mid x)}
{\pi_\theta(y_l\mid x)\,\pi_{\text{ref}}(y_w\mid x)}-\frac{1}{2\tau}\Big)^2\Big].
\label{eq:dpo:ipo}
\end{equation}$$ The margin is pulled *towards* $1/(2\tau)$ and penalised for overshooting it.
$\tau$ plays $\beta$'s role. Many implementations feed length-averaged log-probs into
([-@eq:dpo:ipo]).

##### KTO.

Ethayarajh et al. (2024) observe that paired comparisons are expensive to collect, while binary
"this response is good/bad" signals (thumbs up/down) are cheap. Drawing on prospect theory, KTO
assigns each unpaired example a value relative to a *reference point* $z_0$: $$\begin{align}
r_\theta(x,y)&=\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)},\qquad
z_0=\max\!\big(0,\ \widehat{\operatorname{KL}}(\pi_\theta\,\|\,\pi_{\text{ref}})\big)\ \text{(detached)},\\
v(x,y)&=\begin{cases}\lambda_D\,\sigma\big(\beta(r_\theta(x,y)-z_0)\big) & y\ \text{desirable}\\
\lambda_U\,\sigma\big(\beta(z_0-r_\theta(x,y))\big) & y\ \text{undesirable}\end{cases},\qquad
\mathcal{L}_{\text{KTO}}=\mathbb{E}\big[\lambda_y-v(x,y)\big].
\end{align}$$ The logistic value function is concave for gains and convex for losses, like Kahneman
and Tversky's ([@fig:dpo:kto]). $\lambda_D,\lambda_U$ rebalance classes (setting
$\lambda_U>\lambda_D$ encodes loss aversion). $z_0$ is estimated per batch, without sampling, by
scoring *mismatched* pairs (prompt $i$ with the response of example $i{+}1$) under policy and
reference. Ethayarajh et al. frame DPO, KTO and others as *human-aware losses* (HALOs). From a
systems view, KTO's per-example (rather than per-pair) batches pack better and allow unbalanced
data.

##### SimPO.

Meng et al. (2024) drop the reference model and use the *length-normalised* log-probability as the
reward, plus a target margin $\gamma$:
$\mathcal{L}=-\log\sigma\big(\frac{\beta}{|y_w|}\log\pi_\theta(y_w\mid x)-\frac{\beta}{|y_l|}\log\pi_\theta(y_l\mid x)-\gamma\big)$.
No reference means no cache and no second model, and the length normalisation counters length
exploitation. The cost: no explicit anchor to $\pi_{\text{ref}}$, so the learning rate and step
count carry the regularisation.

##### ORPO.

Hong et al. (2024) fold SFT and preference learning into one stage with no reference model: the loss
is the SFT negative log-likelihood of $y_w$ plus
$-\lambda\log\sigma\big(\log\frac{\text{odds}_\theta(y_w)}{\text{odds}_\theta(y_l)}\big)$, where
$\text{odds}(y)=p/(1-p)$ with $p$ the length-normalised sequence probability. Starting from a base
model, one ORPO run replaces SFT followed by DPO.

##### CPO.

Contrastive preference optimisation (Xu et al., 2024), proposed for machine translation, uses a
reference-free DPO-like term $-\log\sigma(\beta(\log\pi_\theta(y_w)-\log\pi_\theta(y_l)))$ plus an
NLL term on $y_w$. Like ORPO and SimPO, it trades the KL anchor for simplicity.

##### Earlier and adjacent methods.

SLiC-HF (Zhao et al., 2023) uses a hinge loss on the log-probability margin plus a regulariser
towards SFT. RSO (Liu et al., 2023) argues for sampling pairs from (an estimate of) the optimal
policy via statistical rejection sampling. sDPO (Kim et al., 2024) splits the data into stages and
uses the previous stage's model as the next reference. Anchored preference optimisation (APO,
D'Oosterlinck et al., 2024) controls whether the winner's likelihood should go up or down
explicitly.

##### Online and iterative DPO.

Generate $k$ responses per prompt from the *current* policy, label them (reward model, verifier or
judge), form pairs, run a DPO epoch, and repeat. This recovers much of the benefit of on-policy RL
at a fraction of the systems complexity. Self-rewarding LMs (Yuan et al., 2024) use the model itself
as the judge. SPIN (Chen et al., 2024) treats SFT data as winners against the model's own
generations. Nash learning from human feedback (Munos et al., 2023) treats alignment as a game
against a preference model rather than a reward.

![Direct-alignment losses as functions of the margin (analytic; IPO scaled by 1/4 to share the
axis). DPO never stops rewarding a larger margin; cDPO's minimum is at a finite margin
($\log 9\approx2.2$ for $\varepsilon=0.1$); IPO penalises overshooting its target; SimPO shifts
DPO's curve by the target margin $\gamma$; the hinge stops at margin
1.](figures/ch43_dpo-7.svg){#fig:dpo:losses}

![The shape of KTO's value function (analytic, $\beta=0.6$): a logistic stand-in for
Kahneman--Tversky's value curve, centred at the reference point $z_0$. Gains (desirable outputs
above the reference point) saturate; losses are weighted by
$\lambda_U$.](figures/ch43_dpo-8.svg){#fig:dpo:kto}

::: {#tab:dpo:family .table-float}
| **Method** | **Ref.** | **Data** | **Per-example signal** | **Extra knobs** |
|:---|:--:|:--:|:---|:---|
| DPO | yes | pairs | $\beta(\Delta\log\pi_\theta-\Delta\log\pi_{\text{ref}})$, logistic | $\beta$ |
| cDPO | yes | pairs | as DPO with label noise $\varepsilon$ | $\beta,\varepsilon$ |
| IPO | yes | pairs | squared error to margin $1/(2\tau)$ | $\tau$ |
| KTO | yes | unpaired | $\sigma(\pm\beta(r-z_0))$, asymmetric weights | $\beta,\lambda_D,\lambda_U$ |
| SimPO | no | pairs | length-normalised log-prob difference, margin $\gamma$ | $\beta,\gamma$ |
| ORPO | no | pairs | NLL$(y_w)$ + log odds-ratio term | $\lambda$ |
| CPO | no | pairs | ref-free DPO term + NLL$(y_w)$ | $\beta$ |
| SLiC-HF | no | pairs | hinge on log-prob margin + SFT regulariser | margin, weight |
| online DPO | yes | pairs (fresh) | DPO on pairs sampled from current policy | sampler, labeller |

: The direct-alignment family at a glance. "Ref." is whether a reference model (or cache) is needed.
All except online DPO train on a fixed dataset.
:::

``` {#lst:dpo:family .python caption="Four relatives in a few lines each (from `llmsys/prefopt.py`). Sums are over response tokens; `len_*` are token counts."}
def ipo_loss(pc, pr, rc, rr, tau=0.1):
    h = (pc - rc) - (pr - rr)
    return (h - 1.0 / (2.0 * tau)) ** 2

def kto_loss(pol, ref, desirable, z0, beta=0.1, lam_d=1.0, lam_u=1.0):
    r, z0 = pol - ref, z0.detach().clamp(min=0)
    v = torch.where(desirable, lam_d * torch.sigmoid(beta * (r - z0)),
                    lam_u * torch.sigmoid(beta * (z0 - r)))
    lam = torch.where(desirable, torch.full_like(r, lam_d),
                      torch.full_like(r, lam_u))
    return lam - v

def simpo_loss(pc_sum, pr_sum, len_c, len_r, beta=2.0, gamma=0.5):
    h = beta * pc_sum / len_c - beta * pr_sum / len_r
    return -F.logsigmoid(h - gamma)

def orpo_loss(pc_sum, pr_sum, len_c, len_r, lam=0.1):
    a_c, a_r = pc_sum / len_c, pr_sum / len_r          # mean log-probs < 0
    log_odds = lambda a: a - torch.log(-torch.expm1(a))  # log p/(1-p)
    return -a_c - lam * F.logsigmoid(log_odds(a_c) - log_odds(a_r))
```

##### One trainer for every loss.

Because all these losses consume the same per-pair quantities, one training loop serves them all;
only the reduction differs. [@lst:dpo:dispatch] shows the dispatch in `llmsys.prefopt`. KTO
is the only one that needs an extra forward, over the mismatched pairs used to estimate its
reference point. The reference cache gets a third column for it, so the reference model is still
never loaded during training.

``` {#lst:dpo:dispatch .python caption="Dispatching on the loss type in one training step (`preference_step_loss`, abridged)."}
lp, n = sequence_logps(model, b["input_ids"], b["loss_mask"])
pc, pr = split_pair_logps(lp); nc, nr = split_pair_logps(n)
if cfg.needs_reference:                       # dpo, ipo, kto
    rc, rr = ref[:, 0], ref[:, 1]
if cfg.loss == "dpo":
    loss, a, c = dpo_loss(pc, pr, rc, rr, cfg.beta, cfg.label_smoothing)
elif cfg.loss == "ipo":
    loss, a, c = ipo_loss(pc, pr, rc, rr, cfg.ipo_tau)
elif cfg.loss == "simpo":
    loss, a, c = simpo_loss(pc, pr, nc, nr, cfg.beta, cfg.simpo_gamma)
elif cfg.loss == "orpo":
    loss, a, c = orpo_loss(pc, pr, nc, nr, cfg.orpo_lambda)
elif cfg.loss == "kto":                       # winners desirable, losers not
    mism = [mismatched_pair(all_pairs, i) for i in idx]   # prompt i, y_w of i+1
    mb = build_pair_batch(mism, pad_id)
    mlp, _ = sequence_logps(model, mb["input_ids"], mb["loss_mask"])
    z0 = kto_kl_estimate(split_pair_logps(mlp)[0], ref[:, 2])
    ...
metrics = preference_metrics(a, c)
```

## DPO vs PPO vs GRPO {#sec:dpo:compare}

::: {#tab:dpo:compare .table-float}
|  | **PPO-RLHF** ([@ch:rlhf]) | **DPO** (this chapter) | **GRPO** ([@ch:grpo; @ch:grpo2]) |
|:---|:---|:---|:---|
| data | prompts + reward model | fixed preference pairs | prompts + verifier / reward |
| on/off-policy | on-policy (minibatch reuse) | offline | on-policy (near) |
| reward model | explicit, trained first | implicit $\beta\log\pi_\theta/\pi_{\text{ref}}$ | rule/verifier or RM |
| critic | value network | none | none (group baseline) |
| models in memory | policy, ref, RM, value | policy (+ ref or cache) | policy, ref (optional) |
| generation in loop | yes | no | yes, $G$ samples/prompt |
| step cost | rollouts + 4 forwards + 2 backwards | 2 seq. fwd+bwd (+ref fwd) | rollouts + fwd/bwd on $G$ seqs |
| main knobs | clip, GAE $\lambda$, KL coef., value coef. | $\beta$, lr, epochs | $G$, clip, KL coef., lengths |
| strengths | explores, optimises a real reward | simple, stable, cheap | no critic, verifiable rewards |
| weaknesses | complex, expensive, brittle | off-policy, overfits, length bias | needs verifiable reward, rollout cost |

: The three post-training workhorses compared. The rows are characteristics of the methods as
usually run; each has online/offline hybrids.
:::

[@tab:dpo:compare] compares the three methods. Which works best is an empirical,
setting-dependent question. Some careful comparisons (e.g. Xu et al., 2024, "Is DPO superior to PPO
for LLM alignment?") report that a well-tuned PPO can outperform DPO, especially on tasks such as
code where on-policy exploration and an accurate reward matter, and that DPO is sensitive to the
distribution shift between the data and the policy. Others find DPO competitive at much lower cost
for chat-style preference data. A practical reading for a systems person:

- If you have a **fixed pairwise dataset** and want a dependable improvement cheaply, start with DPO
  (or SimPO/ORPO if a reference is inconvenient).

- If you have a **verifiable reward** (math, code, tests), the on-policy, critic-free methods of
  [@ch:grpo] are the natural fit.

- If you have a **learned reward model** and the budget for an RL system, PPO or online DPO can
  exploit it, with the reward-hacking caveats of [@ch:rlhf].

- Many production post-training pipelines use *several*: SFT, then DPO-style preference tuning, then
  RL with verifiable rewards, sometimes iterated.

## Systems: what a DPO step costs {#sec:dpo:systems}

::: napkinmath
[FLOPs per pair and per epoch]{.box-title}

Take Qwen3-8B ($N\approx8.19\times10^9$ parameters), prompts of 256 tokens and responses of 256
tokens each. A pair is two sequences of 512 tokens: 1,024 tokens. The policy costs about $6N$ FLOPs
per token (forward + backward), the reference $2N$ (forward only): $$\begin{align*}
\text{policy: }& 6\cdot8.19\times10^9\cdot1024\approx5.0\times10^{13}\ \text{FLOPs/pair},\\
\text{reference: }& 2\cdot8.19\times10^9\cdot1024\approx1.7\times10^{13}\ \text{FLOPs/pair}.
\end{align*}$$ For 160k pairs (about the size of a large public chat preference set) that is
$1.07\times10^{19}$ FLOPs per epoch, or about **7.5 H100-hours** at 40% MFU
($0.4\times989$ TFLOP/s). The reference forward is a quarter of the total: caching it moves
$2.7\times10^{18}$ FLOPs into a one-off precomputation that can run on inference-optimised hardware,
and it is saved outright for every additional epoch.
:::

::: napkinmath
[Memory: full DPO vs LoRA-DPO for 8B]{.box-title}

Full fine-tuning with AdamW and fp32 master weights needs $\approx16$ bytes/param
([@ch:pretrain]): $16\times8.19\times10^9\approx131$ GB of weights, gradients and optimizer
state, plus activations for $2B$ sequences. A bf16 reference adds 16.4 GB. This does not fit on one
80 GB GPU: shard the policy with FSDP/ZeRO-3 ([@ch:sharding]) over 8 GPUs ($\approx16$ GB of
states per GPU) and cache the reference.

LoRA-DPO with rank 16 on every linear layer of Qwen3-8B trains about $44$M parameters: a frozen bf16
base of 16.4 GB plus $16\times4.4\times10^7\approx0.7$ GB of adapter states. The reference costs
nothing extra, because it *is* the base with adapters disabled. Activations then dominate: use
activation checkpointing ([@ch:perf]) and keep the logits small (next box).
:::

![Analytic memory for DPO on an 8B model (16 bytes/param for full training, bf16 frozen weights).
Caching the reference removes 16 GB; FSDP divides policy state across GPUs; LoRA-DPO needs only the
frozen base plus tiny adapter states, with the reference for
free.](figures/ch43_dpo-9.svg){#fig:dpo:memory}

::: systemslens
[Logits are the hidden activation hog]{.box-title}

The output layer produces $(2B,T,V)$ logits. For $B=8$ pairs, $T=1024$ and Qwen's $V=151{,}936$ in
fp32 that is $2\cdot8\cdot1024\cdot151936\cdot4\approx10$ GB, before the log-softmax copy. Three
standard fixes: (1) compute log-probs with `gather` minus `logsumexp` ([@lst:dpo:logps])
and never materialise `log_softmax`; (2) slice to response positions *before* the LM head when
prompts are long; (3) chunk the LM head and log-prob computation over the sequence (the same trick
as chunked cross-entropy in [@ch:distill]). Fused "linear + cross-entropy" kernels go further
by never writing the logits to HBM.
:::

::: systemslens
[Share the prompt: one prefix, two continuations]{.box-title}

Winner and loser share the prompt, yet the pair batch processes it twice. With $T_p=256$,
$T_w=T_l=256$, a pair costs $2T_p+T_w+T_l=1024$ tokens; computing the prompt once costs
$T_p+T_w+T_l=768$, saving **25%** of FLOPs and activation memory, and much more for long multi-turn
prompts. Two implementations: (a) pack prompt, $y_w$ and $y_l$ into one sequence with a custom mask
in which $y_l$ attends to the prompt but not to $y_w$ (a two-branch tree mask, as in tree-attention
speculative verification, [@ch:efficient]), with $y_l$'s positions restarting after the
prompt; or (b) prefill the prompt once and run both continuations against its KV cache
([@ch:serving]), which is natural at inference time and possible at training time with
attention kernels that accept a shared prefix. Either way, verify against the naive pair batch on a
few examples: the log-probs must agree to numerical precision.
:::

::: deepdive
[Padding, bucketing and packing pairs]{.box-title}

Pairs have wildly different lengths, so right-padding a random batch to its longest row wastes FLOPs
([@ch:classify]). Bucket pairs by $\max(|x|+|y_w|,|x|+|y_l|)$, or pack several pairs into
fixed-length rows with block-diagonal attention masks and per-segment position ids
([@sec:instruct:packing]). Packing requires keeping track of which segment is the winner
and which the loser of which pair, which is just bookkeeping on the `loss_mask`.
:::

##### Placement recipes.

\(1\) *Single GPU, small model*: policy and reference both on the GPU, or LoRA-DPO. (2) *Two GPUs*:
policy on GPU 0, reference on GPU 1; compute the reference log-probs for the next micro-batch while
the policy runs backward on the current one. (3) *Many GPUs*: FSDP-shard the policy and use a
precomputed cache built by an inference job beforehand. (4) *Online DPO*: add the generation engine
of [@ch:serving] or [@ch:rlsys] on separate GPUs; generation is batched per round, so
the simpler synchronous design of [@ch:rlsys] is usually adequate.

## Summary {#sec:dpo:summary}

- The KL-regularised RLHF objective has a closed-form optimum
  $\pi^*\propto\pi_{\text{ref}}e^{r/\beta}$. Inverting it writes the reward as
  $\beta\log\pi/\pi_{\text{ref}}$ up to a prompt constant that cancels in Bradley--Terry
  comparisons. Substituting gives the DPO loss, a logistic loss on the difference of log-ratios.

- The DPO gradient is likelihood on winners and unlikelihood on losers, weighted by how mis-ranked
  each pair is under the implicit reward.

- Implementation is SFT-like: a $(2B,T)$ pair batch with a response mask, summed response log-probs,
  a reference that is cached or (with LoRA) the frozen base, and the sanity check that the step-0
  loss equals $\ln 2$.

- Watch reward accuracy, margins, *and* the raw log-probs of both responses: the chosen log-prob
  often falls (likelihood displacement). Over-optimising fixed data costs capability.

- Variants trade the reference model, pairing and length handling: cDPO and IPO bound the margin,
  KTO uses unpaired good/bad labels, SimPO, ORPO and CPO drop the reference, and online DPO restores
  on-policy data.

- DPO costs $\approx(6N+2N)$ FLOPs per token of the pair, has no generation in the loop, and scales
  with the same tools as pretraining. Logit memory and duplicated prompts are the main
  inefficiencies to engineer away.

## Exercises {#sec:dpo:exercises}

::: exercise
[Gibbs policy, continuous check]{.box-title}

For a toy "response space" of three outcomes with $\pi_{\text{ref}}=(0.5,0.3,0.2)$ and $r=(0,1,2)$,
compute $\pi^*$ for $\beta\in\{0.1,1,10\}$ and verify numerically that $\pi^*$ maximises
$\mathbb{E}_\pi[r]-\beta\operatorname{KL}(\pi\|\pi_{\text{ref}})$ by evaluating the objective at
random distributions.
:::

::: exercise
[Z cancels]{.box-title}

Show that adding any function $f(x)$ to the reward leaves the BT likelihood unchanged and does not
change $\pi^*$ in ([-@eq:dpo:gibbs]). What does this imply about comparing implicit rewards
across prompts?
:::

::: exercise
[The gradient]{.box-title}

Derive ([-@eq:dpo:grad]) from ([-@eq:dpo:loss]). Then show that at initialisation
($\pi_\theta=\pi_{\text{ref}}$) the DPO gradient equals $\beta/2$ times the gradient of
$\log\pi_\theta(y_w\mid x)-\log\pi_\theta(y_l\mid x)$.
:::

::: exercise
[cDPO's optimum]{.box-title}

For a single pair, find the margin $h^*$ that minimises
$-(1-\varepsilon)\log\sigma(h)-\varepsilon\log\sigma(-h)$, and the minimum value. Interpret
$\varepsilon\to0$.
:::

::: exercise
[IPO vs DPO on deterministic data]{.box-title}

Consider one prompt with two responses where $y_w$ is *always* preferred. Parameterise the policy by
a single logit $z$, with $\pi(y_w)=\sigma(z)$ and $\pi_{\text{ref}}$ uniform. Write the DPO and IPO
losses as functions of $z$ and find their minimisers. Which one keeps $\pi$ away from the
deterministic policy?
:::

::: exercise
[Sum vs mean]{.box-title}

Show that replacing summed with mean log-probs in DPO is equivalent to DPO with a per-response
$\beta_y=\beta/|y|$. For a winner twice as long as the loser, which side does this favour, compared
with standard DPO?
:::

::: exercise
[Step-0 sanity check, by hand]{.box-title}

In [@lab:dpo], deliberately build the reference cache with a different truncation length than training
uses. Predict, then measure, the step-0 loss and reward accuracy. Repeat with dropout left on in the
policy forward. Write a three-line assertion block for your training script that catches both bugs.
:::

::: exercise
[Prompt sharing]{.box-title}

Implement the packed "one prompt, two continuations" layout of the second Systems Lens for the toy
model (Qwen3 uses RoPE, so give $y_l$ positions starting right after the prompt). Verify that the
log-probs match `sequence_logps` on the pair batch, and measure tokens processed per pair for prompt
lengths 4, 16, 64.
:::

::: exercise
[Logit memory]{.box-title}

For Qwen3-0.6B ($V=151{,}936$), compute the fp32 logit memory for a DPO micro-batch of $B=4$ pairs
at $T=2048$. How much is saved by (a) slicing to the response half when $T_p=T_y$, (b) chunking the
LM head over 8 sequence chunks, (c) bf16 logits with fp32 `logsumexp`?
:::

::: exercise
[Online DPO loop design]{.box-title}

Design a 3-round online DPO pipeline for a 7B model on 8 GPUs with a reward model of 1B parameters:
per round, generate 4 samples for each of 20k prompts, score them, form (best, worst) pairs, train
one epoch. Estimate generation time with the decode roofline of [@ch:inference], training
time with the napkin math above, and decide whether generation and training should share GPUs.
:::

## Lab 43: Preference optimisation from the data path up {#sec:dpo:lab}

::: labbox
[Preference optimisation from the data path up]{.box-title}

**Code:** `code/labs/lab43_dpo`

**Goals**

Implement DPO and five relatives on top of a correct pair-batch data path. Show on a laptop CPU that
preference optimisation moves a measurable behaviour, and observe likelihood displacement and the
alignment tax yourself.

**Setup**

``` bash
cd code
LAB_IMPL=starter  python -m pytest labs/lab43_dpo     # your work
LAB_IMPL=solution python -m pytest labs/lab43_dpo     # reference
python labs/lab43_dpo/run.py --out results            # full experiment, minutes on CPU
```

**Tasks**

1.  `build_pair_batch`: the $(2B,T)$ layout of [@fig:dpo:batch] with a response mask.

2.  `sequence_logps`: shifted, masked sums using `gather` $-$ `logsumexp`; also return token counts.

3.  `dpo_loss` with conservative label smoothing; return detached implicit rewards.

4.  `ipo_loss`, `kto_loss`, `simpo_loss`, `orpo_loss`.

5.  `precompute_reference_logps` and `preference_metrics` (ties count one half).

6.  `dpo_train_loop`: shuffled micro-batches, gradient accumulation, clipping, per-step metrics.

The tests check exact values, gradient directions, the step-0 $\ln 2$ property, that the cache
matches on-the-fly log-probs, that disabling LoRA reproduces the base model, and that 25 DPO steps
raise $P(\texttt{sure})$ substantially.

**Measurements**

- Run `run.py`: plot $P(\texttt{sure})$, greedy accuracy and the two log-probs against steps for
  $\beta\in\{0.05,0.1,0.5\}$. Expect faster behaviour change and larger log-prob drift for small
  $\beta$.

- Compare final $P(\texttt{sure})$, accuracy and wall-clock for DPO, cDPO, IPO, KTO, SimPO, ORPO,
  CPO and LoRA-DPO. Which reference-free methods damage the reversal skill, and why?

- Time the reference cache against one training epoch; relate the ratio to the $2N$ vs $6N$ FLOP
  model.

- On a GPU, repeat with Qwen3-0.6B and a public preference set (`load_preference_jsonl` +
  `tokenize_pair`); log peak memory with full DPO + reference model, full DPO + cache, and LoRA-DPO.

**Deliverables**

Passing tests; the plots; a table of final metrics per method; one paragraph explaining likelihood
displacement in your run using ([-@eq:dpo:grad]).

**Stretch goals**

Add an NLL term on the chosen response (RPO-style) and show it reduces displacement; implement
online DPO on the toy task (sample from the current policy, label with the exact rule, retrain);
implement the shared-prefix packed layout and measure the savings.
:::
