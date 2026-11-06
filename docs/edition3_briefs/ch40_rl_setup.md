# Brief: Chapter 40 "The RL Setup for Language Models, End to End" (NEW) + Lab 40

Read first: BRIEF_conventions.md (all, esp. §7), docs/COVERAGE.md rows for ch:rlsetup, book/chapters/
ch41_policy_gradients.md (now follows this chapter: keep its derivations there; this chapter sets up the whole
problem and machinery so ch41-46 can focus on algorithms and systems), ch42_rlhf_ppo.md, ch44_grpo.md,
ch46_rl_systems.md, ch37_evaluation.md (verifiers), code/llmsys/rl/*, code/llmsys/rlhf/*, labs 41-46.
GPU: CUDA_VISIBLE_DEVICES=0 (short runs).

Files you own: book/chapters/ch40_rl_setup.md {#ch:rlsetup}, solutions, figures, code/llmsys/rl/env.py (environment
interface: reset/step over text, single-turn and multi-turn, tool-calling environments with a sandboxed Python tool,
verifier-backed rewards, reward-model-backed rewards, episode records), code/llmsys/rl/config.py (one dataclass
describing a complete RL run: models and roles, sampling params, batch geometry, KL/entropy settings, optimizer,
schedules, checkpointing, logging; YAML round-trip; validation of inconsistent settings), code/llmsys/rl/dataflow.py
(the batch as it flows: prompts → rollouts → rewards → reference/old log-probs → advantages → minibatches →
updates; with shapes, masks and dtypes asserted at each hop), code/labs/lab40_rlsetup/*, refs file.

Chapter content:
1. Language generation as a sequential decision problem, precisely: state = prompt + generated prefix, action = next
   token (or a whole turn/tool call at the turn level), policy = the LM, transition deterministic concatenation (plus
   environment observations for tools/multi-turn), episode end (EOS, max length, truncation and its bias), reward
   timing (terminal/outcome vs per-token/process rewards), discounting (γ=1 in practice and why), token-level vs
   sequence-level MDP views; the bandit view used by DPO/GRPO.
2. Sources of reward: rule-based verifiers (math, code execution with tests, format), learned reward models (BT),
   generative/LLM-judge rewards, process reward models, rubric-based rewards; reward hacking surfaces in each.
3. The cast of models and their roles: policy (actor), old policy (behaviour policy for importance ratios), reference
   policy (KL anchor), critic/value model, reward model; which algorithms need which (table: REINFORCE, RLOO, PPO,
   GRPO, DPO, online DPO); memory footprint of each configuration for a 1.5B/7B model.
4. The data flow of one iteration with exact tensor shapes: prompt sampling, rollout generation with an inference
   engine (sampling params, n samples per prompt, stop criteria, max tokens), reward computation, log-prob passes,
   advantage computation, packing/minibatching, optimizer steps, weight sync; off-policyness and staleness.
5. Environments: single-turn verifiable tasks, multi-turn dialogue, tool use (code interpreter, search) with
   observation tokens masked from the loss, agentic environments with milestones; environment API design and
   sandboxing; datasets and curricula; difficulty filtering.
6. Configuring a run: every knob explained with defaults from open recipes (learning rate, batch/group size, KL
   coefficient and estimator, entropy, clip ranges, max length, temperature, number of epochs per batch), what to
   log, sanity checks before scaling (reward of the initial policy, KL at step 0 = 0, ratio = 1 on-policy), a
   worked end-to-end setup on a small model that the following chapters reuse.
7. Research frontier (§7): long-horizon agentic RL, process rewards, verifier quality, RL at scale (asynchronous,
   off-policy corrections), exploration in LLM RL; Reading-the-research box (e.g. DeepSeek-R1 or the Tülu 3 RLVR
   recipe, read for its setup rather than its results).

Lab 40 (CPU tests): text environments (single-turn verifier, multi-turn, tool with sandbox and observation masking;
tests on deterministic toy policies), run config validation and YAML round-trip, dataflow shape/mask assertions on a
tiny model with a fake engine, memory-footprint calculator for role configurations; run.py: one full iteration on a
tiny model on CPU and a short GPU run. Verification per protocol.
