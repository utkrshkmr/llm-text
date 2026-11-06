# Brief: additions to Chapter 42 "RLHF: Reward Models, Value Functions, and PPO"

Read first: BRIEF_conventions.md (all, esp. §7), the existing chapter book/chapters/ch42_rlhf_ppo.md (keep all
content), ch40_rl_setup.md brief, code/llmsys/rlhf/*.

## DeepSeek contributions (owner request, 2026-11-05)
- Generative reward models: DeepSeek-GRM (self-principled critique tuning: the model writes principles and critiques
  before scoring; inference-time scaling of reward quality by sampling several critiques and voting, with a meta
  reward model to filter them), contrasted with scalar Bradley–Terry reward models; cost per reward call and when it
  pays. Implement a toy generative judge with voting and measure agreement vs a scalar RM on the chapter's task.
- Research frontier section and Reading-the-research box (§7) if missing.
