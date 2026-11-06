# Brief: Project 5, post-training at matched compute (chapter 46, kit p5_reasoning_rl)
Read projects_common.md first, then docs/PLAN.md §4 "P5", the existing chapter and kit (keep env_check, run_matrix,
seeds_ci and tests), Chapters 40, 54, 58-64 (book + code: llmsys.rl, rlhf, prefopt, distill, rlsys, eval). GPU:
CUDA_VISIBLE_DEVICES=1, short runs only; no checkpoint downloads.
Scope: SFT a 0.6-1.7B Qwen3, then at matched GPU-hours compare GRPO and DAPO-style variants, PPO with a learned value
function, DPO on preference data, and a distillation baseline; rollout engine (llmsys.serving) and trainer on
separate GPUs or colocated; an asynchronous variant with staleness control; reward-hacking audit. reference/: one
train.py with --method {sft,grpo,dapo,ppo,dpo,distill}, --placement {colocated,split}, --async-staleness K, the
evaluation harness on MATH-500/GSM8K (llmsys.eval), budgeting by GPU-hours, configs smoke/full (~30 GPU-h).
Acceptance (CPU, tiny GPT + toy verifiable task from llmsys.rl.toy): loss and advantage computations equal
llmsys.rl/rlhf/prefopt references on fixed inputs for every method; the harness is reproducible (same seed -> same
accuracy); staleness control rejects rollouts older than K; the matched-compute accounting sums to budget; a
20-step tiny GRPO run improves the toy reward over the SFT start.
