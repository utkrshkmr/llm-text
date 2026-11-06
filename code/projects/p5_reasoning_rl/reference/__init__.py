"""Project 5 reference solution (Chapter 66): post-training at matched compute.

    methods     -- the loss and advantage maths of every method, written out
    budget      -- GPU-hour accounting for the matched-compute protocol
    staleness   -- rollout versioning, the staleness filter, the asynchronous rollout worker
    placement   -- colocated vs. split trainer/rollout placement and weight sync
    harness     -- reproducible evaluation (toy task, MATH-500/GSM8K) and the reward-hacking audit
    tasks       -- the toy task (llmsys.rl.toy) and the Qwen3 + MATH task behind one interface
    train       -- the single driver: --method {sft,grpo,dapo,ppo,dpo,distill}

Attempt the project before reading this package; the acceptance tests in
``test_p5_acceptance.py`` run against it by default and against your own code via
``P5_IMPL=/path/to/your/package``.
"""
