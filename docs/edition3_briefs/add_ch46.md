# Brief: additions to Chapter 46 "RL Training Systems" + Lab 46 extension

Read first: BRIEF_conventions.md (all, esp. §7), ch40_rl_setup.md brief and chapter (the setup this chapter now
assumes), the existing chapter book/chapters/ch46_rl_systems.md (keep all content), code/labs/lab46_rlsys/*.

Additions (owner asked for much more detail on the RL setup):
- A complete reference deployment on 2×A100: placement of policy/reference/reward/critic and the rollout engine
  (colocated vs split), memory budget per GPU, weight-sync path and its bytes/time over PCIe, the timeline of one
  iteration measured with the ch25 tooling, and the config file that produces it.
- Framework anatomy in depth (verl, OpenRLHF, TRL, NeMo-RL, slime, AReaL): how each maps the dataflow of
  [@ch:rlsetup] onto processes and GPUs; asynchronous/off-policy pipelines with staleness bounds and corrections.
- Failure diagnosis playbook: reward flat, KL explosion, entropy collapse, length blow-up, train/inference
  log-prob mismatch (with measured examples), and the metrics that detect each.
