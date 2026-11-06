# Brief: additions to Chapter 63 "RL Training Systems" + Lab 63 extension

Read first: BRIEF_conventions.md (all, esp. §7), ch57_rl_setup.md brief and chapter (the setup this chapter now
assumes), the existing chapter book/chapters/ch63_rl_systems.md (keep all content), code/labs/lab63_rlsys/*.

Additions (owner asked for much more detail on the RL setup):
- A complete reference deployment on 2×A100: placement of policy/reference/reward/critic and the rollout engine
  (colocated vs split), memory budget per GPU, weight-sync path and its bytes/time over PCIe, the timeline of one
  iteration measured with the ch33 tooling, and the config file that produces it.
- Framework anatomy in depth (verl, OpenRLHF, TRL, NeMo-RL, slime, AReaL): how each maps the dataflow of
  [@ch:rlsetup] onto processes and GPUs; asynchronous/off-policy pipelines with staleness bounds and corrections.
- Failure diagnosis playbook: reward flat, KL explosion, entropy collapse, length blow-up, train/inference
  log-prob mismatch (with measured examples), and the metrics that detect each.


## Gap-check additions (owner request, 2026-11-05). Depth rules: BRIEF_conventions.md §7
- Faster rollouts: speculative decoding for RL rollouts (distribution-aware drafting, MLSys 2026), partial rollouts
  and length-bucketed scheduling, heterogeneous clusters for rollout vs training (HetRL), co-scheduling in
  disaggregated RL (OSDI 2026 papers as frontier).
