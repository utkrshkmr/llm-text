# Brief: Project 2, training-systems performance on PCIe GPUs (chapter 43, kit p2_systems)
Read projects_common.md first, then docs/PLAN.md §4 "P2", the existing chapter and kit (keep acceptance_p2.py working),
Chapters 17, 18, 19, 20, 24 (book + code), code/llmsys/ops/flash_attn.py and llmsys/ckpt_planner.py if present (else
use the Triton FA2 of llmsys.kernels.flash2_triton and llmsys.perf, and note it), code/llmsys/dist/*. GPUs: both,
short runs only.
Scope: maximise MFU for a 350M-1B model on 2 PCIe A100s: FA2 CUDA custom op, fused kernels + torch.compile, DDP with
bucketing/overlap vs ZeRO-1/2 (llmsys.dist) vs FSDP2 (torch.distributed.fsdp.fully_shard), optimal activation
checkpoint planner, TP=2 as a studied counter-example on PCIe. reference/: the trainer with every strategy behind a
flag, a profiling script (torch.profiler + nsys instructions), the MFU calculator, sweep configs smoke/full (~10 GPU-h).
Acceptance (CPU gloo, tiny): every parallel configuration's parameters after N steps equal single-process training
(DDP, ZeRO-1, ZeRO-2, FSDP, TP=2) within tolerance; checkpoint planner output respects the memory budget; MFU
function values on fixed inputs. GPU-marked: a 20-step bf16 run on 2 GPUs and the profile breakdown file.
