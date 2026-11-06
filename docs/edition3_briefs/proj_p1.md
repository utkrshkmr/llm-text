# Brief: Project 1, a transformer LM from scratch in Python + C++ + CUDA (chapter 42, kit p1_lm)
Read projects_common.md first, then docs/PLAN.md §4 "P1", the existing chapter and kit, Chapters 4, 6, 7, 13, 14, 21, 23
(book + code), code/llmsys/ops/{bpe,rmsnorm,silu_mul,xent}.py (xent may not exist yet: then use a Python chunked CE
and note it). GPU: CUDA_VISIBLE_DEVICES=0, short runs only.
Scope: own byte-level BPE (C++ trainer/encoder via llmsys.ops.bpe, Python fallback), 125-350M pre-norm decoder, own
AdamW + schedules, fused RMSNorm/SwiGLU/cross-entropy via llmsys.ops, memory-mapped uint16 token loader, resumable
bit-exact checkpoints, experiment tracking (reuse llmsys.explog). Research questions and acceptance exactly as in
PLAN §4 P1. reference/ = full training stack with train.py, prepare_data.py (FineWeb/OpenWebText sample via
`datasets`, plus --synthetic), configs smoke/full (full: 3-5B tokens, ~20 GPU-h on 2x A100 PCIe with DDP).
Acceptance tests (CPU, tiny): tokenizer round-trip and equality with llmsys.tokenizer; every custom op matches its
reference and passes opcheck (skip if not built); 50-step tiny training reaches a loss threshold on synthetic data;
checkpoint resume is bit-exact; AdamW matches torch.optim.AdamW. Keep the existing test_p1_infra tests passing and
rework test_p1_acceptance.py so it runs against reference/ by default under LAB_IMPL=solution.
