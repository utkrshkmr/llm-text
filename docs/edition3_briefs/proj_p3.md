# Brief: Project 3, a scaling-law study under a fixed budget (chapter 44, kit p3_scaling)
Read projects_common.md first, then docs/PLAN.md §4 "P3", the existing chapter and kit (training_service, ledger: keep
them and their tests), Chapters 12, 13, 15 (book + code, llmsys.scaling), code/llmsys/optim.py. GPU:
CUDA_VISIBLE_DEVICES=1, short runs only.
Scope: sweep 2-100M-parameter models under a fixed GPU-hour budget on real data (sample via `datasets`, plus
--synthetic); μP parametrisation for hyperparameter transfer (implement muP width scaling for the llmsys GPT:
init, per-layer LR, output multiplier; coordinate check); WSD schedule with branched decays sharing the stable
phase; IsoFLOP and parametric (Chinchilla form) fits with bootstrap CIs; pre-registered prediction file committed
before a held-out ~300M run. reference/: sweep driver, muP module, fitting module, prediction/registration tool,
report tables; configs smoke/full (~20 GPU-h).
Acceptance (CPU, tiny): coordinate check shows activation scales stable across widths; branched WSD runs reproduce
the stable-phase prefix bit-exactly; fitters recover the parameters of a synthetic law within CI; the
pre-registration file format and hash check; existing infra tests still pass.
