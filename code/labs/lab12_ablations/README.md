# Lab 12: The architecture design space

Chapter 12 (`ch:archdesign`). You implement the sharp pieces of a modern decoder (einsum
plumbing, truncated-normal init, parameter-matched MLP widths, positional schemes, norm
placement, z-loss and soft-capping, exact parameter accounting, ablation summaries) and then
run an ablation sweep with the configurable model in `llmsys/ablate.py`.

```
LAB_IMPL=starter  python -m pytest -q labs/lab12_ablations   # your work
LAB_IMPL=solution python -m pytest -q labs/lab12_ablations   # reference
python labs/lab12_ablations/run.py --steps 400 --seeds 0 1    # CPU sweep (~minutes)
python labs/lab12_ablations/run.py --plot abl.json            # learning curves
```
`pip install einops` if you want to try the einops versions shown in the chapter (the lab
itself only needs PyTorch).
