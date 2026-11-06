# Lab 17 — Optimizers and a resumable training loop (Chapter 17)

Implement SGD/momentum, AdamW (loop and `foreach`), warmup-cosine and WSD schedules,
global-norm clipping, a resumable memmap batch sampler, Newton–Schulz orthogonalisation (Muon's
core), state-size accounting, a JSONL run logger, atomic checkpoints, and a training loop that
resumes **bit-exactly** after a simulated crash.

```
cd code
LAB_IMPL=starter  python -m pytest labs/lab17_optim -q   # grade your work (all TODOs fail at first)
LAB_IMPL=solution python -m pytest labs/lab17_optim -q   # reference (re-exports llmsys.optim / explog)
```

Experiments (CPU-friendly defaults; use `--device cuda` on a GPU):
```
python labs/lab17_optim/run.py make-markov --data data/markov.bin      # known entropy floor
python labs/lab17_optim/run.py sweep-lr --data data/markov.bin --lrs 1e-3,3e-3,1e-2,3e-2,0.1,0.3,1
python labs/lab17_optim/run.py compare  --data data/markov.bin           # SGD / AdamW / Lion / Muon
python labs/lab17_optim/run.py wsd      --data data/markov.bin --branches 200,400,600
python labs/lab17_optim/run.py plot runs/lr-* --y val_loss --x step
```
Tiny models run fastest with one thread (`--threads 1`, the default); extra threads only add contention.
