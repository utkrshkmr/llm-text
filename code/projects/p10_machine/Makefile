# Project 10: know your machine (Chapter 6). Run from this directory.
PY ?= python
OUT ?= results

.PHONY: test test-gpu smoke side full figures

test:      ## CPU acceptance tests against reference/ (P10_IMPL=... for your package)
	cd ../.. && LAB_IMPL=$${LAB_IMPL:-solution} $(PY) -m pytest -q projects/p10_machine -m "not gpu"

test-gpu:  ## brief GPU acceptance tests (about 15 s)
	cd ../.. && LAB_IMPL=$${LAB_IMPL:-solution} $(PY) -m pytest -q projects/p10_machine -m gpu

smoke:     ## about 40 s of GPU time: every benchmark once, small sweeps
	$(PY) run.py --config configs/smoke.json --out $(OUT)/smoke.json

side:      ## the side experiments the chapter quotes (about 1 min; peer checks need 2 GPUs)
	$(PY) side.py --out $(OUT)/side_experiments.json

full:      ## the full study: five repeats on GPU0, one on GPU1 (estimate; see README)
	for i in 1 2 3 4 5; do $(PY) run.py --config configs/full.json --out $(OUT)/full_$$i.json; done
	CUDA_VISIBLE_DEVICES=1,0 $(PY) run.py --config configs/full.json --out $(OUT)/full_gpu1.json

figures:   ## regenerate the chapter figures from a results file
	$(PY) figures.py --json $(OUT)/smoke_a100_pcie.json
