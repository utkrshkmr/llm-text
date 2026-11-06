# Project 11: a low-precision training study (Chapter 5). Run from this directory.
PY ?= python
OUT ?= results

.PHONY: test test-gpu dry smoke full figures

test:      ## CPU acceptance tests against reference/ (P11_IMPL=... for your package)
	cd ../.. && LAB_IMPL=$${LAB_IMPL:-solution} $(PY) -m pytest -q projects/p11_lowprec -m "not gpu"

test-gpu:  ## brief GPU acceptance tests (about a minute)
	cd ../.. && LAB_IMPL=$${LAB_IMPL:-solution} $(PY) -m pytest -q projects/p11_lowprec -m gpu

dry:       ## every code path on the CPU in seconds (not a result)
	$(PY) run.py --config configs/cpu.json --device cpu --out $(OUT)/cpu_dry.json --quiet

smoke:     ## every recipe once, one seed, two break-point sweeps: about 53 min on one A100 (measured)
	$(PY) run.py --config configs/smoke.json --out $(OUT)/smoke.json --synthetic

full:      ## the study, split over the two GPUs (estimate: about 10 GPU-h; see README)
	CUDA_VISIBLE_DEVICES=0 $(PY) run.py --config configs/full.json --only gpt/ --out $(OUT)/full_gpt.json --quiet & \
	CUDA_VISIBLE_DEVICES=1 $(PY) run.py --config configs/full.json --only mlp/ --out $(OUT)/full_mlp.json --quiet; \
	wait

figures:   ## regenerate the chapter's measured figures from the smoke results
	$(PY) figures.py --json $(OUT)/smoke_a100_pcie.json
