# Brief: Chapter 3 "Numbers on the Machine: Formats, Precision and Stable Algorithms" (NEW) + Lab 3

Read first: BRIEF_conventions.md (all, esp. §7 depth rules), docs/COVERAGE.md rows for ch:numerics,
book/chapters/ch02_pytorch_hardware.md (what the reader knows: tensors, dtypes, roofline basics),
book/chapters/ch14_pretraining.md §"Mixed precision" and ch17_perf_engineering.md §"Mixed precision in practice"
(those later chapters will be trimmed to point back here: make THIS chapter the complete treatment of formats and
numerics; they keep the training-loop recipes), ch33_efficient_inference.md §"Quantization fundamentals" (quant
maths lives here; ch33 keeps calibration algorithms GPTQ/AWQ/SmoothQuant and kernels), code/llmsys/quant.py.
GPU for measurements: CUDA_VISIBLE_DEVICES=0.

Files you own (create):
- book/chapters/ch03_numerics.md            # Numbers on the Machine: Formats, Precision and Stable Algorithms {#ch:numerics}
- book/solutions/ch03_solutions.md
- book/figures/ch03_numerics-<k>.svg        bit layouts of every format; number line with subnormals; MX block layout;
                                             quantisation grids (uniform/non-uniform, sym/asym); stochastic rounding
- code/llmsys/numerics.py                    bit-exact format simulator: encode/decode/round for FP32, TF32, FP16, BF16,
                                             FP8 E4M3 (OCP, with/without inf, NaN encodings) and E5M2, FP6 E2M3/E3M2,
                                             FP4 E2M1, MX block formats (MXFP8/MXFP6/MXFP4, shared E8M0 scale per 32
                                             elements) and NVFP4 (16-element blocks, E4M3 scale + per-tensor fp32 scale),
                                             INT8/INT4 with zero points; rounding modes (RNE, toward zero, stochastic);
                                             ulp/next-after; summation algorithms (naive, pairwise, Kahan, Welford);
                                             online softmax and logsumexp; accumulation-precision simulator for dot products
- code/labs/lab03_numerics/{README.md,starter.py,solution.py,test_lab03.py,bench.py}
- docs/edition3_briefs/refs/ch03_numerics.md

Chapter content (all of it, from first principles):
1. Why numbers are a systems topic: bytes per parameter set memory, bandwidth and tensor-core throughput (A100: TF32
   156 TF, BF16 312 TF; H100/B200 FP8/FP4 numbers from datasheets, labelled as such).
2. Integers and fixed point; two's complement; saturation vs wraparound.
3. IEEE 754 from the bit level: sign/exponent/mantissa, bias, normal and subnormal numbers, gradual underflow,
   inf/NaN, machine epsilon vs ulp, dynamic range in decades; worked encodings. FP32, TF32, FP16, BF16 compared
   (table: bits, range, epsilon, smallest normal/subnormal), why BF16 won for training, where FP16 overflows (exp,
   attention logits, activations of large models) and underflows (gradients).
4. Low-precision floats: FP8 E4M3 vs E5M2 (forward vs backward use), the OCP spec details (E4M3 has no inf, max
   448), FP6, FP4 E2M1; microscaling (MX) formats: block of 32 sharing an E8M0 scale; NVFP4's two-level scaling;
   why block scaling recovers range; error analysis of block scaling vs per-tensor.
5. Rounding: RNE, ties, toward zero; stochastic rounding (unbiasedness proof, variance, why it rescues low-precision
   accumulation and optimizer updates: the "update smaller than half an ulp vanishes" problem with a measured
   example of BF16 weights without master copies); Kahan compensated summation.
6. Accumulation precision: dot products in FP16/BF16 with FP32 accumulators (tensor cores), split-K, the FP8
   accumulation issue on Hopper (limited-precision accumulators, periodic promotion as in DeepSeek-V3); error
   bounds for summation (n·eps growth vs pairwise log n).
7. Quantisation maths: uniform affine quantisation (scale, zero point), symmetric vs asymmetric, clipping range and
   the MSE trade-off between clipping and rounding error, per-tensor / per-channel / per-group / per-block / per-token
   scales and what each costs in metadata bytes and kernel complexity, non-uniform grids (NF4 quantiles, log grids,
   lookup tables), outliers and why they dominate error (activation outlier channels); Hessian-weighted error
   (second-order Taylor of the loss, the OBS/OBQ idea) as the bridge to GPTQ/AWQ/SmoothQuant in [@ch:efficient].
8. Mixed precision for training and inference: the general recipe (low-precision storage and matmul inputs, high-
   precision accumulation, master weights, optimizer state precision, loss scaling static/dynamic, which ops stay in
   FP32: softmax, norms, losses, reductions), FP8 training recipes (per-tensor delayed scaling vs current scaling vs
   block scaling; Transformer-Engine style amax history), inference precision choices (weight-only vs W8A8 vs FP8 KV
   cache) with a decision table. Deep links forward to [@ch:perf], [@ch:pretrain], [@ch:efficient], [@ch:kvcache].
9. Numerically stable online algorithms: streaming max/sum, logsumexp, online softmax derivation (the rescaling
   identity) as a single-pass streaming reduction that never materialises N×N scores — the mathematical root of
   FlashAttention ([@ch:attention] uses it); Welford mean/variance (LayerNorm kernels); stable cross-entropy.
10. Research frontier (§7): FP4 training, MX/NVFP4 hardware, low-precision optimizer states, stochastic rounding
   in hardware, outlier-free architectures, posits/log number systems; Reading-the-research box: one of
   "FP8 Formats for Deep Learning" (Micikevicius et al. 2022) or the OCP MX spec paper (Rouhani et al. 2023).

Lab 3 (CPU tests): implement encode/decode for FP16/BF16/E4M3/E5M2/E2M1 and MX blocks (tests: bit-exact against
torch.float16/bfloat16/float8_e4m3fn/float8_e5m2 casts over all representable values and random inputs; MX vs a
reference), stochastic rounding (tests: unbiased mean over many draws, variance bound), Kahan/pairwise sums
(tests: error vs exact fractions.Fraction sum), online softmax and Welford (tests vs float64 two-pass),
quantise/dequantise with per-tensor/channel/group scales and asymmetric zero points (tests: round-trip error
bounds), Hessian-weighted error on a tiny linear layer. bench.py (GPU optional): BF16 weight updates with and
without master weights / with stochastic rounding over 10k steps (plot drift), FP8 matmul error vs block size,
accumulate-in-FP16 vs FP32 error vs K.

Verification: LAB_IMPL=solution pytest -q code/labs/lab03_numerics; pytest -q code/tests; site build --strict
(allowlist only); figure check in both themes.


## DeepSeek contributions (owner request, 2026-11-05). Cover each from the primary paper or repository, at §7 depth
- The DeepSeek-V3 FP8 training recipe in full, as the main case study of section 8: E4M3 for all tensors with
  fine-grained scaling (1×128 tiles for activations, 128×128 blocks for weights), online (current) scaling,
  increased accumulation precision by promoting partial sums from tensor cores to FP32 CUDA-core registers every
  128 elements along K, which components stay in BF16/FP32 (embedding, output head, MoE gating, normalisation,
  attention), low-precision storage and communication of activations and optimizer states, and the reported loss
  gap vs BF16. Simulate the tile/block scaling and the promotion interval with the lab's format simulator and
  measure the error as a function of K and of the promotion interval.
