# Lab A — GPU kernels with Triton

Write Triton kernels for vector add, fused row softmax, RMSNorm, a tiled/grouped matmul,
and a FlashAttention forward pass; plus a PyTorch online-softmax reference and roofline
napkin math. `_w8a16_kernel` (int8-weight GEMM) is provided complete as a worked example.

```bash
cd code
# correctness on any machine (CPU uses Triton's interpreter automatically):
LAB_IMPL=starter  python -m pytest -q labs/labA_triton
LAB_IMPL=solution python -m pytest -q labs/labA_triton
# performance (GPU only):
python labs/labA_triton/bench.py --impl starter --hw H100-SXM
```
Interpreter tips: `TRITON_INTERPRET=1` must be set *before* `import triton`; inside a
kernel you can then `print()` tensors and use a debugger. Use small shapes; it is slow.
