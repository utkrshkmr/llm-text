"""The shortest path from a C++ function to a Python call: torch.utils.cpp_extension.load_inline
(Chapter 29).

    python inline_ops.py

The C++ source is a Python string. load_inline writes it to a build directory (``$TORCH_EXTENSIONS_DIR``
or ``~/.cache/torch_extensions``), generates the PYBIND11_MODULE boilerplate for every name in
``functions``, compiles with ninja, and imports the result. Rebuilds are skipped when the source
hash is unchanged, so the second run is instant.

Two flavours are shown: a function exposed through pybind11 (``functions=[...]``), and an
operator registered with the dispatcher (``TORCH_LIBRARY`` inside the source), which is what the
rest of the chapter builds on.
"""
from __future__ import annotations

import os
import time

import torch
from torch.utils.cpp_extension import load_inline

PAIR_COUNT_SRC = r"""
#include <torch/extension.h>

// Non-overlapping occurrences of (a, b) in ids, left to right: the count that one BPE merge of
// that pair would remove from the sequence.
int64_t pair_count(const at::Tensor& ids, int64_t a, int64_t b) {
  TORCH_CHECK(ids.dim() == 1 && ids.scalar_type() == at::kLong, "ids must be a 1-D int64 tensor");
  const auto c = ids.contiguous();
  const int64_t* p = c.data_ptr<int64_t>();
  const int64_t n = c.numel();
  int64_t count = 0;
  for (int64_t i = 0; i + 1 < n; ++i) {
    if (p[i] == a && p[i + 1] == b) { ++count; ++i; }
  }
  return count;
}
"""

SCALE_OP_SRC = r"""
#include <torch/extension.h>

// A dispatcher operator defined inline: schema + CPU kernel. Visible as torch.ops.inline_demo.scale.
at::Tensor scale_cpu(const at::Tensor& x, double s) {
  return x * s;      // ATen does the work; the point is the registration, not the kernel
}

TORCH_LIBRARY(inline_demo, m) {
  m.def("scale(Tensor x, float s) -> Tensor");
}
TORCH_LIBRARY_IMPL(inline_demo, CPU, m) {
  m.impl("scale", &scale_cpu);
}
"""


def build():
    build_dir = os.environ.get("LLMSYS_JIT_DIR")
    kw = dict(extra_cflags=["-O3", "-std=c++20"], verbose=False)
    if build_dir:
        os.makedirs(os.path.join(build_dir, "inline_demo"), exist_ok=True)
        kw["build_directory"] = os.path.join(build_dir, "inline_demo")
    # Both sources go in one module; `functions` lists the pybind11-exposed names only.
    return load_inline(name="inline_demo", cpp_sources=[PAIR_COUNT_SRC, SCALE_OP_SRC],
                       functions=["pair_count"], **kw)


def main() -> None:
    t0 = time.perf_counter()
    mod = build()
    print(f"build/load: {time.perf_counter() - t0:.2f}s")
    ids = torch.tensor([1, 2, 1, 2, 2, 1, 2, 7])
    print("pair_count((1,2)) =", mod.pair_count(ids, 1, 2), "(expected 3)")
    x = torch.arange(4.0)
    print("torch.ops.inline_demo.scale:", torch.ops.inline_demo.scale(x, 2.5))


if __name__ == "__main__":
    main()
