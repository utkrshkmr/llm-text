"""Load the CMake-built operator library and call its op (Chapter 29).

    TORCH_CUDA_ARCH_LIST=8.0 cmake -S . -B build -G Ninja \
          -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
          -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
    cmake --build build
    python driver.py
"""
from __future__ import annotations

import glob
import os

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    libs = glob.glob(os.path.join(HERE, "build", "libsilu_mul_ex*"))
    if not libs:
        raise FileNotFoundError("build the library first (see the module docstring)")
    torch.ops.load_library(libs[0])          # dlopen: TORCH_LIBRARY static initialisers register the op
    g, u = torch.randn(4, 1024), torch.randn(4, 1024)
    y = torch.ops.llmsys_ex.silu_mul(g, u)
    print("max |diff| vs F.silu(g)*u:", (y - F.silu(g) * u).abs().max().item())
    print("schema:", torch.ops.llmsys_ex.silu_mul.default._schema)


if __name__ == "__main__":
    main()
