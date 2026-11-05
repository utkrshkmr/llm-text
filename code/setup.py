"""Build the ``llmsys._C`` extension: C++ (always) and CUDA (when a toolkit is found).

    cd code
    pip install -e . --no-build-isolation -v        # CPU ops, plus CUDA ops when nvcc is available
    LLMSYS_NO_CUDA=1 pip install -e . --no-build-isolation   # force a CPU-only build
    TORCH_CUDA_ARCH_LIST="8.0" python setup.py build_ext --inplace   # in-tree .so, no install

Every source under ``csrc/`` and ``csrc/cpu/`` is compiled; every ``csrc/cuda/*.cu`` is compiled when
CUDA is enabled, and ``LLMSYS_WITH_CUDA`` is defined so the C++ side can register CUDA kernels.
Adding an operator therefore never needs a change here: drop a ``.cpp``/``.cu`` file in and rebuild.
"""
from __future__ import annotations

import glob
import os
import platform

from setuptools import find_packages, setup
from torch.utils import cpp_extension

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "csrc")


def _cuda_wanted() -> bool:
    if os.environ.get("LLMSYS_NO_CUDA"):
        return False
    return cpp_extension.CUDA_HOME is not None


def _cxx_flags() -> list[str]:
    flags = ["-O3", "-std=c++20", "-fopenmp", "-fvisibility=hidden", "-Wall"]
    if os.environ.get("LLMSYS_CXXFLAGS"):
        flags += os.environ["LLMSYS_CXXFLAGS"].split()
    elif platform.machine() in ("x86_64", "AMD64"):
        flags += ["-march=native"]
    return flags


def _nvcc_flags() -> list[str]:
    flags = ["-O3", "-std=c++20", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-lineinfo",
             "-DLLMSYS_WITH_CUDA", "-Xcompiler", "-fopenmp"]
    if os.environ.get("LLMSYS_NVCCFLAGS"):
        flags += os.environ["LLMSYS_NVCCFLAGS"].split()
    return flags


sources = sorted(glob.glob(os.path.join(CSRC, "*.cpp")) + glob.glob(os.path.join(CSRC, "cpu", "*.cpp")))
with_cuda = _cuda_wanted()
if with_cuda:
    sources += sorted(glob.glob(os.path.join(CSRC, "cuda", "*.cu")))
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
    Ext = cpp_extension.CUDAExtension
    extra = {"cxx": _cxx_flags() + ["-DLLMSYS_WITH_CUDA"], "nvcc": _nvcc_flags()}
    libraries = ["cublas"]
else:
    Ext = cpp_extension.CppExtension
    extra = {"cxx": _cxx_flags()}
    libraries = []

ext = Ext(
    name="llmsys._C",
    sources=[os.path.relpath(s, HERE) for s in sources],
    include_dirs=[CSRC],
    extra_compile_args=extra,
    extra_link_args=["-fopenmp"],
    libraries=libraries,
)

setup(
    name="llmsys",
    version="2.0.0",
    description="Companion code for LLM Systems from Scratch",
    packages=find_packages(include=["llmsys", "llmsys.*"]),
    ext_modules=[ext],
    cmdclass={"build_ext": cpp_extension.BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.10",
    zip_safe=False,
)
