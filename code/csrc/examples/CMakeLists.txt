# Standalone binding examples for Chapter 20. Each subdirectory is also a complete project of its
# own; this file only lets you build them all at once:
#
#   TORCH_CUDA_ARCH_LIST=8.0 cmake -S csrc/examples -B /dev/shm/ex-build -G Ninja \
#         -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
#         -DPython_EXECUTABLE=$(which python) -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
#   cmake --build /dev/shm/ex-build
#
# Turn an example off with -DEXAMPLE_<NAME>=OFF (the torch one is off by default: it compiles the
# ATen headers, which takes a minute, and needs CMAKE_PREFIX_PATH to point at torch's cmake files).
cmake_minimum_required(VERSION 3.18)
project(llmsys_examples LANGUAGES CXX)

option(EXAMPLE_PYBIND11 "pybind11 module around bpe_core.h" ON)
option(EXAMPLE_NANOBIND "nanobind module around bpe_core.h" ON)
option(EXAMPLE_CTYPES   "C-ABI shared library for ctypes" ON)
option(EXAMPLE_TORCH    "TORCH_LIBRARY operator library built with find_package(Torch)" OFF)

if(EXAMPLE_PYBIND11)
  add_subdirectory(pybind11_bpe)
endif()
if(EXAMPLE_NANOBIND)
  add_subdirectory(nanobind_bpe)
endif()
if(EXAMPLE_CTYPES)
  add_subdirectory(ctypes_bpe)
endif()
if(EXAMPLE_TORCH)
  add_subdirectory(torch_cmake)
endif()
