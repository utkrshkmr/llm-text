# A standalone operator library for PyTorch built with CMake (Chapter 29).
#
#   TORCH_CUDA_ARCH_LIST=8.0 cmake -S . -B build -G Ninja \
#         -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
#         -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
#   (a CUDA wheel's TorchConfig looks for the toolkit and, without TORCH_CUDA_ARCH_LIST, probes the
#   installed GPUs with a test project that old CMake versions cannot generate)
#   cmake --build build
#   python driver.py          # torch.ops.load_library("build/libsilu_mul_ex.so")
#
# find_package(Torch) comes from the torch wheel (share/cmake/Torch) and provides the TORCH_LIBRARIES
# list and TORCH_CXX_FLAGS, which carries -D_GLIBCXX_USE_CXX11_ABI=<0|1> to match the wheel's ABI.
# No Python headers are involved: TORCH_LIBRARY registers operators with the dispatcher at load
# time, and torch.ops.load_library is a dlopen.
cmake_minimum_required(VERSION 3.18)
project(silu_mul_ex LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(silu_mul_ex SHARED silu_mul_ex.cpp)
target_include_directories(silu_mul_ex PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(silu_mul_ex PRIVATE "${TORCH_LIBRARIES}")
target_compile_options(silu_mul_ex PRIVATE -O3 -Wall)
