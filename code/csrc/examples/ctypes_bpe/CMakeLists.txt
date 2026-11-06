# A plain shared library with a C ABI, called from Python through ctypes (Chapter 29).
# No Python headers, no binding library: the .so knows nothing about Python at all.
#
#   cmake -S . -B build -G Ninja && cmake --build build      # or simply:
#   g++ -O3 -std=c++20 -shared -fPIC -I../../cpu bpe_c.cpp -o build/libbpe_c.so
#   python driver.py
cmake_minimum_required(VERSION 3.18)
project(bpe_c LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(bpe_c SHARED bpe_c.cpp)
target_include_directories(bpe_c PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../cpu)
target_compile_options(bpe_c PRIVATE -O3 -Wall -fvisibility=hidden)
