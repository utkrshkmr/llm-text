# A pybind11 extension module built with CMake (Chapter 31).
#
#   cmake -S . -B build -G Ninja -DPython_EXECUTABLE=$(which python) && cmake --build build
#   PYTHONPATH=build python driver.py
#
# pybind11 ships its CMake package inside the pip wheel; `python -m pybind11 --cmakedir` prints the
# directory, and we append it to CMAKE_PREFIX_PATH so find_package(pybind11) works without any
# system-wide install.
cmake_minimum_required(VERSION 3.18)
project(bpe_pybind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                OUTPUT_VARIABLE pybind11_cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${pybind11_cmake_dir}")
find_package(pybind11 CONFIG REQUIRED)

# pybind11_add_module sets the Python extension suffix, hidden visibility and -fPIC for us.
pybind11_add_module(bpe_pybind MODULE bpe_pybind.cpp)
target_include_directories(bpe_pybind PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../cpu)
target_compile_options(bpe_pybind PRIVATE -O3 -Wall)
