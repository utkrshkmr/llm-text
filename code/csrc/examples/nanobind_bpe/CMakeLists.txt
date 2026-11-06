# The nanobind variant of pybind11_bpe (Chapter 20). nanobind's CMake package also ships in its
# wheel; `python -m nanobind --cmake_dir` prints it. nanobind_add_module adds size/speed
# optimisations (-Os for the binding code, LTO, a shared libnanobind) that pybind11 leaves to you.
cmake_minimum_required(VERSION 3.18)
project(bpe_nanobind LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                OUTPUT_VARIABLE nanobind_cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${nanobind_cmake_dir}")
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(bpe_nanobind NB_STATIC bpe_nanobind.cpp)
target_include_directories(bpe_nanobind PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../cpu)
