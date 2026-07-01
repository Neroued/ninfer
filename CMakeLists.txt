cmake_minimum_required(VERSION 3.28)
project(qus LANGUAGES C CXX CUDA)

# --- Global settings ---------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a)         # RTX 5090 / Blackwell (sm_120a)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)

# --- Subdirectories ----------------------------------------------------------
enable_testing()
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(bench)
