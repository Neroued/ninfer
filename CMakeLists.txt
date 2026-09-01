cmake_minimum_required(VERSION 3.28)

# Upstream NInfer is compiled only for sm_120a. This port adds sm_70 (Volta /
# Tesla V100) as a peer compile-time target. Validate before CUDA compiler detection so CMake never
# injects a generic host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a" AND NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a or 70 (Volta port); got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

# __CUDA_ARCH__ is only ever defined during device-code compilation, so it can't guide
# host-side dispatch decisions (e.g. "call the SIMT kernel instead of the mma one on
# Volta") — those live in plain .cpp/.cu host functions. This is the equivalent signal
# for host code.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  add_compile_definitions(NINFER_VOLTA_BUILD=1)
endif()

project(ninfer LANGUAGES C CXX CUDA)

# CUTLASS: Volta (sm_70) has real FP16 tensor cores (mma.sync.m8n8k4) but no bf16/int
# tensor-core support and no equivalent to the mma.cuh helpers this codebase's Ampere+
# kernels use. Rather than hand-writing raw Volta mma.sync PTX (a large, error-prone
# undertaking), the Volta build vendors NVIDIA's own CUTLASS
# GEMM template library, which has first-class, NVIDIA-maintained Sm70 support via its
# legacy ("2.x-style") template API. Header-only; fetched only for the Volta build since
# sm_120a already has its own tensor-core kernels. Pinned to the same v4.4.2 tag
# independently verified (via a standalone sanity GEMM, correctness-checked against
# CUTLASS's own reference kernel on real V100 hardware) to build and run correctly here.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  include(FetchContent)
  set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")
  set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "Disable CUTLASS tests")
  set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "Disable CUTLASS examples")
  FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/nvidia/cutlass.git
    GIT_TAG v4.4.2
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE)
  FetchContent_MakeAvailable(cutlass)
endif()

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

# sm_120a needs CUDA 13.1+; the sm_70 Volta port targets CUDA 12.8, the last
# toolkit that still assembles sm_70 (CUDA 13.x dropped it). Each arch enforces
# its own floor rather than a single global one.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "120a" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  message(FATAL_ERROR
    "sm_120a requires CUDA 13.1 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70" AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
  message(FATAL_ERROR
    "The sm_70 Volta port requires CUDA 12.8 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
endif()
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70" AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  message(FATAL_ERROR
    "sm_70 requires CUDA 12.8; CUDA 13.x no longer assembles Volta device code (found "
    "${CMAKE_CUDA_COMPILER_VERSION})")
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
  libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
if(NINFER_BUILD_MEDIA_ACQUIRE)
  pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
endif()
find_package(Threads REQUIRED)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
