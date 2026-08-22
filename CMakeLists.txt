cmake_minimum_required(VERSION 3.28)

# The current implementation is compiled only for sm_120a. Set and validate the
# architecture before CUDA compiler detection so CMake never injects a generic
# host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  message(FATAL_ERROR
    "NInfer supports only CMAKE_CUDA_ARCHITECTURES=120a; got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX CUDA)

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

if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.1)
  message(FATAL_ERROR
    "NInfer requires CUDA 13.1 or newer; found CUDA compiler "
    "${CMAKE_CUDA_COMPILER_VERSION}")
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

# FFmpeg and libcurl are located cross-platform: pkg-config on POSIX, an
# install prefix / vcpkg tree on Windows. Both are exposed as NInfer::FFmpeg
# and NInfer::Curl (see cmake/NInferMediaDeps.cmake).
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(NInferMediaDeps)
ninfer_find_ffmpeg()
if(NINFER_BUILD_MEDIA_ACQUIRE)
  ninfer_find_curl()
endif()

find_package(Threads REQUIRED)

# MSVC is true for any MSVC-ABI compiler, which includes clang-cl. Settings that
# every such compiler needs go here; settings specific to cl are gated further.
if(MSVC)
  # CUDA 13's CCCL (cuda/std) requires the conforming MSVC preprocessor. cl needs
  # to be told; clang-cl's preprocessor is already conforming and rejects the flag
  # as unknown, so ask only when the C++ frontend really is cl. The -Xcompiler
  # form is unconditional because nvcc drives cl as its host compiler on Windows
  # regardless of which compiler CMake uses for C++, and the host stubs it
  # generates need the flag too. (The 128-byte-aligned TMA descriptor parameters
  # are passed by device pointer rather than by value precisely so MSVC's stub
  # compiler does not reject them.)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor>)
  endif()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor")
  # windows.h defines min/max as macros which clobber std::min/std::max; NOMINMAX
  # suppresses them. UTF8PROC_STATIC keeps the vendored utf8proc header from
  # marking its symbols __declspec(dllimport) since we compile it statically. Both
  # are scoped to C/CXX (no CUDA file needs them) so the CUDA objects keep their
  # cache.
  add_compile_definitions(
    $<$<COMPILE_LANGUAGE:C,CXX>:NOMINMAX>
    $<$<COMPILE_LANGUAGE:C,CXX>:UTF8PROC_STATIC>)
endif()

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
