# Standalone E8 compressed-KV correctness oracle.
#
# This directory is intentionally independent of ninfer_core: the E8
# Conway-Sloane codecs under test are self-contained kernels (no engine,
# no hosts, no weight artifacts). It can be built two ways:
#
#   1. From the NInfer top level with CUDA already configured
#        cmake -S . -B build -GNinja -DCMAKE_CUDA_ARCHITECTURES=120a
#        cmake --build build --target ninfer_kv_e8_verify
#      The top-level CMakeLists does NOT add tools/ automatically; add
#      'add_subdirectory(tools/test_kv)' to the root if you want it wired
#      into a normal 'cmake --build build' cycle.
#
#   2. Standalone with raw nvcc:
#        nvcc -arch=sm_120a -O3 test_e8_codec.cu verify_1m_retrieval.cu -o verify_1m_retrieval
#      (the .cu files carry all their own includes; no library link needed)

cmake_minimum_required(VERSION 3.20)
project(ninfer_kv_e8_verify LANGUAGES CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120a)
endif()

add_executable(ninfer_kv_e8_verify
  test_e8_codec.cu
  verify_1m_retrieval.cu)

set_target_properties(ninfer_kv_e8_verify PROPERTIES
  CUDA_STANDARD 17
  CUDA_STANDARD_REQUIRED ON)

# 1M tokens x 256 dims FP32 keys is ~1 GB on-device; keep a host buffer
# fallback friendly on small systems.
add_test(NAME ninfer_kv_e8_verify COMMAND ninfer_kv_e8_verify 1000000)