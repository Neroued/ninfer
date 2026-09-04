# Standalone E8 compressed-KV correctness oracle.
#
# This is an INDEPENDENT oracle for the E8 Conway-Sloane lattice / 240-root
# codec mathematics that the production rk2v4-e8 (and rk4v4-e8) codecs are
# based on. The kernel header here (test_e8_codec.cuh) is a self-contained,
# deliberately separate implementation — it does not #include the production
# src/ops/kernel/e8_*.cuh, so a self-including (circular) test is avoided.
# It therefore validates the math, not the production 208 B plane layout /
# V rotation / paging (that route is covered by the end-to-end
# boot + needle verification, see PORT-RK2V4E8.md).
#
# It has no engine, host, or weight dependencies, so it builds and runs in
# isolation. It is wired into the top-level build under BUILD_TESTING (see the
# root CMakeLists.txt), so from the NInfer top level:
#     cmake --build build --target ninfer_kv_e8_verify
#     ctest --test-dir build -R ninfer_kv_e8_verify --output-on-failure
#
# It can also be configured as its own standalone project (this file carries
# its own project() + CUDA settings):
#     cmake -S tools/test_kv -B build-test-kv -GNinja -DCMAKE_CUDA_ARCHITECTURES=120a
#     cmake --build build-test-kv --target ninfer_kv_e8_verify
# ...or built directly with raw nvcc (the .cu files carry all their own
# includes; no library link needed):
#     nvcc -arch=sm_120a -O3 test_e8_codec.cu verify_1m_retrieval.cu -o verify_1m_retrieval

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