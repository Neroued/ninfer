cmake_minimum_required(VERSION 3.28)
set(CMAKE_CUDA_ARCHITECTURES 120a)
project(fp8_tma_prefill_pilot LANGUAGES CXX CUDA)

# Task-local comparison: deliberately outside the product benchmark registrations.
# Reuse archives from a matching NInfer build instead of compiling a second engine.
get_filename_component(NINFER_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
set(NINFER_BUILD_DIR "${NINFER_SOURCE_DIR}/build" CACHE PATH "Matching NInfer build directory")
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_executable(fp8_tma_prefill_pilot pilot.cu)
set_target_properties(fp8_tma_prefill_pilot PROPERTIES
  CXX_STANDARD 20 CUDA_STANDARD 20 CXX_STANDARD_REQUIRED ON CUDA_STANDARD_REQUIRED ON)
target_include_directories(fp8_tma_prefill_pilot PRIVATE
  "${NINFER_SOURCE_DIR}/include" "${NINFER_SOURCE_DIR}/src"
  "${NINFER_SOURCE_DIR}/bench/ops")
target_compile_options(fp8_tma_prefill_pilot PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(fp8_tma_prefill_pilot PRIVATE
  "${NINFER_BUILD_DIR}/src/ops/libninfer_ops.a"
  "${NINFER_BUILD_DIR}/src/ops/libninfer_nvfp4_non_rdc.a"
  "${NINFER_BUILD_DIR}/src/core/libninfer_core.a"
  CUDA::cudart CUDA::cuda_driver CUDA::nvtx3 Threads::Threads)
