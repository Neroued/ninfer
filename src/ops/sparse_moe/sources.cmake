target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/host/host_moe_mailbox.cu"
)

# Host (CPU) routed-expert kernels and the weight-major executor: plain C++ and C, no CUDA, so
# their tests and benchmarks link this archive alone. Every file is built for baseline x86-64: the
# AVX2 / AVX-512 kernels carry their ISA in function target attributes or a #pragma GCC target after
# the includes and run only after the cpuid check (tools/check_host_isa.py, ninfer_host_isa_guard_test).
#   host_nvfp4(_avx2)        NVFP4 W4A8/W4A4 dots, AVX-512 and AVX2
#   host_gguf(_avx512|_avx2) ggml block formats of the GSQ-RCO banks (fngine port): host_gguf.cpp
#   host_q8(_avx2)           dispatches, the others hold the AVX-512 / AVX2 kernels and quantizers
#   ggml-iq                  ggml's reference code (third_party/ggml-iq, MIT): tests and oracles
add_library(ninfer_host_moe STATIC
  "${CMAKE_CURRENT_LIST_DIR}/host/host_nvfp4.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host/host_nvfp4_avx2.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host/host_moe_executor.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host/host_moe_executor_gguf.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host/host_gguf.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host/host_gguf_avx512.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host/host_gguf_avx2.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host/host_q8.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host/host_q8_avx2.cpp"
  "${PROJECT_SOURCE_DIR}/third_party/ggml-iq/ggml-iq.c")
ninfer_internal_includes(ninfer_host_moe)
target_include_directories(ninfer_host_moe PUBLIC "${PROJECT_SOURCE_DIR}/third_party")
target_link_libraries(ninfer_host_moe PUBLIC Threads::Threads)
# No FMA contraction: the quantizers stay bit-identical to ggml's reference, and the AVX2 kernels
# to the AVX-512 ones (every FMA is explicit in both).
set_source_files_properties(
  "${CMAKE_CURRENT_LIST_DIR}/host/host_gguf_avx512.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host/host_gguf_avx2.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host/host_q8.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/host/host_q8_avx2.cpp"
  PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
set_source_files_properties("${PROJECT_SOURCE_DIR}/third_party/ggml-iq/ggml-iq.c"
  PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
target_link_libraries(ninfer_ops PUBLIC ninfer_host_moe)
