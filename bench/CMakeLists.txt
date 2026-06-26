# Per-op performance binaries. Each is a separate executable that runs ONE op
# at the real Qwen3.6-27B shapes and is the target for ncu/nsys.
#   qus_add_bench(<name> SOURCES <files...>)
# Built with -lineinfo so ncu can attribute SASS to source lines.
# Performance is GATED by ncu (see docs/l1-op-test-standard.md §2), not by the
# GB/s these binaries print.
function(qus_add_bench name)
  cmake_parse_arguments(arg "" "" "SOURCES" ${ARGN})
  add_executable(${name} ${arg_SOURCES})
  target_link_libraries(${name} PRIVATE qus_core)
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

qus_add_bench(qus_silu_and_mul_bench SOURCES silu_and_mul_bench.cu)
qus_add_bench(qus_residual_add_bench SOURCES residual_add_bench.cu)
qus_add_bench(qus_sigmoid_gate_mul_bench SOURCES sigmoid_gate_mul_bench.cu)
qus_add_bench(qus_rmsnorm_bench SOURCES rmsnorm_bench.cu)
qus_add_bench(qus_l2norm_bench SOURCES l2norm_bench.cu)
qus_add_bench(qus_gdn_gating_bench SOURCES gdn_gating_bench.cu)
qus_add_bench(qus_rope_bench SOURCES rope_bench.cu)
qus_add_bench(qus_embed_gather_bench SOURCES embed_gather_bench.cu)
qus_add_bench(qus_argmax_bench SOURCES argmax_bench.cu)
