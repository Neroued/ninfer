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
