# Native target-private MTP proposal/verification microbenchmark. It enters through the target
# Program facade and executes the production round schedule over a real .ninfer artifact.
add_executable(ninfer_qwen3_5_mtp_round_bench
  "${CMAKE_CURRENT_LIST_DIR}/mtp_round_bench.cpp")
ninfer_internal_includes(ninfer_qwen3_5_mtp_round_bench)
target_link_libraries(ninfer_qwen3_5_mtp_round_bench PRIVATE
  ninfer_engine ninfer_model_runtime ninfer_core)

# Host n-gram (PLE) gather latency for decode windows over a table file.
add_executable(ninfer_qwen3_5_ple_gather_bench
  "${CMAKE_CURRENT_LIST_DIR}/ple_gather_bench.cpp")
ninfer_internal_includes(ninfer_qwen3_5_ple_gather_bench)
target_link_libraries(ninfer_qwen3_5_ple_gather_bench PRIVATE ninfer_model_runtime ninfer_core)
