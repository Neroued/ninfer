# Native target-private MTP proposal/verification microbenchmark. It enters through the target
# Program facade and executes the production round schedule over a real .ninfer artifact.
add_executable(ninfer_qwen3_5_mtp_round_bench
  "${CMAKE_CURRENT_LIST_DIR}/mtp_round_bench.cpp")
ninfer_internal_includes(ninfer_qwen3_5_mtp_round_bench)
target_link_libraries(ninfer_qwen3_5_mtp_round_bench PRIVATE
  ninfer_engine ninfer_model_runtime ninfer_core)

# Complete production DFlash steady round: confirmed-context append, next proposal, target
# verification/acceptance, and host publication over a real 35B artifact.
add_executable(ninfer_qwen3_5_dflash_round_bench
  "${CMAKE_CURRENT_LIST_DIR}/dflash_round_bench.cpp")
ninfer_internal_includes(ninfer_qwen3_5_dflash_round_bench)
target_link_libraries(ninfer_qwen3_5_dflash_round_bench PRIVATE
  ninfer_engine ninfer_model_runtime ninfer_core)
