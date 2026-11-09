find_package(Python3 REQUIRED COMPONENTS Interpreter)

include(cmake/NinferTests.cmake)
include(cmake/CoreTests.cmake)
include(artifact/tests.cmake)
include(cmake/RuntimeTests.cmake)
include(cmake/ConstraintTests.cmake)
include(cmake/ProductTests.cmake)
include(cmake/BenchSupportTests.cmake)
include(models/qwen3_5/tests.cmake)
include(ops/tests.cmake)

# One 16 GB board cannot hold two tests that each need most of it. Every test that loads a model
# artifact (NINFER_TEST_ARTIFACT, NINFER_TEST_QWEN4_ARTIFACT, NINFER_TEST_QWEN4_VISION_ARTIFACT: several
# GB per board) and every test measured above 1 GiB on a board (CUDA context included, with
# `nvidia-smi` during a serial run) holds this lock, so `ctest -j` runs them one at a time; the
# remaining GPU tests, below 1 GiB each, still run beside them.
set_tests_properties(
  ninfer_serve_engine_failure_real_test
  ninfer_serve_tp_transport_recovery_real_test
  ninfer_qwen3_5_prefix_real_test
  ninfer_qwen4_tp2_engine_test
  ninfer_qwen4_mtp_engine_test
  ninfer_qwen4_vision_engine_test
  ninfer_qwen3_5_score_real_test
  ninfer_qwen3_5_score_tp2_real_test
  ninfer_qwen3_5_vision_workspace_test
  ninfer_qwen3_5_dflash2_real_test
  ninfer_qwen3_5_dflash_prefill_real_test
  ninfer_qwen3_5_sharded_load_real_test
  ninfer_qwen3_5_sharded_load_mtp_real_test
  ninfer_qwen3_5_text_context_tp2_real_test
  ninfer_qwen3_5_engine_tp2_real_test
  ninfer_qwen3_5_engine_mtp_tp2_real_test
  ninfer_qwen3_5_engine_mtp_tp2_optimized_real_test
  ninfer_qwen3_5_engine_dflash2_tp2_real_test
  ninfer_qwen3_5_engine_vision_tp2_real_test
  ninfer_qwen3_5_engine_vision_tp2_mtp_real_test
  ninfer_qwen3_5_engine_vision_tp2_dflash2_real_test
  ninfer_qwen3_5_text_context_tp2_test
  ninfer_softmax_attention_test
  ninfer_gdn_replay_fold_test
  ninfer_output_head_split_test
  ninfer_linear_q8_a16_test
  ninfer_linear_fp8_a16_test
  ninfer_linear_q6_a16_test
  ninfer_linear_topk_test
  ninfer_linear_q4_a16_test
  ninfer_linear_q5_a16_test
  ninfer_embedding_test
  ninfer_linear_nvfp4_a16_test
  PROPERTIES RESOURCE_LOCK ninfer_gpu_memory)
