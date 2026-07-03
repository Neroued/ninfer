# Register a unit-test executable linked against qus_core.
#   qus_add_test(<name> SOURCES <files...> [NEEDS_SOURCE_DIR])
# qus_core PUBLIC-exports include/, so tests need no extra include dirs.
# NEEDS_SOURCE_DIR defines QUS_SOURCE_DIR for tests that read repo fixtures.
function(qus_add_test name)
  cmake_parse_arguments(arg "NEEDS_SOURCE_DIR" "" "SOURCES" ${ARGN})
  add_executable(${name} ${arg_SOURCES})
  target_link_libraries(${name} PRIVATE qus_core)
  if(arg_NEEDS_SOURCE_DIR)
    target_compile_definitions(${name} PRIVATE QUS_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

qus_add_test(qus_device_test                 SOURCES test_device.cpp)
qus_add_test(qus_tensor_test                 SOURCES test_tensor.cpp)
qus_add_test(qus_weight_test                 SOURCES test_weight.cpp)
qus_add_test(qus_arena_test                  SOURCES test_arena.cpp)
qus_add_test(qus_kv_cache_test               SOURCES test_kv_cache.cpp)
qus_add_test(qus_state_store_test            SOURCES test_state_store.cpp)
qus_add_test(qus_weight_store_test           SOURCES test_weight_store.cpp       NEEDS_SOURCE_DIR)
qus_add_test(qus_q5090_parser_test           SOURCES test_q5090_parser.cpp       NEEDS_SOURCE_DIR)
qus_add_test(qus_q5090_pack_golden_test      SOURCES test_q5090_pack_golden.cpp  NEEDS_SOURCE_DIR)
qus_add_test(qus_weight_store_real_file_test SOURCES test_weight_store_real.cpp  NEEDS_SOURCE_DIR)
qus_add_test(qus_engine_real_file_test       SOURCES test_engine_real_file.cpp   NEEDS_SOURCE_DIR)
qus_add_test(qus_engine_mtp_e2e_strict_test
             SOURCES test_engine_mtp_e2e.cpp NEEDS_SOURCE_DIR)
qus_add_test(qus_model_config_test           SOURCES test_model_config.cpp)
qus_add_test(qus_model_bind_test             SOURCES test_model_bind.cpp        NEEDS_SOURCE_DIR)
qus_add_test(qus_model_blocks_test           SOURCES test_model_blocks.cpp      NEEDS_SOURCE_DIR)
target_compile_definitions(qus_model_blocks_test PRIVATE QUS_MODEL_TESTING=1)
qus_add_test(qus_engine_memory_stats_test    SOURCES test_engine_memory_stats.cpp NEEDS_SOURCE_DIR)
qus_add_test(qus_qwen_text_cli_test          SOURCES test_qwen_text_cli.cpp)
qus_add_test(qus_qwen_chat_template_test     SOURCES test_qwen_chat_template.cpp)
qus_add_test(qus_qwen_text_tokenizer_test
             SOURCES test_qwen_text_tokenizer.cpp
             NEEDS_SOURCE_DIR)
target_include_directories(qus_qwen_text_tokenizer_test PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
qus_add_test(qus_qwen_text_runner_test SOURCES test_qwen_text_runner.cpp)
qus_add_test(qus_bench_support_test
             SOURCES test_qus_bench_support.cpp
                     ${PROJECT_SOURCE_DIR}/bench/qus_bench_support.cpp
             NEEDS_SOURCE_DIR)
target_include_directories(qus_bench_support_test
                           PRIVATE
                           ${PROJECT_SOURCE_DIR}/bench
                           ${PROJECT_SOURCE_DIR}/third_party)
qus_add_test(qus_runtime_file_tap_test
             SOURCES test_runtime_file_tap.cpp NEEDS_SOURCE_DIR)

add_executable(qus_block_dump ${PROJECT_SOURCE_DIR}/tools/parity/block_dump.cpp)
target_link_libraries(qus_block_dump PRIVATE qus_core)
target_compile_definitions(qus_block_dump PRIVATE QUS_MODEL_TESTING=1)
target_include_directories(qus_block_dump PRIVATE ${PROJECT_SOURCE_DIR}/src
                                                  ${PROJECT_SOURCE_DIR}/third_party)

add_executable(qus_layer_dump ${PROJECT_SOURCE_DIR}/tools/parity/layer_dump.cpp)
target_link_libraries(qus_layer_dump PRIVATE qus_core)
target_compile_definitions(qus_layer_dump PRIVATE QUS_MODEL_TESTING=1)
target_include_directories(qus_layer_dump PRIVATE ${PROJECT_SOURCE_DIR}/src
                                                  ${PROJECT_SOURCE_DIR}/third_party)

qus_add_test(qus_silu_and_mul_test SOURCES kernels/test_silu_and_mul.cpp)
qus_add_test(qus_residual_add_test SOURCES kernels/test_residual_add.cpp)
qus_add_test(qus_sigmoid_gate_mul_test SOURCES kernels/test_sigmoid_gate_mul.cpp)
qus_add_test(qus_rmsnorm_test SOURCES kernels/test_rmsnorm.cpp)
qus_add_test(qus_l2norm_test SOURCES kernels/test_l2norm.cpp)
qus_add_test(qus_gdn_gating_test SOURCES kernels/test_gdn_gating.cpp)
qus_add_test(qus_gdn_in_ab_prefill_test SOURCES kernels/test_gdn_in_ab_prefill.cpp)
qus_add_test(qus_gated_delta_rule_test SOURCES kernels/test_gated_delta_rule.cpp)
qus_add_test(qus_causal_conv1d_test SOURCES kernels/test_causal_conv1d.cpp)
qus_add_test(qus_gdn_commit_test SOURCES kernels/test_gdn_commit.cpp)
qus_add_test(qus_rope_test SOURCES kernels/test_rope.cpp)
qus_add_test(qus_gqa_attention_test SOURCES kernels/test_gqa_attention.cpp)
qus_add_test(qus_embed_gather_test SOURCES kernels/test_embed_gather.cpp)
qus_add_test(qus_argmax_test SOURCES kernels/test_argmax.cpp)
qus_add_test(qus_linear_test SOURCES kernels/test_linear.cpp)
qus_add_test(qus_mtp_pack_test SOURCES kernels/test_mtp_pack.cpp)
qus_add_test(qus_mtp_round_test SOURCES kernels/test_mtp_round.cpp)
