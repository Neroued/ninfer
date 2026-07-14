# Register a unit-test executable linked against ninfer_core.
#   ninfer_add_test(<name> SOURCES <files...> [NEEDS_SOURCE_DIR])
# ninfer_core PUBLIC-exports include/, so tests need no extra include dirs.
# NEEDS_SOURCE_DIR defines NINFER_SOURCE_DIR for tests that read repo fixtures.
function(ninfer_add_test name)
  cmake_parse_arguments(arg "NEEDS_SOURCE_DIR" "" "SOURCES" ${ARGN})
  add_executable(${name} ${arg_SOURCES})
  target_link_libraries(${name} PRIVATE ninfer_core)
  if(arg_NEEDS_SOURCE_DIR)
    target_compile_definitions(${name} PRIVATE NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

ninfer_add_test(ninfer_device_test                 SOURCES test_device.cpp)
ninfer_add_test(ninfer_tensor_test                 SOURCES test_tensor.cpp)
ninfer_add_test(ninfer_weight_test                 SOURCES test_weight.cpp)
ninfer_add_test(ninfer_arena_test                  SOURCES test_arena.cpp)
ninfer_add_test(ninfer_kv_cache_test               SOURCES test_kv_cache.cpp)
ninfer_add_test(ninfer_state_store_test            SOURCES test_state_store.cpp)
ninfer_add_test(ninfer_weight_store_test           SOURCES test_weight_store.cpp       NEEDS_SOURCE_DIR)
ninfer_add_test(ninfer_q5090_parser_test           SOURCES test_q5090_parser.cpp       NEEDS_SOURCE_DIR)
ninfer_add_test(ninfer_q5090_pack_golden_test      SOURCES test_q5090_pack_golden.cpp  NEEDS_SOURCE_DIR)
ninfer_add_test(ninfer_weight_store_real_file_test SOURCES test_weight_store_real.cpp  NEEDS_SOURCE_DIR)
ninfer_add_test(ninfer_engine_real_file_test       SOURCES test_engine_real_file.cpp   NEEDS_SOURCE_DIR)
ninfer_add_test(ninfer_engine_mtp_e2e_test
             SOURCES test_engine_mtp_e2e.cpp NEEDS_SOURCE_DIR)
ninfer_add_test(ninfer_engine_vision_e2e_test
             SOURCES test_engine_vision_e2e.cpp NEEDS_SOURCE_DIR)
add_test(NAME ninfer_engine_mtp_e2e_test_capacity_fallback
         COMMAND ninfer_engine_mtp_e2e_test capacity_fallback)
add_test(NAME ninfer_engine_mtp_e2e_test_fallback_after_accept
         COMMAND ninfer_engine_mtp_e2e_test fallback_after_accept)
add_test(NAME ninfer_engine_mtp_e2e_test_stop_truncation
         COMMAND ninfer_engine_mtp_e2e_test stop_truncation)
add_test(NAME ninfer_engine_mtp_e2e_test_multiturn_prefix_cache
         COMMAND ninfer_engine_mtp_e2e_test multiturn_prefix_cache)
ninfer_add_test(ninfer_model_config_test           SOURCES test_model_config.cpp)
ninfer_add_test(ninfer_model_bind_test             SOURCES test_model_bind.cpp        NEEDS_SOURCE_DIR)
ninfer_add_test(ninfer_model_blocks_test           SOURCES test_model_blocks.cpp      NEEDS_SOURCE_DIR)
target_compile_definitions(ninfer_model_blocks_test PRIVATE NINFER_MODEL_TESTING=1)
ninfer_add_test(ninfer_engine_memory_stats_test    SOURCES test_engine_memory_stats.cpp NEEDS_SOURCE_DIR)
ninfer_add_test(ninfer_qwen_text_cli_test          SOURCES test_qwen_text_cli.cpp)
ninfer_add_test(ninfer_qwen_chat_template_test     SOURCES test_qwen_chat_template.cpp NEEDS_SOURCE_DIR)
target_include_directories(ninfer_qwen_chat_template_test PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
ninfer_add_test(ninfer_qwen_text_tokenizer_test
             SOURCES test_qwen_text_tokenizer.cpp
             NEEDS_SOURCE_DIR)
target_include_directories(ninfer_qwen_text_tokenizer_test PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
ninfer_add_test(ninfer_qwen_text_runner_test SOURCES test_qwen_text_runner.cpp)
ninfer_add_test(ninfer_processor_test SOURCES test_processor.cpp)
target_include_directories(ninfer_processor_test PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer_processor_test PRIVATE Threads::Threads)

# Serving-layer schema/translation contract (consumed by external OpenAI clients).
add_executable(ninfer_openai_schema_test test_openai_schema.cpp)
target_link_libraries(ninfer_openai_schema_test PRIVATE ninfer_serve)
target_include_directories(ninfer_openai_schema_test PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
add_test(NAME ninfer_openai_schema_test COMMAND ninfer_openai_schema_test)
add_executable(ninfer_tool_call_parser_test test_tool_call_parser.cpp)
target_link_libraries(ninfer_tool_call_parser_test PRIVATE ninfer_serve)
target_include_directories(ninfer_tool_call_parser_test PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
add_test(NAME ninfer_tool_call_parser_test COMMAND ninfer_tool_call_parser_test)
# Serving-layer Anthropic Messages schema/translation contract (consumed by Claude Code).
add_executable(ninfer_anthropic_schema_test test_anthropic_schema.cpp)
target_link_libraries(ninfer_anthropic_schema_test PRIVATE ninfer_serve)
target_include_directories(ninfer_anthropic_schema_test PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
add_test(NAME ninfer_anthropic_schema_test COMMAND ninfer_anthropic_schema_test)
add_executable(ninfer_generation_budget_test test_generation_budget.cpp)
target_link_libraries(ninfer_generation_budget_test PRIVATE ninfer_serve)
target_include_directories(ninfer_generation_budget_test PRIVATE ${PROJECT_SOURCE_DIR}/third_party)
add_test(NAME ninfer_generation_budget_test COMMAND ninfer_generation_budget_test)
ninfer_add_test(ninfer_bench_support_test
             SOURCES test_ninfer_bench_support.cpp
                     ${PROJECT_SOURCE_DIR}/bench/ninfer_bench_support.cpp
             NEEDS_SOURCE_DIR)
target_include_directories(ninfer_bench_support_test
                           PRIVATE
                           ${PROJECT_SOURCE_DIR}/bench
                           ${PROJECT_SOURCE_DIR}/third_party)
ninfer_add_test(ninfer_runtime_file_tap_test
             SOURCES test_runtime_file_tap.cpp NEEDS_SOURCE_DIR)

add_executable(ninfer_block_dump ${PROJECT_SOURCE_DIR}/tools/parity/block_dump.cpp)
target_link_libraries(ninfer_block_dump PRIVATE ninfer_core)
target_compile_definitions(ninfer_block_dump PRIVATE NINFER_MODEL_TESTING=1)
target_include_directories(ninfer_block_dump PRIVATE ${PROJECT_SOURCE_DIR}/src
                                                  ${PROJECT_SOURCE_DIR}/third_party)

add_executable(ninfer_layer_dump ${PROJECT_SOURCE_DIR}/tools/parity/layer_dump.cpp)
target_link_libraries(ninfer_layer_dump PRIVATE ninfer_core)
target_compile_definitions(ninfer_layer_dump PRIVATE NINFER_MODEL_TESTING=1)
target_include_directories(ninfer_layer_dump PRIVATE ${PROJECT_SOURCE_DIR}/src
                                                  ${PROJECT_SOURCE_DIR}/third_party)

ninfer_add_test(ninfer_silu_mul_test SOURCES kernels/test_silu_mul.cpp)
ninfer_add_test(ninfer_residual_add_test SOURCES kernels/test_residual_add.cpp)
ninfer_add_test(ninfer_sigmoid_mul_test SOURCES kernels/test_sigmoid_mul.cpp)
ninfer_add_test(ninfer_rmsnorm_test SOURCES kernels/test_rmsnorm.cpp)
ninfer_add_test(ninfer_l2norm_test SOURCES kernels/test_l2norm.cpp)
ninfer_add_test(ninfer_gdn_gating_test SOURCES kernels/test_gdn_gating.cpp)
ninfer_add_test(ninfer_gdn_gating_proj_test SOURCES kernels/test_gdn_gating_proj.cpp)
ninfer_add_test(ninfer_gated_delta_rule_test SOURCES kernels/test_gated_delta_rule.cpp)
ninfer_add_test(ninfer_causal_conv1d_silu_test SOURCES kernels/test_causal_conv1d_silu.cpp)
ninfer_add_test(ninfer_rope_test SOURCES kernels/test_rope.cpp)
ninfer_add_test(ninfer_vision_elementwise_test SOURCES kernels/test_vision_elementwise.cpp)
ninfer_add_test(ninfer_layer_norm_test SOURCES kernels/test_layer_norm.cpp)
ninfer_add_test(ninfer_vision_pos_embed_test SOURCES kernels/test_vision_pos_embed.cpp)
ninfer_add_test(ninfer_vision_attention_test SOURCES kernels/test_vision_attention.cpp)
ninfer_add_test(ninfer_vision_support_test SOURCES test_vision_support.cpp)
target_include_directories(ninfer_vision_support_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
ninfer_add_test(ninfer_gqa_attention_test SOURCES kernels/test_gqa_attention.cpp)
target_include_directories(ninfer_gqa_attention_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
ninfer_add_test(ninfer_embedding_test SOURCES kernels/test_embedding.cpp)
ninfer_add_test(ninfer_argmax_test SOURCES kernels/test_argmax.cpp)
ninfer_add_test(ninfer_sampling_test SOURCES kernels/test_sampling.cpp)
ninfer_add_test(ninfer_linear_test SOURCES kernels/test_linear.cpp)
ninfer_add_test(ninfer_mtp_pack_test SOURCES kernels/test_mtp_pack.cpp)
ninfer_add_test(ninfer_mtp_round_test SOURCES kernels/test_mtp_round.cpp)
target_include_directories(ninfer_mtp_pack_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(ninfer_mtp_round_test PRIVATE ${PROJECT_SOURCE_DIR}/src)
