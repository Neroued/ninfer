find_package(Python3 REQUIRED COMPONENTS Interpreter)

# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the installed product headers stand alone.
add_executable(ninfer_public_api_test test_public_api.cpp)
target_include_directories(ninfer_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME ninfer_public_api_test COMMAND ninfer_public_api_test)

function(ninfer_add_test name)
  cmake_parse_arguments(arg "NEEDS_SOURCE_DIR" "" "SOURCES;LIBRARIES" ${ARGN})
  add_executable(${name} ${arg_SOURCES})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
  if(arg_LIBRARIES)
    target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
  else()
    target_link_libraries(${name} PRIVATE ninfer_core)
  endif()
  if(arg_NEEDS_SOURCE_DIR)
    target_compile_definitions(${name} PRIVATE
      NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
      NINFER_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

function(ninfer_add_op_test name)
  ninfer_add_test(${name} ${ARGN})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${name} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
      $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
endfunction()

function(ninfer_add_linear_test name source)
  ninfer_add_op_test(${name}
    SOURCES ${source}
            ops/linear/linear_test_common.cpp
    LIBRARIES ninfer_ops)
endfunction()

function(ninfer_add_fused_linear_test name source common)
  ninfer_add_op_test(${name}
    SOURCES ${source} ${common}
    LIBRARIES ninfer_ops)
endfunction()

# Artifact framing and model-independent runtime contracts.
ninfer_add_test(ninfer_artifact_reader_test
  SOURCES test_ninfer_artifact_reader.cpp
  LIBRARIES ninfer_artifact)
ninfer_add_test(ninfer_artifact_materialization_test
  SOURCES test_artifact_materialization.cpp
  LIBRARIES ninfer_artifact)
ninfer_add_test(ninfer_media_decode_test
  SOURCES test_media_decode.cpp
  LIBRARIES ninfer_media_decode)
ninfer_add_test(ninfer_device_test       SOURCES test_device.cpp)
ninfer_add_test(ninfer_decode_graph_test SOURCES test_decode_graph.cpp)
ninfer_add_test(ninfer_tensor_test       SOURCES test_tensor.cpp)
ninfer_add_test(ninfer_arena_test        SOURCES test_arena.cpp)
ninfer_add_test(ninfer_admission_policy_test SOURCES test_admission_policy.cpp)
ninfer_add_test(ninfer_context_cost_test SOURCES test_context_cost.cpp)
ninfer_add_test(ninfer_resource_manager_test SOURCES test_resource_manager.cpp)
ninfer_add_test(ninfer_kv_capacity_test SOURCES test_kv_capacity.cpp)
ninfer_add_test(ninfer_sampling_defaults_test
  SOURCES test_sampling_defaults.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_sampling_defaults_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export)
ninfer_add_test(ninfer_perplexity_evaluation_test
  SOURCES test_perplexity_evaluation.cpp
          ${PROJECT_SOURCE_DIR}/apps/perplexity/evaluation.cpp)
target_include_directories(ninfer_perplexity_evaluation_test PRIVATE
  ${PROJECT_SOURCE_DIR}/apps/perplexity)
ninfer_add_test(ninfer_cli_options_test
  SOURCES test_cli_options.cpp ${PROJECT_SOURCE_DIR}/apps/cli/options.cpp)
target_include_directories(ninfer_cli_options_test PRIVATE ${PROJECT_SOURCE_DIR}/apps/cli)
ninfer_add_test(ninfer_kv_cache_test
  SOURCES test_kv_cache.cpp
  LIBRARIES ninfer_core)
ninfer_add_test(ninfer_state_store_test
  SOURCES test_state_store.cpp
  LIBRARIES ninfer_core)
ninfer_add_test(ninfer_gdn_replay_records_test
  SOURCES test_gdn_replay_records.cpp
  LIBRARIES ninfer_core)
set_tests_properties(
  ninfer_artifact_materialization_test
  ninfer_device_test
  ninfer_decode_graph_test
  ninfer_arena_test
  ninfer_kv_cache_test
  ninfer_state_store_test
  PROPERTIES SKIP_RETURN_CODE 77)
# Family frontend and one opt-in real-artifact prefix/MTP integration test.
ninfer_add_test(ninfer_qwen3_6_frontend_test
  SOURCES targets/qwen3_6/test_frontend.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_frontend_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/impl)
ninfer_add_test(ninfer_qwen3_6_runtime_mechanisms_test
  SOURCES targets/qwen3_6/test_runtime_mechanisms.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_runtime_mechanisms_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_qwen3_6_state_image_test
  SOURCES targets/qwen3_6/test_state_image.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_state_image_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
set_tests_properties(ninfer_qwen3_6_state_image_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_state_image_layout_test
  SOURCES targets/qwen3_6/test_state_image_layout.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_state_image_layout_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
ninfer_add_test(ninfer_qwen3_6_context_store_test
  SOURCES targets/qwen3_6/test_context_store.cpp
  LIBRARIES ninfer_engine ninfer_core)
target_include_directories(ninfer_qwen3_6_context_store_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
set_tests_properties(ninfer_qwen3_6_context_store_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_27b_prefix_real_test
  SOURCES targets/qwen3_6_27b/test_engine_prefix_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_27b_prefix_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_27b_score_real_test
  SOURCES targets/qwen3_6_27b/test_engine_score_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_27b_score_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_27b_load_plan_test
  SOURCES targets/qwen3_6_27b/test_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_qwen3_6_27b_load_plan_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export)
target_link_libraries(ninfer_qwen3_6_27b_load_plan_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_6_27b_load_plan_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_real_test
  SOURCES targets/qwen3_6_35b_a3b/test_engine_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_35b_a3b_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_dflash_real_test
  SOURCES targets/qwen3_6_35b_a3b/test_engine_dflash_real.cpp
  LIBRARIES ninfer_engine)
set_tests_properties(ninfer_qwen3_6_35b_a3b_dflash_real_test PROPERTIES SKIP_RETURN_CODE 77)
ninfer_add_test(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test
  SOURCES targets/qwen3_6_35b_a3b/test_dflash_load_plan.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test PRIVATE
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export)
target_link_libraries(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test PRIVATE CUDA::cudart)
set_tests_properties(ninfer_qwen3_6_35b_a3b_dflash_load_plan_test
  PROPERTIES SKIP_RETURN_CODE 77)

# Product protocol contracts.
ninfer_add_test(ninfer_openai_schema_test
  SOURCES test_openai_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_openai_responses_test
  SOURCES test_openai_responses.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_openai_responses_store_test
  SOURCES test_openai_responses_store.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_anthropic_schema_test
  SOURCES test_anthropic_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_tool_call_parser_test
  SOURCES test_tool_call_parser.cpp
  LIBRARIES ninfer_engine)
ninfer_add_test(ninfer_serve_options_test
  SOURCES test_serve_options.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_request_log_test
  SOURCES test_request_log.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_host_timing_test
  SOURCES test_host_timing.cpp
  LIBRARIES ninfer_core)
ninfer_add_test(ninfer_http_error_handler_test
  SOURCES test_http_error_handler.cpp
  LIBRARIES ninfer_serve)
target_include_directories(ninfer_http_error_handler_test PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
ninfer_add_test(ninfer_http_transport_test
  SOURCES test_http_transport.cpp
  LIBRARIES ninfer_serve)
target_include_directories(ninfer_http_transport_test PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
ninfer_add_test(ninfer_bench_support_test
  SOURCES test_ninfer_bench_support.cpp
          ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_6_27b/ninfer_bench_support.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_bench_support_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_6_27b)

# Pure host-side qualification of the fixed-dimension non-negative fit and the exact coefficient
# quantization used by the context-cost calibrator. It deliberately links no Engine/CUDA target.
add_executable(ninfer_context_cost_measure_test
  test_context_cost_measure.cpp
  ${PROJECT_SOURCE_DIR}/bench/context_cost/context_cost_measure.cpp)
target_include_directories(ninfer_context_cost_measure_test PRIVATE
  ${PROJECT_SOURCE_DIR}/bench/context_cost)
add_test(NAME ninfer_context_cost_measure_test COMMAND ninfer_context_cost_measure_test)

# Shared Op numerical tests. Target-composed schedule tests are registered separately below.
set(ninfer_op_tests
  add_bias
  gelu
  silu_mul
  residual_add
  sigmoid_mul
  rmsnorm
  gated_rmsnorm
  l2norm
  gated_delta_net
  causal_conv1d_silu
  layer_norm
  embedding
  argmax
  gdn_gating
  gdn_gating_proj
  rope
  vision_pos_embed
  sampling
  scalar
  cast
  scatter
  target_logprobs
  position)
foreach(op IN LISTS ninfer_op_tests)
  ninfer_add_op_test(ninfer_${op}_test
    SOURCES ops/test_${op}.cpp
    LIBRARIES ninfer_ops)
endforeach()

# Stateful, fused, and exact-shape Ops remain central even when the current implementation domain
# is one checkpoint/device.
ninfer_add_op_test(ninfer_softmax_attention_test
  SOURCES ops/softmax_attention/main.cpp
          ops/softmax_attention/causal_cache.cpp
          ops/softmax_attention/plain_and_packed.cpp
          ops/softmax_attention/context.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_sliding_window_attention_test
  SOURCES ops/test_sliding_window_attention.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_kv_cache_append_test
  SOURCES ops/test_kv_cache_append.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_prepare_masked_block_test
  SOURCES ops/test_prepare_masked_block.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_sparse_moe_test
  SOURCES ops/test_sparse_moe.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_mtp_pack_test
  SOURCES ops/test_mtp_pack.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_mtp_round_test
  SOURCES ops/test_mtp_round.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_speculative_round_test
  SOURCES ops/test_speculative_round.cpp
  LIBRARIES ninfer_ops)
ninfer_add_linear_test(ninfer_linear_q4_a16_test ops/linear/test_q4_a16.cpp)
ninfer_add_linear_test(ninfer_linear_q5_a16_test ops/linear/test_q5_a16.cpp)
ninfer_add_linear_test(ninfer_linear_q6_a16_test ops/linear/test_q6_a16.cpp)
ninfer_add_linear_test(ninfer_linear_w8_a16_test ops/linear/test_w8_a16.cpp)
ninfer_add_linear_test(ninfer_linear_nvfp4_a16_test ops/linear/test_nvfp4_a16.cpp)
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
ninfer_add_linear_test(ninfer_linear_nvfp4_a4_test ops/linear/test_nvfp4_a4.cpp)
endif()
ninfer_add_linear_test(ninfer_linear_fp8_a16_test ops/linear/test_fp8_a16.cpp)
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
ninfer_add_linear_test(ninfer_linear_fp8_a8_test ops/linear/test_fp8_a8.cpp)
endif()
ninfer_add_op_test(ninfer_linear_bf16_a16_test
  SOURCES ops/linear/test_bf16_a16.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_attn_input_proj_test
  SOURCES ops/test_attn_input_proj.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_test
  SOURCES ops/test_gdn_input_proj.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_conv_snapshot_test
  SOURCES ops/test_gdn_input_proj_conv_snapshot.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_input_proj_conv_record_test
  SOURCES ops/test_gdn_input_proj_conv_record.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gated_delta_net_replay_record_test
  SOURCES ops/test_gated_delta_net_replay_record.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_replay_fold_test
  SOURCES ops/test_gdn_replay_fold.cpp
  LIBRARIES ninfer_ops)
ninfer_add_fused_linear_test(
  ninfer_linear_add_q5_a16_test
  ops/linear_add/test_q5_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_add_bf16_a16_test
  ops/linear_add/test_bf16_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_add_w8_a16_test
  ops/linear_add/test_w8_a16.cpp
  ops/linear_add/linear_add_test_common.cpp)
ninfer_add_op_test(ninfer_linear_add_nvfp4_test
  SOURCES ops/linear_add/test_nvfp4.cpp
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_linear_add_fp8_test
  SOURCES ops/linear_add/test_fp8.cpp
  LIBRARIES ninfer_ops)
ninfer_add_fused_linear_test(
  ninfer_linear_pair_w8_a16_test
  ops/linear_pair/test_w8_a16.cpp
  ops/linear_pair/linear_pair_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_q4_a16_test
  ops/linear_swiglu/test_q4_a16.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_w8_a16_test
  ops/linear_swiglu/test_w8_a16.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_nvfp4_test
  ops/linear_swiglu/test_nvfp4.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)
ninfer_add_fused_linear_test(
  ninfer_linear_swiglu_fp8_test
  ops/linear_swiglu/test_fp8.cpp
  ops/linear_swiglu/linear_swiglu_test_common.cpp)

ninfer_add_test(ninfer_qwen3_6_27b_visual_scatter_test
  SOURCES targets/qwen3_6_27b/test_visual_scatter.cpp
  LIBRARIES ninfer_engine ninfer_core)
set_tests_properties(ninfer_qwen3_6_27b_visual_scatter_test PROPERTIES SKIP_RETURN_CODE 77)

# These tests deliberately compile against the exact target's internal contracts; the target does
# not export impl/ to ordinary consumers.
set(ninfer_qwen3_6_27b_internal_tests
  ninfer_qwen3_6_27b_visual_scatter_test)
foreach(test_target IN LISTS ninfer_qwen3_6_27b_internal_tests)
  target_include_directories(${test_target} PRIVATE
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/impl
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
endforeach()
