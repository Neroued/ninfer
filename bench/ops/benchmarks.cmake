ninfer_add_op_bench(ninfer_silu_mul_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/silu_mul_bench.cu")
ninfer_add_op_bench(ninfer_residual_add_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/residual_add_bench.cu")
ninfer_add_op_bench(ninfer_sigmoid_mul_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/sigmoid_mul_bench.cu")
ninfer_add_op_bench(ninfer_rmsnorm_pack_tail_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/rmsnorm_pack_tail_bench.cu")
ninfer_add_op_bench(ninfer_l2norm_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/l2norm_bench.cu")
ninfer_add_op_bench(ninfer_dynamic_grouped_conv_prepare_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv_prepare_bench.cu")
ninfer_add_op_bench(ninfer_linear_dynamic_grouped_conv_add_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/linear_dynamic_grouped_conv_add_bench.cu")
ninfer_add_op_bench(ninfer_kimi_delta_attention_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/kimi_delta_attention_bench.cu")
ninfer_add_op_bench(ninfer_rope_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/rope_bench.cu")
ninfer_add_op_bench(ninfer_rmsnorm_rope_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope_bench.cu")
ninfer_add_op_bench(ninfer_add_bias_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/add_bias_bench.cu")
ninfer_add_op_bench(ninfer_gelu_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/gelu_bench.cu")
ninfer_add_op_bench(ninfer_layer_norm_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/layer_norm_bench.cu")
ninfer_add_op_bench(ninfer_embedding_interpolate_add_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/embedding_interpolate_add_bench.cu")
ninfer_add_op_bench(ninfer_scatter_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/scatter_bench.cu")
ninfer_add_op_bench(ninfer_scatter_bf16_batch_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/scatter_bf16_batch_bench.cu")
ninfer_add_op_bench(ninfer_packed_softmax_attention_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/packed_softmax_attention_bench.cu")
ninfer_add_op_bench(ninfer_prepare_ragged_prefix_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/prepare_ragged_prefix_bench.cu")
ninfer_add_op_bench(ninfer_embedding_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/embedding_bench.cu")
ninfer_add_op_bench(ninfer_position_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/position_bench.cu")
ninfer_add_op_bench(ninfer_cast_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/cast_bench.cu")
ninfer_add_op_bench(ninfer_argmax_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/argmax_bench.cu")
ninfer_add_op_bench(ninfer_vocabulary_logprobs_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/vocabulary_logprobs_bench.cu")
ninfer_add_op_bench(ninfer_target_logprobs_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/target_logprobs_bench.cu")
ninfer_add_op_bench(ninfer_causal_conv1d_silu_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/causal_conv1d_silu_bench.cu")
ninfer_add_op_bench(ninfer_linear_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/linear_bench.cu")
ninfer_add_op_bench(ninfer_linear_topk_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/linear_topk_bench.cu")
ninfer_add_op_bench(ninfer_candidate_selector_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/candidate_selector_bench.cu")
ninfer_add_op_bench(ninfer_linear_split_multi_output_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/linear/split/multi_output.cu")
ninfer_add_op_bench(ninfer_linear_split_two_output_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/linear/split/two_output.cu")
ninfer_add_op_bench(ninfer_q8_linear_swiglu_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/q8_linear_swiglu_bench.cu")
ninfer_add_op_bench(ninfer_nvfp4_linear_swiglu_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/nvfp4_linear_swiglu_bench.cu")
ninfer_add_op_bench(ninfer_fp8_linear_swiglu_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/fp8_linear_swiglu_bench.cu")
ninfer_add_op_bench(ninfer_q4_linear_swiglu_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/q4_linear_swiglu_bench.cu")
ninfer_add_op_bench(ninfer_q8_linear_add_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/q8_linear_add_bench.cu")
ninfer_add_op_bench(ninfer_q5_linear_add_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/q5_linear_add_bench.cu")
ninfer_add_op_bench(ninfer_bf16_linear_add_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/bf16_linear_add_bench.cu")
ninfer_add_op_bench(ninfer_nvfp4_linear_add_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/nvfp4_linear_add_bench.cu")
ninfer_add_op_bench(ninfer_fp8_linear_add_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/fp8_linear_add_bench.cu")
ninfer_add_op_bench(ninfer_causal_softmax_attention_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/causal_softmax_attention_bench.cu")
ninfer_add_op_bench(ninfer_context_softmax_attention_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/context_softmax_attention_bench.cu")
ninfer_add_op_bench(ninfer_sliding_window_attention_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/sliding_window_attention_bench.cu")
ninfer_add_op_bench(ninfer_kv_cache_append_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/kv_cache_append_bench.cu")
ninfer_add_op_bench(ninfer_prepare_masked_block_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/prepare_masked_block_bench.cu")
ninfer_add_op_bench(ninfer_vocabulary_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/vocabulary_bench.cu")
ninfer_add_op_bench(ninfer_sampling_select_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/sampling_select_bench.cu")
ninfer_add_op_bench(ninfer_moe_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/moe/main.cu")
ninfer_add_op_bench(ninfer_moe_block_scaled_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/moe/block_scaled.cu")
if(NINFER_TARGET STREQUAL "rtx5090")
  ninfer_add_op_bench(ninfer_moe_cpu_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/moe/cpu.cpp")
  target_link_libraries(ninfer_moe_cpu_bench PRIVATE ninfer_cpu_executor)
endif()
ninfer_add_op_bench(ninfer_linear_pair_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/linear_pair_bench.cu")
ninfer_add_op_bench(ninfer_context_kv_materialize_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize_bench.cu")
ninfer_add_op_bench(ninfer_mtp_pack_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/mtp_pack_bench.cu")
ninfer_add_op_bench(ninfer_proposal_remap_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/proposal_remap_bench.cu")

include("${CMAKE_CURRENT_LIST_DIR}/gdn_input_proj/benchmarks.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ple/benchmarks.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/qsa/benchmarks.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/gated_delta_net/benchmarks.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rmsnorm/benchmarks.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_control/benchmarks.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gated_residual/benchmarks.cmake")

ninfer_add_op_bench(ninfer_moe_ep_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/moe/ep.cu"
          "${PROJECT_SOURCE_DIR}/tests/ops/moe/ep_fixture.cpp")
target_include_directories(ninfer_moe_ep_bench PRIVATE "${PROJECT_SOURCE_DIR}/tests")
# The encoded fixture's independent FP64 oracle remains ordinary strict C++.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(ninfer_moe_ep_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
    $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
endif()

ninfer_add_op_bench(ninfer_linear_swiglu_clamped_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/linear_swiglu_clamped.cu")

ninfer_add_op_bench(ninfer_swiglu_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/swiglu_bench.cu")
ninfer_add_op_bench(ninfer_sigmoid_route_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/sigmoid_route_bench.cu")
ninfer_add_op_bench(ninfer_moe_clamped_ep_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/moe/clamped_ep.cu"
          "${PROJECT_SOURCE_DIR}/tests/ops/moe/clamped_ep_fixture.cpp")
target_include_directories(ninfer_moe_clamped_ep_bench PRIVATE "${PROJECT_SOURCE_DIR}/tests")
target_compile_definitions(ninfer_moe_clamped_ep_bench PRIVATE NINFER_CLAMPED_EP_BENCH=1)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(ninfer_moe_clamped_ep_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math;-ffp-contract=off>)
endif()
