function(ninfer_add_op_bench name)
  cmake_parse_arguments(arg "" "" "SOURCES" ${ARGN})
  add_executable(${name} ${arg_SOURCES})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ops
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(${name} PRIVATE ninfer_ops)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

ninfer_add_op_bench(ninfer_silu_mul_bench SOURCES ops/silu_mul_bench.cu)
ninfer_add_op_bench(ninfer_residual_add_bench SOURCES ops/residual_add_bench.cu)
ninfer_add_op_bench(ninfer_sigmoid_mul_bench SOURCES ops/sigmoid_mul_bench.cu)
ninfer_add_op_bench(ninfer_rmsnorm_bench SOURCES ops/rmsnorm_bench.cu)
ninfer_add_op_bench(ninfer_l2norm_bench SOURCES ops/l2norm_bench.cu)
ninfer_add_op_bench(ninfer_gdn_gating_bench SOURCES ops/gdn_gating_bench.cu)
ninfer_add_op_bench(ninfer_gdn_gating_proj_bench SOURCES ops/gdn_gating_proj_bench.cu)
ninfer_add_op_bench(ninfer_gated_delta_net_bench SOURCES ops/gated_delta_net_bench.cu)
ninfer_add_op_bench(ninfer_gdn_replay_bench SOURCES ops/gdn_replay_bench.cu)
ninfer_add_op_bench(ninfer_gdn_layer_bench SOURCES ops/gdn_layer_bench.cu)
ninfer_add_op_bench(ninfer_rope_bench SOURCES ops/rope_bench.cu)
ninfer_add_op_bench(ninfer_add_bias_bench SOURCES ops/add_bias_bench.cu)
ninfer_add_op_bench(ninfer_gelu_bench SOURCES ops/gelu_bench.cu)
ninfer_add_op_bench(ninfer_layer_norm_bench SOURCES ops/layer_norm_bench.cu)
ninfer_add_op_bench(ninfer_vision_pos_embed_bench SOURCES ops/vision_pos_embed_bench.cu)
ninfer_add_op_bench(ninfer_scatter_bench SOURCES ops/scatter_bench.cu)
ninfer_add_op_bench(ninfer_packed_softmax_attention_bench
  SOURCES ops/packed_softmax_attention_bench.cu)
ninfer_add_op_bench(ninfer_embedding_bench SOURCES ops/embedding_bench.cu)
ninfer_add_op_bench(ninfer_position_bench SOURCES ops/position_bench.cu)
ninfer_add_op_bench(ninfer_cast_bench SOURCES ops/cast_bench.cu)
ninfer_add_op_bench(ninfer_argmax_bench SOURCES ops/argmax_bench.cu)
ninfer_add_op_bench(ninfer_causal_conv1d_silu_bench SOURCES ops/causal_conv1d_silu_bench.cu)
ninfer_add_op_bench(ninfer_linear_bench SOURCES ops/linear_bench.cu)
ninfer_add_op_bench(ninfer_output_stage_bench SOURCES ops/output_stage_bench.cu)
ninfer_add_op_bench(ninfer_gdn_input_proj_bench SOURCES ops/gdn_input_proj_bench.cu)
ninfer_add_op_bench(ninfer_gdn_input_proj_conv_snapshot_bench
  SOURCES ops/gdn_input_proj_conv_snapshot_bench.cu)
ninfer_add_op_bench(ninfer_attn_input_proj_bench SOURCES ops/attn_input_proj_bench.cu)
ninfer_add_op_bench(ninfer_w8_linear_swiglu_bench SOURCES ops/w8_linear_swiglu_bench.cu)
ninfer_add_op_bench(ninfer_nvfp4_linear_swiglu_bench SOURCES ops/nvfp4_linear_swiglu_bench.cu)
ninfer_add_op_bench(ninfer_fp8_linear_swiglu_bench SOURCES ops/fp8_linear_swiglu_bench.cu)
ninfer_add_op_bench(ninfer_q4_linear_swiglu_bench SOURCES ops/q4_linear_swiglu_bench.cu)
ninfer_add_op_bench(ninfer_w8_linear_add_bench SOURCES ops/w8_linear_add_bench.cu)
ninfer_add_op_bench(ninfer_q5_linear_add_bench SOURCES ops/q5_linear_add_bench.cu)
ninfer_add_op_bench(ninfer_bf16_linear_add_bench SOURCES ops/bf16_linear_add_bench.cu)
ninfer_add_op_bench(ninfer_nvfp4_linear_add_bench SOURCES ops/nvfp4_linear_add_bench.cu)
ninfer_add_op_bench(ninfer_fp8_linear_add_bench SOURCES ops/fp8_linear_add_bench.cu)
ninfer_add_op_bench(ninfer_causal_softmax_attention_bench
  SOURCES ops/causal_softmax_attention_bench.cu)
ninfer_add_op_bench(ninfer_context_softmax_attention_bench
  SOURCES ops/context_softmax_attention_bench.cu)
ninfer_add_op_bench(ninfer_sliding_window_attention_bench
  SOURCES ops/sliding_window_attention_bench.cu)
ninfer_add_op_bench(ninfer_kv_cache_append_bench SOURCES ops/kv_cache_append_bench.cu)
ninfer_add_op_bench(ninfer_prepare_masked_block_bench
  SOURCES ops/prepare_masked_block_bench.cu)
ninfer_add_op_bench(ninfer_sampling_select_bench SOURCES ops/sampling_select_bench.cu)
ninfer_add_op_bench(ninfer_sparse_moe_bench SOURCES ops/sparse_moe_bench.cu)
ninfer_add_op_bench(ninfer_linear_pair_bench SOURCES ops/linear_pair_bench.cu)
ninfer_add_op_bench(ninfer_mtp_pack_bench SOURCES ops/mtp_pack_bench.cu)
ninfer_add_op_bench(ninfer_proposal_remap_bench SOURCES ops/proposal_remap_bench.cu)

# End-to-end product benchmark: public Engine API and native .ninfer artifacts only.
add_executable(ninfer_bench targets/qwen3_6_27b/ninfer_bench.cpp targets/qwen3_6_27b/ninfer_bench_support.cpp)
target_include_directories(ninfer_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_compile_definitions(ninfer_bench PRIVATE NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(ninfer_bench PRIVATE ninfer_engine CUDA::cudart)

# Static context-cost calibrator. Transfer measurements call production Paged-KV copy primitives,
# then release their synthetic fixtures before the public Engine optionally loads one real artifact
# for text/Vision prefill measurements.
add_executable(ninfer_context_cost_bench
  context_cost/context_cost_bench.cpp
  context_cost/context_cost_measure.cpp
  context_cost/qwen3_6_context_fixture.cpp)
target_include_directories(ninfer_context_cost_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/context_cost
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export)
target_compile_definitions(ninfer_context_cost_bench PRIVATE
  NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(ninfer_context_cost_bench PRIVATE
  ninfer_engine ninfer_core CUDA::cudart)

# Native target-private MTP proposal/verification microbenchmark. It enters through the target
# Program facade and executes the production round schedule over a real .ninfer artifact.
add_executable(ninfer_qwen3_6_27b_mtp_round_bench
  targets/qwen3_6_27b/target_mtp_round_bench.cpp)
target_include_directories(ninfer_qwen3_6_27b_mtp_round_bench PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
target_link_libraries(ninfer_qwen3_6_27b_mtp_round_bench PRIVATE
  ninfer_engine ninfer_core)

# Complete production DFlash steady round: confirmed-context append, next proposal, target
# verification/acceptance, and host publication over a real 35B artifact.
add_executable(ninfer_qwen3_6_35b_a3b_dflash_round_bench
  targets/qwen3_6_35b_a3b/dflash_round_bench.cpp)
target_include_directories(ninfer_qwen3_6_35b_a3b_dflash_round_bench PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
target_link_libraries(ninfer_qwen3_6_35b_a3b_dflash_round_bench PRIVATE
  ninfer_engine ninfer_core)
