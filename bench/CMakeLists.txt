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
ninfer_add_op_bench(ninfer_gated_delta_rule_bench SOURCES ops/gated_delta_rule_bench.cu)
ninfer_add_op_bench(ninfer_gdn_layer_bench SOURCES ops/gdn_layer_bench.cu)
ninfer_add_op_bench(ninfer_attention_layer_bench SOURCES ops/attention_layer_bench.cu)
ninfer_add_op_bench(ninfer_rope_bench SOURCES ops/rope_bench.cu)
ninfer_add_op_bench(ninfer_add_bias_bench SOURCES ops/add_bias_bench.cu)
ninfer_add_op_bench(ninfer_gelu_bench SOURCES ops/gelu_bench.cu)
ninfer_add_op_bench(ninfer_layer_norm_bench SOURCES ops/layer_norm_bench.cu)
ninfer_add_op_bench(ninfer_vision_pos_embed_bench SOURCES ops/vision_pos_embed_bench.cu)
ninfer_add_op_bench(ninfer_scatter_bench SOURCES ops/scatter_bench.cu)
ninfer_add_op_bench(ninfer_vision_attention_bench SOURCES ops/vision_attention_bench.cu)
ninfer_add_op_bench(ninfer_embedding_bench SOURCES ops/embedding_bench.cu)
ninfer_add_op_bench(ninfer_position_bench SOURCES ops/position_bench.cu)
ninfer_add_op_bench(ninfer_cast_bench SOURCES ops/cast_bench.cu)
ninfer_add_op_bench(ninfer_argmax_bench SOURCES ops/argmax_bench.cu)
ninfer_add_op_bench(ninfer_causal_conv1d_silu_bench SOURCES ops/causal_conv1d_silu_bench.cu)
ninfer_add_op_bench(ninfer_linear_op_bench SOURCES ops/linear_op_bench.cu)
ninfer_add_op_bench(ninfer_output_stage_bench SOURCES ops/output_stage_bench.cu)
ninfer_add_op_bench(ninfer_input_proj_bench SOURCES ops/input_proj_bench.cu)
ninfer_add_op_bench(ninfer_w8_input_proj_bench SOURCES ops/w8_input_proj_bench.cu)
ninfer_add_op_bench(ninfer_gqa_attention_bench SOURCES ops/gqa_attention_bench.cu)
ninfer_add_op_bench(ninfer_sampling_select_bench SOURCES ops/sampling_select_bench.cu)
ninfer_add_op_bench(ninfer_sparse_moe_bench SOURCES ops/sparse_moe_bench.cu)

# End-to-end product benchmark: public Engine API and native .ninfer artifacts only.
add_executable(ninfer_bench targets/qwen3_6_27b/ninfer_bench.cpp targets/qwen3_6_27b/ninfer_bench_support.cpp)
target_include_directories(ninfer_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_compile_definitions(ninfer_bench PRIVATE NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(ninfer_bench PRIVATE ninfer_engine CUDA::cudart)

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

# Target-side speculative transaction benchmark for the 35B target. Drafts are injected so the
# measurement contains shared target verification/acceptance but no MTP or future dFlash proposal.
add_executable(ninfer_qwen3_6_35b_a3b_target_speculative_round_bench
  targets/qwen3_6_35b_a3b/target_speculative_round_bench.cpp)
target_include_directories(ninfer_qwen3_6_35b_a3b_target_speculative_round_bench PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_35b_a3b/export
  ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6/export)
target_link_libraries(ninfer_qwen3_6_35b_a3b_target_speculative_round_bench PRIVATE
  ninfer_engine ninfer_core)
