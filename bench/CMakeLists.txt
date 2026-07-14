function(ninfer_add_kernel_bench name)
  cmake_parse_arguments(arg "" "" "SOURCES" ${ARGN})
  add_executable(${name} ${arg_SOURCES})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src)
  target_link_libraries(${name} PRIVATE ninfer_kernels)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

function(ninfer_add_target_kernel_bench name source)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b_rtx5090/impl)
  target_link_libraries(${name} PRIVATE ninfer_qwen3_6_27b_rtx5090 ninfer_core)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

ninfer_add_kernel_bench(ninfer_silu_mul_bench SOURCES kernels/silu_mul_bench.cu)
ninfer_add_kernel_bench(ninfer_residual_add_bench SOURCES kernels/residual_add_bench.cu)
ninfer_add_kernel_bench(ninfer_sigmoid_mul_bench SOURCES kernels/sigmoid_mul_bench.cu)
ninfer_add_kernel_bench(ninfer_rmsnorm_bench SOURCES kernels/rmsnorm_bench.cu)
ninfer_add_kernel_bench(ninfer_l2norm_bench SOURCES kernels/l2norm_bench.cu)
ninfer_add_target_kernel_bench(ninfer_gdn_gating_bench kernels/gdn_gating_bench.cu)
ninfer_add_target_kernel_bench(ninfer_gdn_gating_proj_bench kernels/gdn_gating_proj_bench.cu)
ninfer_add_kernel_bench(ninfer_gated_delta_rule_bench SOURCES kernels/gated_delta_rule_bench.cu)
ninfer_add_target_kernel_bench(ninfer_rope_bench kernels/rope_bench.cu)
ninfer_add_kernel_bench(ninfer_add_bias_bench SOURCES kernels/add_bias_bench.cu)
ninfer_add_kernel_bench(ninfer_gelu_bench SOURCES kernels/gelu_bench.cu)
ninfer_add_kernel_bench(ninfer_layer_norm_bench SOURCES kernels/layer_norm_bench.cu)
ninfer_add_target_kernel_bench(ninfer_vision_pos_embed_bench kernels/vision_pos_embed_bench.cu)
ninfer_add_kernel_bench(ninfer_scatter_bench SOURCES kernels/scatter_bench.cu)
ninfer_add_target_kernel_bench(ninfer_vision_attention_bench kernels/vision_attention_bench.cu)
ninfer_add_kernel_bench(ninfer_embedding_bench SOURCES kernels/embedding_bench.cu)
ninfer_add_kernel_bench(ninfer_argmax_bench SOURCES kernels/argmax_bench.cu)
ninfer_add_kernel_bench(ninfer_causal_conv1d_silu_bench SOURCES kernels/causal_conv1d_silu_bench.cu)
ninfer_add_target_kernel_bench(ninfer_linear_bench kernels/linear_bench.cu)
ninfer_add_target_kernel_bench(ninfer_linear_op_bench kernels/linear_op_bench.cu)

ninfer_add_target_kernel_bench(
  ninfer_gqa_attention_bench
  targets/qwen3_6_27b_rtx5090/gqa_attention_bench.cu)
ninfer_add_target_kernel_bench(
  ninfer_sampling_select_bench
  targets/qwen3_6_27b_rtx5090/sampling_select_bench.cu)

# End-to-end product benchmark: public Engine API and native .ninfer artifacts only.
add_executable(ninfer_bench targets/qwen3_6_27b_rtx5090/ninfer_bench.cpp targets/qwen3_6_27b_rtx5090/ninfer_bench_support.cpp)
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
  targets/qwen3_6_27b_rtx5090/target_mtp_round_bench.cpp)
target_include_directories(ninfer_qwen3_6_27b_mtp_round_bench PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/src
  ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(ninfer_qwen3_6_27b_mtp_round_bench PRIVATE
  ninfer_qwen3_6_27b_rtx5090 ninfer_core)
