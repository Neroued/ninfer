function(ninfer_internal_includes target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
endfunction()

function(ninfer_cuda_archive target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

# Device/tensor/layout/graph primitives and the model-independent request runtime.
add_library(ninfer_core STATIC
  core/arena.cu
  core/device.cu
  core/dtype.cpp
  core/kv_cache.cpp
  core/layout.cpp
  core/tensor.cpp
  core/decode_graph.cpp
  runtime/engine/public_types.cpp
  runtime/engine/request_memory.cpp)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC CUDA::cudart CUDA::nvtx3)

# Generic .ninfer framing, binding, and direct final materialization.
add_library(ninfer_artifact STATIC
  artifact/reader.cpp
  artifact/storage_layouts.cpp
  artifact/binder.cpp
  artifact/materializer.cpp
  artifact/typed_binding.cpp)
ninfer_internal_includes(ninfer_artifact)
target_link_libraries(ninfer_artifact PUBLIC ninfer_core)

# Shared mathematical Ops. The list is intentionally explicit: adding a
# source is a build-boundary decision, not an accidental recursive-glob side effect.
add_library(ninfer_ops STATIC
  ops/launcher/add_bias.cu
  ops/launcher/argmax.cu
  ops/launcher/cast.cu
  ops/launcher/causal_conv1d.cu
  ops/launcher/embed_gather.cu
  ops/launcher/gated_delta_rule_chunked.cu
  ops/launcher/gated_delta_rule_recurrent.cu
  ops/launcher/gdn_chunk_output.cu
  ops/launcher/gdn_gating.cu
  ops/launcher/gdn_prepare_wy_wu.cu
  ops/launcher/gdn_state_passing.cu
  ops/launcher/gelu.cu
  ops/launcher/gqa_attention_decode.cu
  ops/launcher/gqa_attention_prefill.cu
  ops/launcher/l2norm.cu
  ops/launcher/layer_norm.cu
  ops/launcher/mtp_pack.cu
  ops/launcher/mtp_round.cu
  ops/launcher/position.cu
  ops/launcher/residual_add.cu
  ops/launcher/rmsnorm.cu
  ops/launcher/rope.cu
  ops/launcher/sampling.cu
  ops/launcher/scalar.cu
  ops/launcher/scatter.cu
  ops/launcher/sigmoid_gate_mul.cu
  ops/launcher/silu_and_mul.cu
  ops/launcher/vision_attention.cu
  ops/launcher/vision_pos_embed.cu
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_gemm_mma.cu
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_small_t.cu
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp
  ops/attn_input_proj/w8/w8_attn_input_decode.cu
  ops/attn_input_proj/w8/w8_attn_input_gemm_mma.cu
  ops/attn_input_proj/w8/w8_attn_input_gemm_simt.cu
  ops/attn_input_proj/w8/w8_attn_input_plan.cpp
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_gemm_mma.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp
  ops/gdn_input_proj/w8/w8_gdn_input_decode.cu
  ops/gdn_input_proj/w8/w8_gdn_input_gemm_mma.cu
  ops/gdn_input_proj/w8/w8_gdn_input_gemm_splitk.cu
  ops/gdn_input_proj/w8/w8_gdn_input_plan.cpp
  ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu
  ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp
  ops/linear/bf16/bf16_contiguous_plan.cpp
  ops/linear/q4/q4_rowsplit_gemm_mma.cu
  ops/linear/q4/q4_rowsplit_gemm_simt.cu
  ops/linear/q4/q4_rowsplit_gemv.cu
  ops/linear/q4/q4_rowsplit_launch.cpp
  ops/linear/q4/q4_rowsplit_plan.cpp
  ops/linear/q5/q5_rowsplit_gemm_mma.cu
  ops/linear/q5/q5_rowsplit_gemm_simt.cu
  ops/linear/q5/q5_rowsplit_gemv.cu
  ops/linear/q5/q5_rowsplit_launch.cpp
  ops/linear/q5/q5_rowsplit_plan.cpp
  ops/linear_add/q5/q5_linear_add_gemm_mma.cu
  ops/linear_add/q5/q5_linear_add_gemv.cu
  ops/linear_add/q5/q5_linear_add_plan.cpp
  ops/linear_add/w8/w8_linear_add_gemm_mma.cu
  ops/linear_add/w8/w8_linear_add_gemm_simt.cu
  ops/linear_add/w8/w8_linear_add_plan.cpp
  ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma.cu
  ops/linear_swiglu/q4/q4_linear_swiglu_gemv.cu
  ops/linear_swiglu/q4/q4_linear_swiglu_plan.cpp
  ops/linear/q6/q6_rowsplit_launch.cpp
  ops/linear/q6/q6_rowsplit_gemm_mma.cu
  ops/linear/q6/q6_rowsplit_gemm_simt.cu
  ops/linear/q6/q6_rowsplit_plan.cpp
  ops/linear/w8/w8_rowsplit_gemm_mma.cu
  ops/linear/w8/w8_rowsplit_gemm_simt.cu
  ops/linear/w8/w8_rowsplit_launch.cpp
  ops/linear/w8/w8_rowsplit_plan.cpp
  ops/linear_pair/w8/w8_pair_gemm_mma.cu
  ops/linear_pair/w8/w8_pair_plan.cpp
  ops/sparse_moe/decode/sparse_moe_decode_kernels.cu
  ops/sparse_moe/decode/sparse_moe_decode_plan.cpp
  ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu
  ops/sparse_moe/prefill/sparse_moe_prefill_plan.cpp
  ops/sparse_moe/small_t/sparse_moe_small_t_kernels.cu
  ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp
  ops/linear/linear.cpp
  ops/wrapper/add_bias.cpp
  ops/wrapper/argmax.cpp
  ops/wrapper/cast.cpp
  ops/wrapper/causal_conv1d_silu.cpp
  ops/wrapper/embedding.cpp
  ops/wrapper/gated_delta_rule.cpp
  ops/wrapper/gdn_gating.cpp
  ops/wrapper/gdn_gating_proj.cpp
  ops/wrapper/gdn_input_proj.cpp
  ops/wrapper/gelu.cpp
  ops/wrapper/gqa_attention.cpp
  ops/wrapper/attn_input_proj.cpp
  ops/wrapper/l2norm.cpp
  ops/wrapper/layer_norm.cpp
  ops/wrapper/linear_add.cpp
  ops/wrapper/linear_pair.cpp
  ops/wrapper/linear_swiglu.cpp
  ops/wrapper/mtp_pack.cpp
  ops/wrapper/mtp_round.cpp
  ops/wrapper/position.cpp
  ops/wrapper/residual_add.cpp
  ops/wrapper/rmsnorm.cpp
  ops/wrapper/rope.cpp
  ops/wrapper/sampling.cpp
  ops/wrapper/scalar.cpp
  ops/wrapper/scatter.cpp
  ops/wrapper/sigmoid_mul.cpp
  ops/wrapper/silu_mul.cpp
  ops/wrapper/sparse_moe.cpp
  ops/wrapper/vision_attention.cpp
  ops/wrapper/vision_pos_embed.cpp)
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_archive(ninfer_ops)
target_link_libraries(ninfer_ops PUBLIC ninfer_core)

# Product-independent text and media mechanisms.
add_library(ninfer_text STATIC
  text/unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c)
ninfer_internal_includes(ninfer_text)

add_library(ninfer_media_decode STATIC
  media/decode/decode.cpp)
ninfer_internal_includes(ninfer_media_decode)
target_link_libraries(ninfer_media_decode PRIVATE PkgConfig::FFMPEG)

if(NINFER_BUILD_MEDIA_ACQUIRE)
  # Product-only path/data/HTTP acquisition. No target package links this library.
  add_library(ninfer_media_acquire STATIC
    product/media_acquire/acquire.cpp)
  ninfer_internal_includes(ninfer_media_acquire)
  target_link_libraries(ninfer_media_acquire PRIVATE PkgConfig::LIBCURL)
endif()

if(NINFER_BUILD_PROMPT_INPUT)
  # Owning command-line message input shared by product entry points and target diagnostics.
  add_library(ninfer_product_prompt_input STATIC
    product/prompt_input/prompt_input.cpp)
  ninfer_internal_includes(ninfer_product_prompt_input)
  target_link_libraries(ninfer_product_prompt_input PRIVATE ninfer_media_acquire)
endif()

# One cold closed registry plus the opaque public product Engine. Family and exact-target
# directories contribute their sources directly to this single compile owner below.
add_library(ninfer_engine STATIC
  targets/registry.cpp
  runtime/engine/engine.cpp)
ninfer_internal_includes(ninfer_engine)
target_include_directories(ninfer_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(targets/qwen3_6)
add_subdirectory(targets/qwen3_6_27b)
add_subdirectory(targets/qwen3_6_35b_a3b)

target_link_libraries(ninfer_engine
  PRIVATE
    ninfer_artifact
    ninfer_core
    ninfer_ops
    ninfer_text
    ninfer_media_decode)
add_library(ninfer::engine ALIAS ninfer_engine)

if(NINFER_BUILD_SERVE)
  # HTTP protocol/product adapter. No serving header is part of the public API.
  add_library(ninfer_serve STATIC
    serve/openai_schema.cpp
    serve/anthropic_schema.cpp
    serve/tool_call_parser.cpp
    serve/translate.cpp
    serve/generation_service.cpp
    serve/request_log.cpp
    serve/http_server.cpp
    serve/serve_options.cpp)
  ninfer_internal_includes(ninfer_serve)
  target_include_directories(ninfer_serve PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  target_link_libraries(ninfer_serve
    PUBLIC ninfer_engine Threads::Threads
    PRIVATE ninfer_media_acquire CUDA::cudart)
endif()
