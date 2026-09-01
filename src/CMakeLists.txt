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
  core/host_worker_pool.cpp
  core/gdn_replay_records.cpp
  core/cyclic_kv_cache.cpp
  core/host_kv_arena.cpp
  core/paged_kv_cache.cpp
  core/linear_attention_state.cpp
  core/layout.cpp
  core/tensor.cpp
  core/decode_graph.cpp
  runtime/engine/admission_policy.cpp
  runtime/engine/context_cost.cpp
  runtime/engine/context_cost_defaults.cpp
  runtime/engine/kv_capacity.cpp
  runtime/engine/public_types.cpp)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC CUDA::cudart CUDA::nvtx3 Threads::Threads)

# Generic .ninfer framing, binding, and direct final materialization.
add_library(ninfer_artifact STATIC
  artifact/reader.cpp
  artifact/storage_layouts.cpp
  artifact/binder.cpp
  artifact/materializer.cpp
  artifact/typed_binding.cpp)
ninfer_internal_includes(ninfer_artifact)
target_link_libraries(ninfer_artifact PUBLIC ninfer_core)

# The warp-specialized SM120 kernel uses setmaxnreg to transfer registers from its
# producer warpgroup to its consumer warpgroups. Relocatable device code makes ptxas
# discard that contract, so this self-contained kernel is intentionally compiled in a
# non-RDC archive and called through a host launcher from ninfer_ops.
add_library(ninfer_nvfp4_tma STATIC
  ops/linear/nvfp4/nvfp4_w4a4_tma.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu)
ninfer_internal_includes(ninfer_nvfp4_tma)
set_target_properties(ninfer_nvfp4_tma PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
target_compile_options(ninfer_nvfp4_tma PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(ninfer_nvfp4_tma PRIVATE ninfer_core CUDA::cudart CUDA::cuda_driver)

# Shared mathematical Ops. The list is intentionally explicit: adding a
# source is a build-boundary decision, not an accidental recursive-glob side effect.
add_library(ninfer_ops STATIC
  ops/launcher/add_bias.cu
  ops/launcher/argmax.cu
  ops/launcher/cast.cu
  ops/launcher/causal_conv1d.cu
  ops/launcher/embed_gather.cu
  ops/launcher/gdn_gating.cu
  ops/launcher/gelu.cu
  ops/launcher/l2norm.cu
  ops/launcher/layer_norm.cu
  ops/launcher/mtp_pack.cu
  ops/launcher/mtp_round.cu
  ops/launcher/speculative_round.cu
  ops/launcher/position.cu
  ops/launcher/prepare_ragged_prefix.cu
  ops/launcher/prepare_masked_block.cu
  ops/launcher/residual_add.cu
  ops/launcher/rmsnorm.cu
  ops/launcher/rope.cu
  ops/launcher/sampling.cu
  ops/launcher/scalar.cu
  ops/launcher/scatter.cu
  ops/launcher/sigmoid_gate_mul.cu
  ops/launcher/silu_and_mul.cu
  ops/launcher/target_logprobs.cu
  ops/launcher/vision_pos_embed.cu
  # Softmax Attention and KV-cache state transitions.
  ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp
  ops/softmax_attention/dense/causal_cache/small_t.cu
  ops/softmax_attention/dense/causal_cache/small_t_fp8.cu
  ops/softmax_attention/dense/causal_cache/prompt.cu
  ops/softmax_attention/dense/causal_cache/prompt_fp8.cu
  ops/softmax_attention/dense/packed/packed_softmax_attention.cpp
  ops/softmax_attention/dense/packed/launch.cu
  ops/softmax_attention/dense/context/context_softmax_attention.cpp
  ops/softmax_attention/dense/context/launch.cu
  ops/softmax_attention/sliding_window/sliding_window_attention.cpp
  ops/softmax_attention/sliding_window/launch.cu
  ops/kv_cache/append/kv_cache_append.cpp
  ops/kv_cache/append/launch.cu
  # Gated DeltaNet Linear Attention core.
  ops/linear_attention/gated_delta_net/gated_delta_net.cpp
  ops/linear_attention/gated_delta_net/replay.cpp
  ops/linear_attention/gated_delta_net/recurrent.cu
  ops/linear_attention/gated_delta_net/chunked/launch.cu
  ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cu
  ops/linear_attention/gated_delta_net/chunked/state_passing.cu
  ops/linear_attention/gated_delta_net/chunked/output.cu
  ops/attn_input_proj/bf16/bf16_attn_input_decode.cu
  ops/attn_input_proj/bf16/bf16_attn_input_gemm_mma.cu
  ops/attn_input_proj/bf16/bf16_attn_input_small_t.cu
  ops/attn_input_proj/bf16/bf16_attn_input_plan.cpp
  ops/attn_input_proj/fp8/fp8_attn_input_decode.cu
  ops/attn_input_proj/fp8/fp8_attn_input_small_t.cu
  ops/attn_input_proj/fp8/fp8_attn_input_a8.cu
  ops/attn_input_proj/fp8/fp8_attn_input_plan.cpp
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_decode.cu
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_small_t.cu
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_w4a4.cu
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_gemm_mma.cu
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_small_t.cu
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp
  ops/attn_input_proj/w8/w8_attn_input_decode.cu
  ops/attn_input_proj/w8/w8_attn_input_gemm_mma.cu
  ops/attn_input_proj/w8/w8_attn_input_gemm_simt.cu
  ops/attn_input_proj/w8/w8_attn_input_gemm_splitk.cu
  ops/attn_input_proj/w8/w8_attn_input_plan.cpp
  ops/gdn_input_proj/fp8/fp8_gdn_input_decode.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_small_t.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_plan.cpp
  ops/gdn_input_proj/fp8/fp8_gdn_conv_fused.cu
  ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.cpp
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_decode.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_small_t.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_w4a4.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_decode.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_small_t.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_post.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_gemm_mma.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp
  ops/gdn_input_proj/gdn_projected_conv.cu
  ops/gdn_input_proj/w8/w8_gdn_input_decode.cu
  ops/gdn_input_proj/w8/w8_gdn_input_gemm_mma.cu
  ops/gdn_input_proj/w8/w8_gdn_input_gemm_splitk.cu
  ops/gdn_input_proj/w8/w8_gdn_input_plan.cpp
  ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu
  ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp
  ops/linear/bf16/bf16_gemv.cu
  ops/linear/bf16/bf16_gemm_mma.cu
  ops/linear/bf16/bf16_small_t.cu
  ops/linear/bf16/bf16_dispatch.cpp
  ops/linear/fp8/fp8_format.cpp
  ops/linear/fp8/fp8_gemv.cu
  ops/linear/fp8/fp8_small_t.cu
  ops/linear/fp8/fp8_a16_small_t_mma.cu
  ops/linear/fp8/fp8_a16_gemm_mma.cu
  ops/linear/fp8/fp8_a8.cu
  ops/linear/fp8/fp8_dispatch.cpp
  ops/linear/nvfp4/nvfp4_format.cpp
  ops/linear/nvfp4/nvfp4_gemv.cu
  ops/linear/nvfp4/nvfp4_small_t.cu
  ops/linear/nvfp4/nvfp4_w4a4.cu
  ops/linear/nvfp4/nvfp4_dispatch.cpp
  ops/linear/q4/q4_small_t_mma.cu
  ops/linear/q4/q4_rowsplit_gemm_mma.cu
  ops/linear/q4/q4_rowsplit_gemm_simt.cu
  ops/linear/q4/q4_rowsplit_gemv.cu
  ops/linear/q4/q4_dispatch.cpp
  ops/linear/q5/q5_dispatch.cpp
  ops/linear/q5/q5_rowsplit_gemm_mma.cu
  ops/linear/q5/q5_rowsplit_gemm_simt.cu
  ops/linear/q5/q5_rowsplit_gemv.cu
  ops/linear_add/bf16/bf16_linear_add_decode.cu
  ops/linear_add/bf16/bf16_linear_add_gemm_mma.cu
  ops/linear_add/bf16/bf16_linear_add_small_t.cu
  ops/linear_add/bf16/bf16_linear_add_plan.cpp
  ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu
  ops/linear_add/nvfp4/nvfp4_linear_add_small_t.cu
  ops/linear_add/nvfp4/nvfp4_linear_add_w4a4.cu
  ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp
  ops/linear_add/fp8/fp8_linear_add_decode.cu
  ops/linear_add/fp8/fp8_linear_add_small_t.cu
  ops/linear_add/fp8/fp8_linear_add_a8.cu
  ops/linear_add/fp8/fp8_linear_add_plan.cpp
  ops/linear_add/q5/q5_linear_add_gemm_mma.cu
  ops/linear_add/q5/q5_linear_add_gemm_simt.cu
  ops/linear_add/q5/q5_linear_add_gemv.cu
  ops/linear_add/q5/q5_linear_add_plan.cpp
  ops/linear_add/w8/w8_linear_add_gemm_mma.cu
  ops/linear_add/w8/w8_linear_add_gemm_simt.cu
  ops/linear_add/w8/w8_linear_add_gemm_splitk.cu
  ops/linear_add/w8/w8_linear_add_plan.cpp
  ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma.cu
  ops/linear_swiglu/q4/q4_linear_swiglu_gemv.cu
  ops/linear_swiglu/q4/q4_linear_swiglu_plan.cpp
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_decode.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_small_t.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp
  ops/linear_swiglu/fp8/fp8_linear_swiglu_decode.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_small_t.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp
  ops/linear_swiglu/w8/w8_linear_swiglu_decode.cu
  ops/linear_swiglu/w8/w8_linear_swiglu_gemm_mma.cu
  ops/linear_swiglu/w8/w8_linear_swiglu_gemm_splitk.cu
  ops/linear_swiglu/w8/w8_linear_swiglu_plan.cpp
  ops/linear/q6/q6_dispatch.cpp
  ops/linear/q6/q6_rowsplit_gemm_mma.cu
  ops/linear/q6/q6_rowsplit_gemm_simt.cu
  ops/linear/w8/w8_rowsplit_gemm_mma.cu
  ops/linear/w8/w8_rowsplit_gemm_simt.cu
  ops/linear/w8/w8_rowsplit_gemm_decode.cu
  ops/linear/w8/w8_small_t.cu
  ops/linear/w8/w8_rowsplit_gemm_splitk.cu
  ops/linear/w8/w8_dispatch.cpp
  ops/linear_pair/w8/w8_pair_decode.cu
  ops/linear_pair/w8/w8_pair_gemm_concat.cu
  ops/linear_pair/w8/w8_pair_gemm_mma.cu
  ops/linear_pair/w8/w8_pair_gemm_splitk.cu
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
  ops/wrapper/gdn_gating.cpp
  ops/wrapper/gdn_gating_proj.cpp
  ops/wrapper/gdn_input_proj.cpp
  ops/wrapper/gelu.cpp
  ops/wrapper/attn_input_proj.cpp
  ops/wrapper/l2norm.cpp
  ops/wrapper/layer_norm.cpp
  ops/wrapper/linear_add.cpp
  ops/wrapper/linear_pair.cpp
  ops/wrapper/linear_swiglu.cpp
  ops/wrapper/mtp_pack.cpp
  ops/wrapper/mtp_round.cpp
  ops/wrapper/speculative_round.cpp
  ops/wrapper/position.cpp
  ops/wrapper/prepare_ragged_prefix.cpp
  ops/wrapper/prepare_masked_block.cpp
  ops/wrapper/residual_add.cpp
  ops/wrapper/rmsnorm.cpp
  ops/wrapper/rope.cpp
  ops/wrapper/sampling.cpp
  ops/wrapper/scalar.cpp
  ops/wrapper/scatter.cpp
  ops/wrapper/sigmoid_mul.cpp
  ops/wrapper/silu_mul.cpp
  ops/wrapper/sparse_moe.cpp
  ops/wrapper/target_logprobs.cpp
  ops/wrapper/vision_pos_embed.cpp)
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  # Volta-only execution leaves. Shared Op contracts and planners remain in the
  # ordinary source list above; these files provide the sm_70 implementations.
  target_sources(ninfer_ops PRIVATE
    ops/attn_input_proj/fp8/fp8_attn_input_volta_qpn.cu
    ops/attn_input_proj/fp8/fp8_attn_input_cutlass_sm70.cu
    ops/attn_input_proj/bf16/bf16_attn_input_cutlass_sm70.cu
    ops/attn_input_proj/q4_q5/q4_q5_attn_input_cutlass_sm70.cu
    ops/gdn_input_proj/fp8/fp8_gdn_input_volta_qpn.cu
    ops/gdn_input_proj/fp8/fp8_gdn_input_cutlass_sm70.cu
    ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_cutlass_sm70.cu
    ops/gdn_input_proj/w8/w8_gdn_input_cutlass_sm70.cu
    ops/linear/fp8/fp8_volta_qpn_gemm.cu
    ops/linear/fp8/fp8_cutlass_sm70.cu
    ops/linear/bf16/bf16_cutlass_sm70.cu
    ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cu
    ops/linear/nvfp4/nvfp4_volta_mma_gemm.cu
    ops/linear/nvfp4/nvfp4_cutlass_sm70.cu
    ops/linear/nvfp4/nvfp4_prepack_sm70.cu
    ops/linear/q4/q4_volta_mma_gemm.cu
    ops/linear/q4/q4_volta_qpn_gemm.cu
    ops/linear/q5/q5_volta_mma_gemm.cu
    ops/linear/w8/w8_volta_mma_gemm.cu
    ops/linear/w8/w8_volta_qpn_gemm.cu
    ops/linear_add/q5/q5_linear_add_cutlass_sm70.cu
    ops/linear_add/bf16/bf16_linear_add_volta.cu
    ops/linear_swiglu/fp8/fp8_linear_swiglu_volta_qpn.cu
    ops/linear_swiglu/fp8/fp8_linear_swiglu_qpn_split.cu
    ops/linear_swiglu/fp8/fp8_linear_swiglu_cutlass_sm70.cu
    ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_volta_qpn.cu
    ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_qpn_split.cu
    ops/linear_swiglu/q4/q4_linear_swiglu_cutlass_sm70.cu
    ops/launcher/gqa_attention_volta_flash.cu)
  set_source_files_properties(
    ops/launcher/gqa_attention_volta_flash.cu
    TARGET_DIRECTORY ninfer_ops
    PROPERTIES
      INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/third_party/llama_cpp_fattn"
      COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-diag-suppress=940>")
  target_link_libraries(ninfer_ops PRIVATE nvidia::cutlass::cutlass)
  target_compile_options(ninfer_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_archive(ninfer_ops)
target_link_libraries(ninfer_ops PUBLIC ninfer_core PRIVATE ninfer_nvfp4_tma)

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

if(NINFER_BUILD_APPS OR BUILD_TESTING)
  add_library(ninfer_product_load_progress STATIC
    product/load_progress/load_progress.cpp)
  ninfer_internal_includes(ninfer_product_load_progress)
endif()

# One cold closed registry plus the opaque public product Engine. Family and exact-target
# directories contribute their sources directly to this single compile owner below.
add_library(ninfer_engine STATIC
  targets/registry.cpp
  runtime/contract/sampling.cpp
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
    serve/http_transport.cpp
    serve/request_validation.cpp
    serve/openai_chat_request.cpp
    serve/openai_chat_response.cpp
    serve/openai_chat_http.cpp
    serve/openai_common.cpp
    serve/openai_responses_request.cpp
    serve/openai_responses_state.cpp
    serve/openai_responses_response.cpp
    serve/anthropic_thinking_signature.cpp
    serve/anthropic_messages_request.cpp
    serve/anthropic_messages_response.cpp
    serve/anthropic_messages_http.cpp
    serve/openai_responses_store.cpp
    serve/translate.cpp
    serve/console_log.cpp
    serve/generation_service.cpp
    serve/request_log.cpp
    serve/http_server.cpp
    serve/openai_responses_http.cpp
    serve/serve_options.cpp)
  ninfer_internal_includes(ninfer_serve)
  target_include_directories(ninfer_serve PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  target_link_libraries(ninfer_serve
    PUBLIC ninfer_engine Threads::Threads
    PRIVATE ninfer_media_acquire CUDA::cudart)
endif()
