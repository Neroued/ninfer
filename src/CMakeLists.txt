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
  core/layout.cpp
  core/tensor.cpp
  core/decode_graph.cpp
  runtime/engine/product_cookie.cpp
  runtime/engine/public_types.cpp
  runtime/engine/request_memory.cpp)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC CUDA::cudart)

# Generic .ninfer framing, binding, and direct final materialization.
add_library(ninfer_artifact STATIC
  artifact/reader.cpp
  artifact/storage_layouts.cpp
  artifact/binder.cpp
  artifact/materializer.cpp)
ninfer_internal_includes(ninfer_artifact)
target_link_libraries(ninfer_artifact PUBLIC ninfer_core)

# Shared mathematical operators. The list is intentionally explicit: adding a
# source is a build-boundary decision, not an accidental recursive-glob side effect.
add_library(ninfer_kernels STATIC
  kernels/add_bias/add_bias.cpp
  kernels/add_bias/launch.cu
  kernels/argmax/argmax.cpp
  kernels/argmax/launch.cu
  kernels/causal_conv1d/causal_conv1d_silu.cpp
  kernels/causal_conv1d/launch.cu
  kernels/embedding/embedding.cpp
  kernels/embedding/launch.cu
  kernels/gated_delta_rule/gated_delta_rule.cpp
  kernels/gated_delta_rule/chunk_output.cu
  kernels/gated_delta_rule/chunked.cu
  kernels/gated_delta_rule/prepare_wy_wu.cu
  kernels/gated_delta_rule/recurrent.cu
  kernels/gated_delta_rule/state_passing.cu
  kernels/gelu/gelu.cpp
  kernels/gelu/launch.cu
  kernels/l2norm/l2norm.cpp
  kernels/l2norm/launch.cu
  kernels/layer_norm/layer_norm.cpp
  kernels/layer_norm/launch.cu
  kernels/residual_add/residual_add.cpp
  kernels/residual_add/launch.cu
  kernels/rmsnorm/rmsnorm.cpp
  kernels/rmsnorm/launch.cu
  kernels/scatter/scatter.cpp
  kernels/scatter/launch.cu
  kernels/sigmoid_mul/sigmoid_mul.cpp
  kernels/sigmoid_mul/launch.cu
  kernels/silu_mul/silu_mul.cpp
  kernels/silu_mul/launch.cu
  )
ninfer_internal_includes(ninfer_kernels)
ninfer_cuda_archive(ninfer_kernels)
target_link_libraries(ninfer_kernels PUBLIC ninfer_core)

# Product-independent text and media mechanisms.
add_library(ninfer_text STATIC
  text/unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c)
ninfer_internal_includes(ninfer_text)

add_library(ninfer_media_decode STATIC
  media/decode/decode.cpp)
ninfer_internal_includes(ninfer_media_decode)
target_link_libraries(ninfer_media_decode PRIVATE PkgConfig::FFMPEG)

# Product-only path/data/HTTP acquisition. No target package links this library.
add_library(ninfer_media_acquire STATIC
  product/media_acquire/acquire.cpp)
ninfer_internal_includes(ninfer_media_acquire)
target_link_libraries(ninfer_media_acquire PRIVATE PkgConfig::LIBCURL)

# Owning command-line message input shared by product entry points and target diagnostics.
add_library(ninfer_product_prompt_input STATIC
  product/prompt_input/prompt_input.cpp)
ninfer_internal_includes(ninfer_product_prompt_input)
target_link_libraries(ninfer_product_prompt_input PRIVATE ninfer_media_acquire)

# The first closed exact-checkpoint / exact-GPU package owns its complete source and include graph.
add_subdirectory(targets/qwen3_6_27b_rtx5090)

# One cold closed registry plus the opaque public product Engine.
add_library(ninfer_engine STATIC
  targets/registry.cpp
  runtime/engine/engine.cpp)
ninfer_internal_includes(ninfer_engine)
target_include_directories(ninfer_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ninfer_engine
  PRIVATE ninfer_qwen3_6_27b_rtx5090 ninfer_artifact ninfer_core)
add_library(ninfer::engine ALIAS ninfer_engine)

# HTTP protocol/product adapter. No serving header is part of the installed API.
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
  PRIVATE ninfer_media_acquire)
