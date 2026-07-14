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

# Artifact framing and model-independent runtime contracts.
ninfer_add_test(ninfer_artifact_reader_test
  SOURCES test_ninfer_artifact_reader.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_artifact)
ninfer_add_test(ninfer_device_test       SOURCES test_device.cpp)
ninfer_add_test(ninfer_tensor_test       SOURCES test_tensor.cpp)
ninfer_add_test(ninfer_weight_test       SOURCES test_weight.cpp)
ninfer_add_test(ninfer_arena_test        SOURCES test_arena.cpp)
ninfer_add_test(ninfer_kv_cache_test
  SOURCES test_kv_cache.cpp
  LIBRARIES ninfer_qwen3_6_27b_rtx5090 ninfer_core)
ninfer_add_test(ninfer_state_store_test
  SOURCES test_state_store.cpp
  LIBRARIES ninfer_qwen3_6_27b_rtx5090 ninfer_core)
ninfer_add_test(ninfer_generation_controller_test SOURCES test_generation_controller.cpp)
ninfer_add_test(ninfer_row_split_pack_golden_test
  SOURCES test_row_split_pack_golden.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_core)

# Exact-target frontend and one opt-in real-artifact prefix/MTP integration test.
ninfer_add_test(ninfer_qwen3_6_27b_frontend_test
  SOURCES targets/qwen3_6_27b/test_frontend.cpp
  LIBRARIES ninfer_qwen3_6_27b_rtx5090 ninfer_core)
ninfer_add_test(ninfer_qwen3_6_27b_prefix_real_test
  SOURCES targets/qwen3_6_27b/test_engine_prefix_real.cpp
  LIBRARIES ninfer_engine)

# Product protocol contracts.
ninfer_add_test(ninfer_openai_schema_test
  SOURCES test_openai_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_anthropic_schema_test
  SOURCES test_anthropic_schema.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_tool_call_parser_test
  SOURCES test_tool_call_parser.cpp
  LIBRARIES ninfer_serve)
ninfer_add_test(ninfer_bench_support_test
  SOURCES test_ninfer_bench_support.cpp
          ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_6_27b_rtx5090/ninfer_bench_support.cpp
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine)
target_include_directories(ninfer_bench_support_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/targets/qwen3_6_27b_rtx5090)

# Shared operator numerical tests. Target-composed schedule tests are registered
# separately below so the generic operator library does not acquire model policy.
set(ninfer_kernel_tests
  silu_mul
  residual_add
  sigmoid_mul
  rmsnorm
  l2norm
  gated_delta_rule
  causal_conv1d_silu
  layer_norm
  embedding
  argmax)
foreach(op IN LISTS ninfer_kernel_tests)
  ninfer_add_test(ninfer_${op}_test
    SOURCES kernels/test_${op}.cpp
    LIBRARIES ninfer_kernels)
endforeach()

# Operators whose implemented contract or dispatch names this exact checkpoint/GPU.
set(ninfer_qwen3_6_27b_operator_tests
  gdn_gating
  gdn_gating_proj
  rope
  vision_pos_embed
  vision_attention
  sampling)
foreach(op IN LISTS ninfer_qwen3_6_27b_operator_tests)
  ninfer_add_test(ninfer_${op}_test
    SOURCES kernels/test_${op}.cpp
    LIBRARIES ninfer_qwen3_6_27b_rtx5090 ninfer_core)
endforeach()

# These exercise fixed target schedule geometry in addition to target-owned kernels.
ninfer_add_test(ninfer_gqa_attention_test
  SOURCES targets/qwen3_6_27b/test_gqa_attention.cpp
  LIBRARIES ninfer_qwen3_6_27b_rtx5090 ninfer_core)
ninfer_add_test(ninfer_mtp_pack_test
  SOURCES targets/qwen3_6_27b/test_mtp_pack.cpp
  LIBRARIES ninfer_qwen3_6_27b_rtx5090 ninfer_core)
ninfer_add_test(ninfer_mtp_round_test
  SOURCES targets/qwen3_6_27b/test_mtp_round.cpp
  LIBRARIES ninfer_qwen3_6_27b_rtx5090 ninfer_core)
ninfer_add_test(ninfer_linear_test
  SOURCES targets/qwen3_6_27b/test_linear_kernels.cpp
  LIBRARIES ninfer_qwen3_6_27b_rtx5090 ninfer_core)
ninfer_add_test(ninfer_vision_elementwise_test
  SOURCES targets/qwen3_6_27b/test_vision_elementwise.cpp
  LIBRARIES ninfer_qwen3_6_27b_rtx5090 ninfer_core)

# These tests deliberately compile against the exact target's internal contracts; the target does
# not export impl/ to ordinary consumers.
set(ninfer_qwen3_6_27b_internal_tests
  ninfer_kv_cache_test
  ninfer_state_store_test
  ninfer_qwen3_6_27b_frontend_test
  ninfer_gdn_gating_test
  ninfer_gdn_gating_proj_test
  ninfer_rope_test
  ninfer_vision_pos_embed_test
  ninfer_vision_attention_test
  ninfer_sampling_test
  ninfer_gqa_attention_test
  ninfer_mtp_pack_test
  ninfer_mtp_round_test
  ninfer_linear_test
  ninfer_vision_elementwise_test)
foreach(test_target IN LISTS ninfer_qwen3_6_27b_internal_tests)
  target_include_directories(${test_target} PRIVATE
    ${PROJECT_SOURCE_DIR}/src/targets/qwen3_6_27b_rtx5090/impl)
endforeach()
