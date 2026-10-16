set(ninfer_op_tests
  add_bias
  gelu
  silu_mul
  residual_add
  sum_contributions
  sigmoid_mul
  rmsnorm_pack_tail
  l2norm
  kimi_delta_attention
  causal_conv1d_silu
  layer_norm
  embedding
  argmax
  rope
  embedding_interpolate_add
  sampling
  scalar
  cast
  prepare_ragged_prefix
  scatter
  scatter_bf16_batch
  target_logprobs
  position)
foreach(op IN LISTS ninfer_op_tests)
  ninfer_add_op_test(ninfer_${op}_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_${op}.cpp"
    LIBRARIES ninfer_ops)
endforeach()


ninfer_add_op_test(ninfer_candidate_selector_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_candidate_selector.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_softmax_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/causal_cache.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/plain_and_packed.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/context.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_sliding_window_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sliding_window_attention.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_kv_cache_append_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_kv_cache_append.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_rmsnorm_rope_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_rmsnorm_rope.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_context_kv_materialize_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_context_kv_materialize.cpp"
  LIBRARIES ninfer_ops)

add_test(NAME ninfer_kv_cache_append_nvfp4_test
  COMMAND ninfer_kv_cache_append_test --nvfp4-only)

add_test(NAME ninfer_kv_cache_append_k8v4_test
  COMMAND ninfer_kv_cache_append_test --k8v4-only)

set_tests_properties(
  ninfer_kv_cache_append_nvfp4_test
  ninfer_kv_cache_append_k8v4_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_op_test(ninfer_prepare_masked_block_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prepare_masked_block.cpp"
  LIBRARIES ninfer_ops)

include("${CMAKE_CURRENT_LIST_DIR}/moe/tests.cmake")

ninfer_add_op_test(ninfer_mtp_pack_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_pack.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_mtp_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_round.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_speculative_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_round.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_dynamic_grouped_conv_prepare_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dynamic_grouped_conv_prepare.cpp"
  LIBRARIES ninfer_ops)

include("${CMAKE_CURRENT_LIST_DIR}/linear_dynamic_grouped_conv_add/tests.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/linear/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/qsa/tests.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/gdn_input_proj/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ple/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/gated_delta_net/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rmsnorm/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_control/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gated_residual/tests.cmake")
