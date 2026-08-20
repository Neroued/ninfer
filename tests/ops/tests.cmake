set(ninfer_op_tests
  add_bias
  gelu
  silu_mul
  residual_add
  sigmoid_mul
  rmsnorm
  rmsnorm_pack_tail
  gated_rmsnorm
  l2norm
  gated_delta_net
  kimi_delta_attention
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

ninfer_add_op_test(ninfer_linear_topk_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_topk.cu"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_candidate_selector_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_candidate_selector.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_softmax_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/causal_cache.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/plain_and_packed.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/softmax_attention/context.cpp"
  LIBRARIES ninfer_ops)

add_test(NAME ninfer_softmax_attention_nvfp4_test
  COMMAND ninfer_softmax_attention_test --nvfp4-only)

add_test(NAME ninfer_softmax_attention_k8v4_test
  COMMAND ninfer_softmax_attention_test --k8v4-only)

set_tests_properties(
  ninfer_softmax_attention_nvfp4_test
  ninfer_softmax_attention_k8v4_test
  PROPERTIES SKIP_RETURN_CODE 77)

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

ninfer_add_op_test(ninfer_sparse_moe_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sparse_moe.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_mtp_pack_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_pack.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_mtp_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_round.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_speculative_round_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_speculative_round.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_attn_input_proj_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attn_input_proj.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_input_proj_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_dynamic_grouped_conv_prepare_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dynamic_grouped_conv_prepare.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_linear_dynamic_grouped_conv_add_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_linear_dynamic_grouped_conv_add.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_input_proj_conv_snapshot_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj_conv_snapshot.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_input_proj_conv_record_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_input_proj_conv_record.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gated_delta_net_replay_record_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gated_delta_net_replay_record.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_gdn_replay_fold_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gdn_replay_fold.cpp"
  LIBRARIES ninfer_ops)

include("${CMAKE_CURRENT_LIST_DIR}/linear/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/tests.cmake")
