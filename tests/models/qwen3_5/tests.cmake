ninfer_add_test(ninfer_qwen3_5_loading_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_real.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen3_5_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp"
  LIBRARIES ninfer_model_loading)

ninfer_add_test(ninfer_qwen3_5_runtime_mechanisms_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_runtime_mechanisms.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_state_image_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_state_image_layout_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state_image_layout.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_qwen3_5_native_transactions_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_native_transactions.cpp"
  LIBRARIES ninfer_model_execution ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen3_5_mtp_history_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_history_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_model_loading ninfer_core)

# Packed prefill shares one Main traversal between sequences. The test replaces the three mixer Ops
# (to feed the FP64 oracles) and the device allocator (to prove execution never allocates).
ninfer_add_test(ninfer_qwen3_5_packed_prefill_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_packed_prefill_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_model_loading ninfer_core ninfer_op_test_support)
ninfer_op_oracle_options(ninfer_qwen3_5_packed_prefill_real_test)
ninfer_test_interpose(ninfer_qwen3_5_packed_prefill_real_test
  _ZN6ninfer3ops24causal_softmax_attentionERKNS_6TensorES3_S3_S3_S3_S3_NS0_21AttentionHeadGeometryEfNS_21PagedKVBatchLayerViewENS0_32CausalAttentionExecutionEnvelopeERNS_11MemoryArenaERS1_NS_17CudaExecutionViewE
  _ZN6ninfer3ops15gated_delta_netERKNS_6TensorES3_S3_S3_S3_fbRNS_11MemoryArenaES3_RS1_S6_NS_17CudaExecutionViewE
  _ZN6ninfer3ops24causal_conv1d_silu_splitERKNS_6TensorES3_S3_RS1_S4_S4_S4_NS_17CudaExecutionViewE
  cudaMalloc cudaMallocAsync)

ninfer_add_test(ninfer_qwen3_5_prefix_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_prefix_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine)
ninfer_test_interpose(ninfer_qwen3_5_prefix_real_test cudaEventQuery)

add_test(NAME ninfer_qwen3_5_terminal_publication_real_test
  COMMAND ninfer_qwen3_5_prefix_real_test)
set_tests_properties(ninfer_qwen3_5_terminal_publication_real_test PROPERTIES
  ENVIRONMENT "NINFER_PREFIX_REAL_SCENARIO=terminal-publication")

ninfer_add_test(ninfer_qwen3_5_preemption_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_preemption_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_fairness_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_fairness_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_grammar_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_grammar_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_tools_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_tools_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine ninfer::json)

ninfer_add_test(ninfer_qwen3_5_score_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_score_real.cpp"
  LIBRARIES ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_vision_workspace_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision_workspace.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_dflash2_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash2_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine)

ninfer_add_test(ninfer_qwen3_5_dflash_prefill_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_dflash_prefill_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_model_loading ninfer_core)

ninfer_add_test(ninfer_qwen3_5_moe_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_moe_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine)

add_test(NAME ninfer_qwen3_5_moe_offload_real_test
  COMMAND ninfer_qwen3_5_moe_real_test --offload)
set_tests_properties(ninfer_qwen3_5_moe_offload_real_test PROPERTIES
  ENVIRONMENT "NINFER_VRAM_BUDGET_MIB=16384")

ninfer_add_test(ninfer_qwen3_5_dflash_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_engine_dflash_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_engine)

add_test(NAME ninfer_qwen3_5_agent_continuation_real_test
  COMMAND ninfer_qwen3_5_prefix_real_test)
set_tests_properties(ninfer_qwen3_5_agent_continuation_real_test PROPERTIES
  ENVIRONMENT "NINFER_PREFIX_REAL_SCENARIO=agent-continuation")

# A real-model test owns the single GPU while its artifact is resident.
set(ninfer_qwen3_5_real_tests
  ninfer_qwen3_5_loading_real_test
  ninfer_qwen3_5_native_transactions_test
  ninfer_qwen3_5_mtp_history_real_test
  ninfer_qwen3_5_packed_prefill_real_test
  ninfer_qwen3_5_prefix_real_test
  ninfer_qwen3_5_terminal_publication_real_test
  ninfer_qwen3_5_agent_continuation_real_test
  ninfer_qwen3_5_preemption_real_test
  ninfer_qwen3_5_fairness_real_test
  ninfer_qwen3_5_grammar_real_test
  ninfer_qwen3_5_tools_real_test
  ninfer_qwen3_5_score_real_test
  ninfer_qwen3_5_vision_workspace_test
  ninfer_qwen3_5_dflash2_real_test
  ninfer_qwen3_5_dflash_prefill_real_test
  ninfer_qwen3_5_moe_real_test
  ninfer_qwen3_5_moe_offload_real_test
  ninfer_qwen3_5_dflash_real_test)
set_tests_properties(${ninfer_qwen3_5_real_tests} PROPERTIES
  SKIP_RETURN_CODE 77
  RUN_SERIAL TRUE
  LABELS "gpu;real")

set_tests_properties(
  ninfer_qwen3_5_state_image_test
  PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")

ninfer_add_test(ninfer_qwen3_5_t3_capture_nonblocking_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_t3_capture_nonblocking.cpp"
  LIBRARIES ninfer_model_execution ninfer_model_loading ninfer_core)
set_tests_properties(ninfer_qwen3_5_t3_capture_nonblocking_test
  PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu;real")
