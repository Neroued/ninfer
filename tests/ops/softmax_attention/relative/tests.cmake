ninfer_add_op_test(ninfer_relative_attention_prepare_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prepare.cpp"
  LIBRARIES ninfer_ops
  FORMATS bf16 fp8)

if(NINFER_TARGET STREQUAL "rtx5090")
  ninfer_add_op_test(ninfer_relative_softmax_attention_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_attention.cpp"
    LIBRARIES ninfer_ops
    FORMATS bf16 fp8)
  ninfer_add_op_test(ninfer_relative_attention_chain_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_chain.cpp"
    LIBRARIES ninfer_ops
    FORMATS bf16 fp8)
endif()
