ninfer_add_op_test(ninfer_relative_attention_prepare_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prepare.cpp"
  LIBRARIES ninfer_ops
  FORMATS bf16 fp8)
