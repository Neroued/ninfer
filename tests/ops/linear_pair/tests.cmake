ninfer_add_op_test(ninfer_linear_pair_q8_a16_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q8_a16.cpp" "${CMAKE_CURRENT_LIST_DIR}/linear_pair_test_common.cpp"
  LIBRARIES ninfer_ops)
