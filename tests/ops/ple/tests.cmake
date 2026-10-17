ninfer_add_op_test(ninfer_ple_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/test_gate.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/test_conv.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/test_state_extract.cpp"
  LIBRARIES ninfer_ops)
