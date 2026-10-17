ninfer_add_op_test(ninfer_reduction_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/argmax.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/target_logprobs.cpp"
  LIBRARIES ninfer_ops)
