ninfer_add_op_test(ninfer_elementwise_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/add_bias.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/gelu.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/silu_mul.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/residual_add.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/sum_contributions.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/sigmoid_mul.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/scalar.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/cast.cpp"
  LIBRARIES ninfer_ops)
