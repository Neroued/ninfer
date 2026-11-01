ninfer_add_op_test(ninfer_dynamic_grouped_conv_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/prepare.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/linear_add.cpp"
  LIBRARIES ninfer_ops)
