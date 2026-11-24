ninfer_add_op_test(ninfer_gdn_control_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/test_control.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/fixtures.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/oracle.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/cases.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/runner.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/validation.cpp"
  LIBRARIES ninfer_ops)
