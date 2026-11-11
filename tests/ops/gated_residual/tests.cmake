ninfer_add_op_test(ninfer_gated_residual_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/norm.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/fixtures.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/oracle.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/validation.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/mix/bf16/cases.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/mix/bf16/fixtures.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/mix/bf16/oracle.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/mix/bf16/validation.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/mix/fp8/fixtures.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/mix/fp8/oracle.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/mix/fp8/cases.cpp"
  LIBRARIES ninfer_ops)
