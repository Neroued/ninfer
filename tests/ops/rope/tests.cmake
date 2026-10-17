ninfer_add_op_test(ninfer_rope_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/rope.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope.cpp"
  LIBRARIES ninfer_ops)
