ninfer_add_op_test(ninfer_embedding_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/embedding.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/interpolate_add.cpp"
  LIBRARIES ninfer_ops)
