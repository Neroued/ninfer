ninfer_add_op_test(ninfer_normalization_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/l2norm.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/layer_norm.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/rmsnorm_pack_tail.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/rmsnorm.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/gated_rmsnorm.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/fused.cpp"
  LIBRARIES ninfer_ops)
