ninfer_add_op_test(ninfer_kv_cache_append_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/append_main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/append_full.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/append_prefix.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_context_kv_materialize_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize.cpp"
  LIBRARIES ninfer_ops)
