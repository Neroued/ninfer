target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/hyper_connection.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/elementwise.cu"
  "${CMAKE_CURRENT_LIST_DIR}/mix_small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/split.cu"
  "${CMAKE_CURRENT_LIST_DIR}/split_fused.cu"
  "${CMAKE_CURRENT_LIST_DIR}/up_collapse.cu"
)
