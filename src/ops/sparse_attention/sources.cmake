target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/sparse_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/staging.cu"
)
