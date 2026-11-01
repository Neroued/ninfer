target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/linear_pair.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n1024_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n1024_k5120.cu"
)
