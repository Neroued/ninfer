target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/append/kv_cache_append.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/append/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/append/nvfp4_launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/append/k8v4_launch.cu"
)
