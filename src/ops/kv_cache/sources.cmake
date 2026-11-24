target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/append/kv_cache_append.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/append/dispatch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kernels/write.cu"
)

include("${CMAKE_CURRENT_LIST_DIR}/context_materialize/sources.cmake")
