target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dispatch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16.cu"
)
