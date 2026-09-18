target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/linear_topk.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8_m64.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8_m64.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_m64.cu"
  "${CMAKE_CURRENT_LIST_DIR}/merge.cu"
)
