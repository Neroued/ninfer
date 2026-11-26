target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/linear_pair.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n1024_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n1024_k5120.cu"
)

if(NINFER_TARGET STREQUAL "rtx5090")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/bf16/rtx5090/shapes/n32_k2304.cu"
    "${CMAKE_CURRENT_LIST_DIR}/bf16/rtx5090/shapes/n1024_k2304.cu"
  )
endif()
