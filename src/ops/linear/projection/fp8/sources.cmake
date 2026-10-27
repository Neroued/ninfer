target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/fp8_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n34816_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k17408.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n248320_k5120.cu"
)
if(NINFER_TARGET STREQUAL "rtx5090")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/shapes/n14976_k5120.cu"
  )
endif()
