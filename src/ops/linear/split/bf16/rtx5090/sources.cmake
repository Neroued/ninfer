target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1152_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1664_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n13952_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n8576_k2304.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n6528_k2304.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k2304.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2048_k2304.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n3072_k1024_bias.cu"
)
