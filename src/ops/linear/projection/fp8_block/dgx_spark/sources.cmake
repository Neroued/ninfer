target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2048_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n512_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k1536.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n4096_k16384.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n4096_k1024.cu"
)
