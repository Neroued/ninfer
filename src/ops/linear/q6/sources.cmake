target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/q6_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n248320_k5120.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n34816_k5120.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n248320_k2048.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1152_k1536.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q6_rowsplit_gemm_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q6_rowsplit_gemm_simt.cu"
)
