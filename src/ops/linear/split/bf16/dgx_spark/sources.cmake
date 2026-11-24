target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n20480_k1536.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n640_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1152_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n7296_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1664_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n13952_k2560.cu"
)
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/n288_k4096.cu")

target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/n3072_k1024_bias.cu")
