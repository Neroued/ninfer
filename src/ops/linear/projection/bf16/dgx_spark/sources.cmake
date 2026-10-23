target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1152_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n7296_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n640_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2560_k320.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n77440_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n4096_k1024.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2048_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n4096_k128.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n288_k4096.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n48_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2560_k3072.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n124160_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1664_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2560_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n96_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n13952_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2560_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n12800_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n324_k10240.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n320_k10240.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n10240_k320.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1152_k1536.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n3456_k1152.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1152_k1152.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n4304_k1152.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1152_k4304.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n4608_k4608.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2560_k4608.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n248320_k2560.cu"
)
target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/n20480_k1536.cu")
target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k1536.cu")
target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/n4096_k8192.cu")
target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/n1024_k1176.cu")
target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/n3072_k1024.cu")
target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/n1024_k1024.cu")
target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/n8192_k1024.cu")
