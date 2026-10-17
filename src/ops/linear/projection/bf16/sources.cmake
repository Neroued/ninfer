target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n256_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n48_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n248320_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n2560_k4608.cu"
)

include("${CMAKE_CURRENT_LIST_DIR}/${NINFER_TARGET}/sources.cmake")
