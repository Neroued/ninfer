target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_format.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_w4a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n34816_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k17408.cu"
)

target_sources(ninfer_nvfp4_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4_w4a4_tma.cu"
)
