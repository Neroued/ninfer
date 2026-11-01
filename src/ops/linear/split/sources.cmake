target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bias.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/shapes/n14336_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n14336_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n14336_k5120/a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n14336_k5120/shape.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n16384_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n16384_k5120/a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n16384_k5120/shape.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/linear_split.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n14336_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n14336_k5120/a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n14336_k5120/shape.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n16384_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n16384_k5120/a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n16384_k5120/shape.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n4096_n12288_k5120/mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n4096_n12288_k5120/shape.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n4096_n12288_k5120/small.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n7168_n7168_k5120/mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n7168_n7168_k5120/shape.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4_q5/shapes/n7168_n7168_k5120/small.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n12288_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n6144_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n6144_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n9216_k2048.cu"
)

target_sources(ninfer_ops_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n14336_k5120/a4_tma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n16384_k5120/a4_tma.cu"
)

include("${CMAKE_CURRENT_LIST_DIR}/bf16/${NINFER_TARGET}/sources.cmake")

include("${CMAKE_CURRENT_LIST_DIR}/q8/${NINFER_TARGET}/sources.cmake")
