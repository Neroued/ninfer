target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bias.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n34816_k5120/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/linear_swiglu.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n34816_k5120/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q4/shapes/n34816_k5120/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n34816_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n34816_k5120/a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n34816_k5120/a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n34816_k5120/a4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4/shapes/n34816_k5120/materialized.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4/shapes/n34816_k5120/mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4/shapes/n34816_k5120/small.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n12288_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n34816_k5120.cu"
)

target_sources(ninfer_ops_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/shapes/n34816_k5120/a4_tma.cu"
)

if(NINFER_TARGET STREQUAL "dgx_spark")
  target_sources(ninfer_ops_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/bf16/dgx_spark/shapes/n8192_k1024_bias.cu"
    "${CMAKE_CURRENT_LIST_DIR}/bf16/dgx_spark/shapes/n20480_k4096.cu")
  target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/q8/dgx_spark/shapes/n2048_k4096.cu")
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/nvfp4/dgx_spark/shapes/n12288_k4096.cu")
  target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/nvfp4/dgx_spark/shapes/n12288_k4096_tma.cu")
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/bf16/dgx_spark/shapes/n2048_k4096.cu")
endif()
