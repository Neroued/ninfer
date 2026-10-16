target_sources(ninfer_ops PRIVATE
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
