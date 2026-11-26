target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/linear_bias.cpp")
include("${CMAKE_CURRENT_LIST_DIR}/mxfp8/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/nvfp4_mul/sources.cmake")
if(NINFER_TARGET STREQUAL "rtx5090")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/rtx5090/shapes/n768_k2304.cu"
    "${CMAKE_CURRENT_LIST_DIR}/rtx5090/shapes/n1024_k1024.cu"
    "${CMAKE_CURRENT_LIST_DIR}/rtx5090/shapes/n4096_k1024.cu"
    "${CMAKE_CURRENT_LIST_DIR}/rtx5090/shapes/n1024_k4096.cu")
endif()
if(NINFER_TARGET STREQUAL "dgx_spark")
  target_sources(ninfer_ops_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/shapes/n1024_k1176.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/shapes/n1024_k1024.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/shapes/n1024_k4096.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/shapes/n4096_k4096.cu")
endif()
