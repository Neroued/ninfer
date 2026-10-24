target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/linear_bias.cpp")
if(NINFER_TARGET STREQUAL "dgx_spark")
  target_sources(ninfer_ops_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/shapes/n1024_k1176.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/shapes/n1024_k1024.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/shapes/n1024_k4096.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/shapes/n4096_k4096.cu")
endif()
