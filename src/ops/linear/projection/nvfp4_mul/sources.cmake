target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp")

if(NINFER_TARGET STREQUAL "dgx_spark")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/shapes/n8192_k1024.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/shapes/n1024_k4096.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/shapes/n4096_k4096.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/shapes/n20480_k4096.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/shapes/n4096_k10240.cu"
  )
endif()
