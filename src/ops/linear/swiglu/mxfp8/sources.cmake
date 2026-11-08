target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp")

if(NINFER_TARGET STREQUAL "dgx_spark")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/shapes/n12288_k4096.cu"
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/shapes/n2048_k4096.cu"
  )
endif()
