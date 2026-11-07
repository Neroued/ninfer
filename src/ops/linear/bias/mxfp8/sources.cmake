target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp")

if(NINFER_TARGET STREQUAL "dgx_spark")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/shapes/n1024_k1024.cu"
  )
endif()
