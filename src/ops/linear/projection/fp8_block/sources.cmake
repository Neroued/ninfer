target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp"
)
if(NINFER_TARGET STREQUAL "dgx_spark")
  include("${CMAKE_CURRENT_LIST_DIR}/dgx_spark/sources.cmake")
endif()
