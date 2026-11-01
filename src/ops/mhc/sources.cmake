target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/entry.cpp")
if(NINFER_TARGET STREQUAL "dgx_spark")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/select.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/launch.cu")
else()
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/unsupported.cpp")
endif()
