target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/entry.cpp")
if(NINFER_TARGET STREQUAL "rtx5090")
  target_sources(ninfer_ops_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/rtx5090/small_m.cu"
    "${CMAKE_CURRENT_LIST_DIR}/rtx5090/cluster_m.cu"
    "${CMAKE_CURRENT_LIST_DIR}/rtx5090/large_m.cu")
endif()
