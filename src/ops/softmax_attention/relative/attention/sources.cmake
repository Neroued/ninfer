target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/plan.cpp")
if(NINFER_TARGET STREQUAL "rtx5090")
  target_sources(ninfer_ops_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/rtx5090/decode_launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/rtx5090/ragged_launch.cu")
endif()
