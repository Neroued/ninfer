target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/mix.cpp")
if(NINFER_TARGET STREQUAL "rtx5090")
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/rtx5090/launch.cu")
endif()
