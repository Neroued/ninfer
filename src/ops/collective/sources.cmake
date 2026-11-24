if(NINFER_ENABLE_DISTRIBUTED)
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/entry.cpp")
  target_link_libraries(ninfer_ops PRIVATE ninfer_distributed_cuda)
endif()
