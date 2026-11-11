target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/mix.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/implementation.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp")

include("${CMAKE_CURRENT_LIST_DIR}/${NINFER_TARGET}/sources.cmake")
