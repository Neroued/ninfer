target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/norm_combine.cu"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/mix/mix.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/mix/operands.cpp")
include("${CMAKE_CURRENT_LIST_DIR}/mix/${NINFER_TARGET}/sources.cmake")
