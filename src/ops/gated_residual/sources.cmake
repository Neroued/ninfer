target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/norm_combine.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/norm_combine/${NINFER_TARGET}/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/mix/mix.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/mix/operands.cpp")
include("${CMAKE_CURRENT_LIST_DIR}/mix/sources.cmake")
