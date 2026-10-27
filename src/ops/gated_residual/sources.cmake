target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/norm_combine.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/mix/mix.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/mix/operands.cpp")
include("${CMAKE_CURRENT_LIST_DIR}/mix/sources.cmake")

if(NINFER_TARGET STREQUAL "rtx5090")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/norm_combine/rtx5090/dispatch.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/norm_combine/rtx5090/shapes/h2560.cu"
    "${CMAKE_CURRENT_LIST_DIR}/norm_combine/rtx5090/shapes/h5120.cu")
else()
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/norm_combine/${NINFER_TARGET}/launch.cu")
endif()
