include("${CMAKE_CURRENT_LIST_DIR}/control/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/input_conv/sources.cmake")
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/recurrence/chunked/prepare.cu"
  "${CMAKE_CURRENT_LIST_DIR}/recurrence/chunked/recurrence.cu"
  "${CMAKE_CURRENT_LIST_DIR}/recurrence/gated_delta_net.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/recurrence/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/recurrence/${NINFER_TARGET}/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/recurrence/${NINFER_TARGET}/recurrent.cu"
  "${CMAKE_CURRENT_LIST_DIR}/replay/fold.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/replay/${NINFER_TARGET}/fold.cu"
  "${CMAKE_CURRENT_LIST_DIR}/replay/record.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/replay/${NINFER_TARGET}/record.cu"
)
