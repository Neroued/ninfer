target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/input_conv/input_conv.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/chunked/prepare.cu"
  "${CMAKE_CURRENT_LIST_DIR}/chunked/recurrence.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kimi_delta_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/replay/fold.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/replay/fold.cu"
  "${CMAKE_CURRENT_LIST_DIR}/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/recurrent.cu"
)

target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/${NINFER_TARGET}/select.cpp")

if(NINFER_TARGET STREQUAL "dgx_spark")
  target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/input_conv.cu")
else()
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/input_conv/launch.cu")
endif()
