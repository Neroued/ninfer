target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n64_k2048.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/n96_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/control.cu"
  "${CMAKE_CURRENT_LIST_DIR}/execute.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/projection.cpp"
)

include("${CMAKE_CURRENT_LIST_DIR}/${NINFER_TARGET}/sources.cmake")
