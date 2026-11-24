target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/linear_topk.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n248320_k5120/mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n248320_k5120/sliced.cu"
  "${CMAKE_CURRENT_LIST_DIR}/merge.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4/shapes/n131072_k5120/mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4/shapes/n131072_k5120/sliced.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n248320_k5120/mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n248320_k5120/sliced.cu"
)
