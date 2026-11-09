target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/linear_topk.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/keys/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/keys/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n248320_k5120/mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/fp8/shapes/n248320_k5120/sliced.cu"
  "${CMAKE_CURRENT_LIST_DIR}/merge.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4/shapes/n131072_k5120/mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q4/shapes/n131072_k5120/sliced.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n248320_k5120/mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q8/shapes/n248320_k5120/sliced.cu"
)

if(NINFER_TARGET STREQUAL "dgx_spark")
  target_sources(ninfer_ops_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/keys/dgx_spark/bf16.cu"
    "${CMAKE_CURRENT_LIST_DIR}/keys/dgx_spark/mxfp8.cu"
    "${CMAKE_CURRENT_LIST_DIR}/keys/dgx_spark/q4.cu")
endif()
