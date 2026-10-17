target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/causal_conv1d_silu.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/launch.cu"
)
if(NINFER_TARGET STREQUAL "dgx_spark")
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/dgx_spark/split.cu")
endif()
