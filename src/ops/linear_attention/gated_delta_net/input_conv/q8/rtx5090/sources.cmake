target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/convolution_n16384_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/projection_n16384_k2560.cu"
)
