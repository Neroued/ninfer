target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/q5_a16_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q5_a16_sliced_k_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q5_a16_simt.cu"
)
