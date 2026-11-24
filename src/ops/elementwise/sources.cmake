target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/add_bias.cu"
  "${CMAKE_CURRENT_LIST_DIR}/cast.cu"
  "${CMAKE_CURRENT_LIST_DIR}/scalar.cu"
  "${CMAKE_CURRENT_LIST_DIR}/position.cu"
  "${CMAKE_CURRENT_LIST_DIR}/residual_add.cu"
  "${CMAKE_CURRENT_LIST_DIR}/gelu.cu"
  "${CMAKE_CURRENT_LIST_DIR}/silu_mul.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sigmoid_mul.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sum_contributions.cu"
)
