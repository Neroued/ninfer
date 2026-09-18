target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/decode/sparse_moe_decode_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/decode/sparse_moe_decode_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/prefill/sparse_moe_prefill_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/prefill/sparse_moe_prefill_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/small_t/sparse_moe_small_t_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/small_t/sparse_moe_small_t_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../wrapper/sparse_moe.cpp"
)
