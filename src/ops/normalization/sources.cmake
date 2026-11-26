target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/l2norm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/layer_norm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/weighted_sum_rmsnorm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/scaled_residual_add_layer_norm.cu"
)

include("${CMAKE_CURRENT_LIST_DIR}/rmsnorm/sources.cmake")
