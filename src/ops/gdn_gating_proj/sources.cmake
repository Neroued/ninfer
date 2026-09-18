target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16/bf16_gdn_norm_gating_proj_27.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/bf16_gdn_gating_proj_kernels.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/bf16_gdn_gating_proj_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/../wrapper/gdn_gating_proj.cpp"
)
