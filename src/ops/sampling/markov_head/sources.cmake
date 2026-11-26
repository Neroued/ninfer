target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/entry.cpp"
)
target_sources(ninfer_ops_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/context.cu"
  "${CMAKE_CURRENT_LIST_DIR}/step.cu"
  "${CMAKE_CURRENT_LIST_DIR}/logits_add_softcap.cu"
)
