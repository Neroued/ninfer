target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/vocabulary_logprobs.cu"
  "${CMAKE_CURRENT_LIST_DIR}/argmax.cu"
  "${CMAKE_CURRENT_LIST_DIR}/target_logprobs.cu"
)
