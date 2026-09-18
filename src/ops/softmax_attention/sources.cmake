target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/causal_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_fp8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_nvfp4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_k8v4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_fp8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_nvfp4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_k8v4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/packed_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/context_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/sliding_window_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/launch.cu"
)

target_sources(ninfer_nvfp4_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_nvfp4_non_rdc.cu"
)
