target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/gated_delta_net.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/replay.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/recurrent.cu"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/chunked/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/chunked/prepare_wy_wu.cu"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/chunked/state_passing.cu"
  "${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/chunked/output.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kimi_delta_attention/kimi_delta_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/kimi_delta_attention/recurrent.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kimi_delta_attention/chunked/prepare.cu"
  "${CMAKE_CURRENT_LIST_DIR}/kimi_delta_attention/chunked/recurrence.cu"
)
