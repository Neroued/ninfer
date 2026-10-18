target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/vocabulary/selection.cu"
  "${CMAKE_CURRENT_LIST_DIR}/vocabulary/decision.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sample/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sample/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/acceptance/greedy_drafts/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/acceptance/greedy_drafts/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/acceptance/sparse_drafts/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/acceptance/sparse_drafts/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/candidate_path/entry.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/candidate_path/execute.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/candidate_path/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/token_counts.cu"
)
