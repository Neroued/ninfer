target_sources(ninfer_model_execution PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/execution/parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/dense_mlp.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/gdn.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/mtp.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/text.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/draft.cpp"
)
