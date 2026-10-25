target_sources(ninfer_model_execution PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/vision/execution.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/profile.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/workspace.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/state/context.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/frame.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/planning.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/resources.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/parallel/preparation.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/startup.cpp")
