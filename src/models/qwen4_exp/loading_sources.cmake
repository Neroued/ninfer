target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/definition.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/model.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/parallel/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/parallel/weights.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/parallel/expert_priority.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load/weights.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/profile.cpp")
