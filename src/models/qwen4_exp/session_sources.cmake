target_sources(ninfer_model_control PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/parallel/vision.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/work.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/session.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/request.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/context.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/context_execution.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/checkpoints.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/reclaim.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/execution.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/commit.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/scoring.cpp")

if(NINFER_ENABLE_DISTRIBUTED)
  target_sources(ninfer_model_control PRIVATE "${CMAKE_CURRENT_LIST_DIR}/wire.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/parallel/wire.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/parallel/codec.cpp")
  target_sources(ninfer_model_control PRIVATE "${CMAKE_CURRENT_LIST_DIR}/parallel/inputs.cpp")
  target_link_libraries(ninfer_model_control PRIVATE ninfer_runtime_distributed)
endif()
