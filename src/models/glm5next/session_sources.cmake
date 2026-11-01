target_sources(ninfer_model_execution PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/session/binding.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/checkpoints.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/commit.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/context.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/execution.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/pause.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/planning.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/reclaim.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/request.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/scoring.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/sequence.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/session.cpp"
)

target_sources(ninfer_model_control PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/work.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/vision/control.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/state/identity.cpp")
