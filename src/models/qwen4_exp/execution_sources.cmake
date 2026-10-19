target_sources(ninfer_model_execution PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/parallel/execution.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/startup.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/parameters.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/workspace.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/gdn.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/qsa.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/ple.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/text.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/execution/mtp.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/state/state_image.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/state/kv_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/session/planning.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/buffers.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/ple_pipeline.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/device_ple.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/vocabulary.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/observations.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/control.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/vision.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/mtp.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/text_graph.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/graphs.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/device_graphs.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/graph_preparation.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/execution.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/commit.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/context.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/scoring.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/planning.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/resources.cpp")

if(NINFER_ENABLE_DISTRIBUTED)
  target_sources(ninfer_model_execution PRIVATE "${CMAKE_CURRENT_LIST_DIR}/parallel/startup.cpp")
  target_compile_definitions(ninfer_model_execution PRIVATE NINFER_ENABLE_DISTRIBUTED=1)
  target_sources(ninfer_model_execution PRIVATE "${CMAKE_CURRENT_LIST_DIR}/parallel/preparation.cpp")
  target_link_libraries(ninfer_model_execution PRIVATE ninfer_runtime_distributed)
endif()
