target_sources(ninfer_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/frontend/frontend.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/output_session.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/chat_template.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/prompt_layout.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/digest.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/processor.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/media_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_call_parser.cpp"
)
