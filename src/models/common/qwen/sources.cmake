target_sources(ninfer_model_control PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/prefix_identity.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/frontend.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/output.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/chat_template.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/prompt_layout.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/processor.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_call_parser.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_contract.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_grammar.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/frontend/tool_name_constraint.cpp"
)

target_sources(ninfer_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/frontend/resources.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/vision/parameters.cpp"
)

target_sources(ninfer_model_execution PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/vision/vision.cpp"
)

target_sources(ninfer_model_control PRIVATE "${CMAKE_CURRENT_LIST_DIR}/vision/control.cpp")
