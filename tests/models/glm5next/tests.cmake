ninfer_add_test(ninfer_glm5next_frontend_resources_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_resources.cpp"
  LIBRARIES ninfer_model_loading ninfer::json)

ninfer_add_test(ninfer_glm5next_chat_template_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_chat_template.cpp"
  LIBRARIES ninfer_model_control ninfer_jinja ninfer::json
  NEEDS_SOURCE_DIR)

ninfer_add_test(ninfer_glm5next_preparation_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_preparation.cpp"
  LIBRARIES ninfer_model_control ninfer::json
  NEEDS_SOURCE_DIR)

ninfer_add_test(ninfer_glm5next_output_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_output.cpp"
  LIBRARIES ninfer_model_control ninfer::json
  NEEDS_SOURCE_DIR)

ninfer_add_test(ninfer_glm5next_tool_constraints_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tool_constraints.cpp"
  LIBRARIES ninfer_model_control ninfer_grammar ninfer::json
  NEEDS_SOURCE_DIR)

add_test(NAME ninfer_glm5next_tool_schema_oracle_test
  COMMAND ${CMAKE_COMMAND} -E env
    "NINFER_GLM_TOOL_PROBE=$<TARGET_FILE:ninfer_glm5next_tool_constraints_test>"
    ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/test_tool_schema.py")
