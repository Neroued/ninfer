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
