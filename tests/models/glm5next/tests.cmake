ninfer_add_test(ninfer_glm5next_frontend_resources_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_resources.cpp"
  LIBRARIES ninfer_model_loading ninfer::json)
