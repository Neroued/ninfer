ninfer_add_test(ninfer_inkling2_frontend_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend.cpp"
  LIBRARIES ninfer_model_control ninfer_jinja ninfer_grammar ninfer::json
  NEEDS_SOURCE_DIR)
set_tests_properties(ninfer_inkling2_frontend_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=")
