ninfer_add_test(ninfer_media_decode_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_media_decode.cpp"
  LIBRARIES ninfer_media_decode)

ninfer_add_test(ninfer_prompt_input_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_prompt_input.cpp"
  LIBRARIES ninfer_product_prompt_input)

ninfer_add_test(ninfer_pretty_logging_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_pretty_logging.cpp"
  LIBRARIES ninfer_product_logging)

ninfer_add_test(ninfer_perplexity_evaluation_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_perplexity_evaluation.cpp"
          ${PROJECT_SOURCE_DIR}/apps/perplexity/evaluation.cpp
  LIBRARIES ninfer_core)

target_include_directories(ninfer_perplexity_evaluation_test PRIVATE
  ${PROJECT_SOURCE_DIR}/apps/perplexity)

ninfer_add_test(ninfer_cli_options_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_cli_options.cpp" ${PROJECT_SOURCE_DIR}/apps/cli/options.cpp
  LIBRARIES ninfer_runtime_support ninfer_product_logging)

target_include_directories(ninfer_cli_options_test PRIVATE ${PROJECT_SOURCE_DIR}/apps/cli)

ninfer_add_test(ninfer_openai_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_schema.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_openai_responses_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_responses.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_openai_responses_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_openai_responses_store.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_anthropic_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_anthropic_schema.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_serve_options_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_serve_options.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_request_log_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_request_log.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_http_error_handler_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_error_handler.cpp"
  LIBRARIES ninfer_serve)

ninfer_add_test(ninfer_http_transport_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_http_transport.cpp"
  LIBRARIES ninfer_serve)
