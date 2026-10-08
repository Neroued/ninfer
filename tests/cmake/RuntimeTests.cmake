ninfer_add_test(ninfer_scheduler_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_scheduler.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_context_cost_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_context_cost.cpp"
  LIBRARIES ninfer_runtime_support ninfer::json)

ninfer_add_test(ninfer_resource_manager_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_resource_manager.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_kv_capacity_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_kv_capacity.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_sampling_defaults_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_sampling_defaults.cpp"
  LIBRARIES ninfer_engine ninfer_core)

ninfer_add_test(ninfer_grammar_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_grammar.cpp"
  LIBRARIES ninfer_grammar)

ninfer_add_test(ninfer_regex_choice_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../text/test_regex_choice.cpp"
  LIBRARIES ninfer_grammar ninfer::json)

add_test(NAME ninfer_regex_choice_oracle_test
  COMMAND ${CMAKE_COMMAND} -E env
    "NINFER_REGEX_PROBE=$<TARGET_FILE:ninfer_regex_choice_test>"
    ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tests/text/test_regex_choice.py)

ninfer_add_test(ninfer_json_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../text/test_json_schema.cpp"
  LIBRARIES ninfer_grammar ninfer::json)

add_test(NAME ninfer_json_schema_oracle_test
  COMMAND ${CMAKE_COMMAND} -E env
    "NINFER_SCHEMA_PROBE=$<TARGET_FILE:ninfer_json_schema_test>"
    ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tests/text/test_json_schema.py)
