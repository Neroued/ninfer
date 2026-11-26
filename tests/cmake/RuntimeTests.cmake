ninfer_add_test(ninfer_scheduler_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_scheduler.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_context_cost_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_context_cost.cpp"
  LIBRARIES ninfer_runtime_support ninfer::json)

ninfer_add_test(ninfer_placement_policy_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_placement_policy.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_resource_manager_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_resource_manager.cpp"
  LIBRARIES ninfer_runtime_support ninfer_model_control)

ninfer_add_test(ninfer_context_capacity_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_context_capacity.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_dp_router_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_dp_router.cpp"
  LIBRARIES ninfer_runtime_support)

add_executable(ninfer_multi_replica_engine_test
  "${CMAKE_CURRENT_LIST_DIR}/../runtime/test_multi_replica_engine.cpp")
ninfer_test_includes(ninfer_multi_replica_engine_test)
target_link_libraries(ninfer_multi_replica_engine_test PRIVATE
  ninfer_model_execution ninfer_runtime_support)
target_compile_definitions(ninfer_multi_replica_engine_test PRIVATE
  NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME ninfer_multi_replica_engine_test
  COMMAND ${Python3_EXECUTABLE} -B "${PROJECT_SOURCE_DIR}/tests/models/glm5next/test_loading.py"
    $<TARGET_FILE:ninfer_multi_replica_engine_test>)
set_tests_properties(ninfer_multi_replica_engine_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=")

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
