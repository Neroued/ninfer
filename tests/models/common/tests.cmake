ninfer_add_test(ninfer_qwen_frontend_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/qwen/test_frontend.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_model_control ninfer_core ninfer::json)

ninfer_add_test(ninfer_qwen_tool_constraints_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/qwen/test_tool_constraints.cpp"
  LIBRARIES ninfer_model_control ninfer_grammar ninfer::json)

add_test(NAME ninfer_qwen_tool_schema_oracle_test
  COMMAND ${CMAKE_COMMAND} -E env
    "NINFER_TOOL_PROBE=$<TARGET_FILE:ninfer_qwen_tool_constraints_test>"
    ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/qwen/test_tool_schema.py")

ninfer_add_test(ninfer_qwen_visual_scatter_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/qwen/test_visual_scatter.cpp"
  LIBRARIES ninfer_model_execution ninfer_core)

ninfer_add_test(ninfer_context_allocation_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/context/test_allocation.cpp"
  LIBRARIES ninfer_model_control)

ninfer_add_test(ninfer_context_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/context/test_store.cpp"
  LIBRARIES ninfer_model_execution ninfer_core)

ninfer_add_test(ninfer_kv_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/context/test_kv_cache.cpp"
  LIBRARIES ninfer_model_execution ninfer_core)

ninfer_add_test(ninfer_kv_history_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/context/test_kv_history.cpp"
  LIBRARIES ninfer_model_execution ninfer_core)

ninfer_add_test(ninfer_recovery_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/context/test_recovery.cpp")

ninfer_add_test(ninfer_tool_call_parser_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/qwen/test_tool_call_parser.cpp"
  LIBRARIES ninfer_model_control ninfer::json)

set_tests_properties(ninfer_context_store_test ninfer_kv_history_test ninfer_kv_cache_test
  ninfer_qwen_visual_scatter_test
  PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")
