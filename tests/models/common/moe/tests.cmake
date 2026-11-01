ninfer_add_op_test(ninfer_moe_execution_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_execution.cpp"
          "${PROJECT_SOURCE_DIR}/tests/ops/moe/fixture.cpp"
  LIBRARIES ninfer_model_execution)
set_tests_properties(ninfer_moe_execution_test PROPERTIES
  LABELS "gpu" RUN_SERIAL TRUE RESOURCE_LOCK cpu_team)
