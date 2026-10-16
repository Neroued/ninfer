if(NINFER_BUILD_ENGINE)
  ninfer_add_test(ninfer_bench_support_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_ninfer_bench_support.cpp"
            ${PROJECT_SOURCE_DIR}/bench/inference/ninfer_bench_support.cpp
    NEEDS_SOURCE_DIR
    LIBRARIES ninfer_engine ninfer::json)

  target_include_directories(ninfer_bench_support_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/inference)
endif()

ninfer_add_op_test(ninfer_bench_fixtures_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_bench_fixtures.cu"
  LIBRARIES ninfer_ops)
target_include_directories(ninfer_bench_fixtures_test PRIVATE ${PROJECT_SOURCE_DIR}/bench/ops)

add_executable(ninfer_context_cost_measure_test
  "${CMAKE_CURRENT_LIST_DIR}/../test_context_cost_measure.cpp"
  ${PROJECT_SOURCE_DIR}/bench/context_cost/context_cost_measure.cpp)

target_include_directories(ninfer_context_cost_measure_test PRIVATE
  ${PROJECT_SOURCE_DIR}/bench/context_cost)

add_test(NAME ninfer_context_cost_measure_test COMMAND ninfer_context_cost_measure_test)
