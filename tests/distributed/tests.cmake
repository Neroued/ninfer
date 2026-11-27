set(ninfer_cluster_topology_libs ninfer_deployment)
if(NINFER_ENABLE_DISTRIBUTED)
  list(APPEND ninfer_cluster_topology_libs ninfer_distributed_wire)
endif()
ninfer_add_test(ninfer_cluster_topology_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_cluster_topology.cpp"
  LIBRARIES ${ninfer_cluster_topology_libs})
if(NINFER_ENABLE_DISTRIBUTED)
  target_compile_definitions(ninfer_cluster_topology_test PRIVATE NINFER_ENABLE_DISTRIBUTED=1)
endif()

if(NINFER_ENABLE_DISTRIBUTED AND NINFER_BUILD_ENGINE)
  # Explicit artifacts and optional device materialization; not an automatic download or GPU test.
  add_executable(ninfer_model_rank_load_test "${CMAKE_CURRENT_LIST_DIR}/test_model_rank_load.cpp")
  ninfer_internal_includes(ninfer_model_rank_load_test)
  target_link_libraries(ninfer_model_rank_load_test PRIVATE ninfer_model_execution ninfer_model_control)
  ninfer_add_test(ninfer_distributed_protocol_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_protocol.cpp"
    LIBRARIES ninfer_runtime_distributed ninfer_model_control)
  # Explicit deployment is required; ordinary CTest does not launch remote processes.
  add_executable(ninfer_distributed_execution_test "${CMAKE_CURRENT_LIST_DIR}/test_execution.cpp")
  ninfer_internal_includes(ninfer_distributed_execution_test)
  target_link_libraries(ninfer_distributed_execution_test PRIVATE
    ninfer_runtime_distributed ninfer_model_execution ninfer_product_config)
  add_executable(ninfer_moe_exchange_test "${CMAKE_CURRENT_LIST_DIR}/test_moe_exchange.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/moe/fixture.cpp" "${CMAKE_CURRENT_LIST_DIR}/moe/real.cpp")
  ninfer_test_includes(ninfer_moe_exchange_test)
  ninfer_op_oracle_options(ninfer_moe_exchange_test)
  target_link_libraries(ninfer_moe_exchange_test PRIVATE ninfer_model_execution ninfer_product_config
    ninfer_runtime_distributed ninfer_op_test_support)
  add_test(NAME ninfer_moe_compact_gpu_test COMMAND ninfer_moe_exchange_test --local)
  set_tests_properties(ninfer_moe_compact_gpu_test PROPERTIES SKIP_RETURN_CODE 77 LABELS gpu TIMEOUT 60)
  add_executable(ninfer_distributed_model_preparation_test "${CMAKE_CURRENT_LIST_DIR}/test_model_preparation.cpp")
  ninfer_internal_includes(ninfer_distributed_model_preparation_test)
  target_link_libraries(ninfer_distributed_model_preparation_test PRIVATE
    ninfer_model_execution ninfer_product_config ninfer_runtime_distributed)
endif()
