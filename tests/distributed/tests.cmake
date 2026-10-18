if(NINFER_ENABLE_DISTRIBUTED AND NINFER_BUILD_ENGINE)
  ninfer_add_test(ninfer_distributed_protocol_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_protocol.cpp"
    LIBRARIES ninfer_runtime_distributed ninfer_model_control)
  # Explicit deployment is required; ordinary CTest does not launch remote processes.
  add_executable(ninfer_distributed_execution_test "${CMAKE_CURRENT_LIST_DIR}/test_execution.cpp")
  ninfer_internal_includes(ninfer_distributed_execution_test)
  target_link_libraries(ninfer_distributed_execution_test PRIVATE
    ninfer_runtime_distributed ninfer_ops ninfer_product_config)
endif()
