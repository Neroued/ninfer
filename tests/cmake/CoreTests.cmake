# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the public product headers stand alone.
add_executable(ninfer_public_api_test "${CMAKE_CURRENT_LIST_DIR}/../test_public_api.cpp")
target_include_directories(ninfer_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME ninfer_public_api_test COMMAND ninfer_public_api_test)

foreach(part IN ITEMS system task_pool)
  ninfer_add_test(ninfer_cpu_${part}_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/../cpu/test_${part}.cpp"
    LIBRARIES ninfer_cpu)
  set_tests_properties(ninfer_cpu_${part}_test PROPERTIES TIMEOUT 30)
endforeach()

if(NINFER_TARGET STREQUAL "rtx5090")
  foreach(part IN ITEMS x86 executor)
    ninfer_add_test(ninfer_cpu_${part}_test
      SOURCES "${CMAKE_CURRENT_LIST_DIR}/../cpu/test_${part}.cpp"
      LIBRARIES ninfer_cpu_executor)
    set_tests_properties(ninfer_cpu_${part}_test PROPERTIES TIMEOUT 30)
  endforeach()
  ninfer_add_test(ninfer_cpu_tensor_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/../cpu/test_tensor.cpp"
    LIBRARIES ninfer_core ninfer_cpu_executor)
endif()

ninfer_add_test(ninfer_device_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_device.cpp"
  LIBRARIES ninfer_core)

set_tests_properties(ninfer_device_test PROPERTIES
  ENVIRONMENT_MODIFICATION "NINFER_CUDA_SYNC=unset:")
set(sync_modes spin blocking yield auto)
set(sync_flags 1 4 2 0)
foreach(mode flags IN ZIP_LISTS sync_modes sync_flags)
  add_test(NAME ninfer_device_sync_${mode}_test COMMAND ninfer_device_test ${flags})
  set_tests_properties(ninfer_device_sync_${mode}_test PROPERTIES
    ENVIRONMENT "NINFER_CUDA_SYNC=${mode}" SKIP_RETURN_CODE 77)
endforeach()
foreach(mode IN ITEMS invalid empty)
  add_test(NAME ninfer_device_sync_${mode}_test COMMAND ninfer_device_test --invalid-sync)
endforeach()
set_tests_properties(ninfer_device_sync_invalid_test PROPERTIES
  ENVIRONMENT "NINFER_CUDA_SYNC=invalid")
set_tests_properties(ninfer_device_sync_empty_test PROPERTIES
  ENVIRONMENT "NINFER_CUDA_SYNC=")

ninfer_add_test(ninfer_decode_graph_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_decode_graph.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_tensor_test       SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_tensor.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_memory_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_memory.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_nf4_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_nf4.cpp"
  LIBRARIES ninfer_core)
add_test(NAME ninfer_nf4_exact_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/../test_nf4_exact.py"
    $<TARGET_FILE:ninfer_nf4_test>)

ninfer_add_test(ninfer_read_queue_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../io/test_read_queue.cpp"
  LIBRARIES ninfer_core PkgConfig::URING)
ninfer_test_interpose(ninfer_read_queue_test io_uring_submit)
set_tests_properties(ninfer_read_queue_test PROPERTIES TIMEOUT 30)

ninfer_add_test(ninfer_arena_test        SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_arena.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_context_pool_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_context_pool.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_state_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_state_store.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_gdn_replay_records_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_gdn_replay_records.cpp"
  LIBRARIES ninfer_core)

set_tests_properties(
  ninfer_device_test
  ninfer_decode_graph_test
  ninfer_arena_test
  ninfer_state_store_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_host_timing_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_host_timing.cpp"
  LIBRARIES ninfer_runtime_support)

ninfer_add_test(ninfer_weight_bank_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_weight_bank.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_expert_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_expert_cache.cu"
          "${CMAKE_CURRENT_LIST_DIR}/../expert_cache_cuda_errors.cpp"
  LIBRARIES ninfer_core)
set_tests_properties(ninfer_expert_cache_test PROPERTIES TIMEOUT 60)

ninfer_test_interpose(ninfer_expert_cache_test
  cudaEventRecord cudaMemcpyAsync cudaMemcpyBatchAsync cudaStreamSynchronize cudaStreamWaitEvent
  cudaFree cudaFreeHost)
