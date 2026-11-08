ninfer_add_test(ninfer_byte_size_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_byte_size.cpp")

# This target deliberately receives no src/, CUDA, artifact, kernel, or target
# include root. It proves that the public product headers stand alone.
add_executable(ninfer_public_api_test "${CMAKE_CURRENT_LIST_DIR}/../test_public_api.cpp")
target_include_directories(ninfer_public_api_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
add_test(NAME ninfer_public_api_test COMMAND ninfer_public_api_test)

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

ninfer_add_test(ninfer_arena_test        SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_arena.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_materialization_budget_test SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_materialization_budget.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_kv_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_kv_cache.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_state_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_state_store.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_disk_kv_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_disk_kv_cache.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_gdn_replay_records_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_gdn_replay_records.cpp"
  LIBRARIES ninfer_core)

set_tests_properties(
  ninfer_device_test
  ninfer_decode_graph_test
  ninfer_arena_test
  ninfer_kv_cache_test
  ninfer_state_store_test
  PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_expert_cache_policy_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_expert_cache_policy.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_expert_cache_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_expert_cache.cu"
  LIBRARIES ninfer_core)
# T158: the fence and mid-window-fill scenarios need a 20 ms reader kernel to overlap a copy; another
# process on the same GPU time-slices them apart (the controls then fail, no slot is ever misread).
set_tests_properties(ninfer_expert_cache_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)

ninfer_add_test(ninfer_elastic_arena_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_elastic_arena.cu"
  LIBRARIES ninfer_core)
target_link_options(ninfer_elastic_arena_test PRIVATE
  "LINKER:--wrap=cuMemCreate,--wrap=cuMemMap,--wrap=cuMemSetAccess,--wrap=cuMemRelease,--wrap=cuMemUnmap")
set_tests_properties(ninfer_elastic_arena_test PROPERTIES SKIP_RETURN_CODE 77)

ninfer_add_test(ninfer_expert_staging_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_expert_staging.cu"
  LIBRARIES ninfer_core)
set_tests_properties(ninfer_expert_staging_test PROPERTIES SKIP_RETURN_CODE 77)

# Replays routing.bin through the policy; it was checked against the offline cache simulators
# window by window (T14; needs a routing trace, so not a CTest).
add_executable(ninfer_expert_cache_replay "${CMAKE_CURRENT_LIST_DIR}/../expert_cache_replay.cpp")
ninfer_test_includes(ninfer_expert_cache_replay)
target_link_libraries(ninfer_expert_cache_replay PRIVATE ninfer_core)

ninfer_add_test(ninfer_host_timing_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_host_timing.cpp"
  LIBRARIES ninfer_core)

ninfer_add_test(ninfer_jinja_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../text/test_jinja.cpp"
  LIBRARIES ninfer_jinja ninfer::json)

add_test(NAME ninfer_chat_templates_test
  COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tests/text/test_chat_templates.py
          $<TARGET_FILE:ninfer_jinja_test>)

ninfer_add_test(ninfer_uva_copy_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_uva_copy.cu"
  LIBRARIES ninfer_core)
set_tests_properties(ninfer_uva_copy_test PROPERTIES SKIP_RETURN_CODE 77)
