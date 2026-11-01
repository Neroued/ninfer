add_executable(ninfer_ple_disk_table_test "${CMAKE_CURRENT_LIST_DIR}/ple/disk/test_table.cpp")
ninfer_test_includes(ninfer_ple_disk_table_test)
target_link_libraries(ninfer_ple_disk_table_test PRIVATE ninfer_model_loading)
add_test(NAME ninfer_ple_disk_table_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/ple/disk/test_table.py"
    $<TARGET_FILE:ninfer_ple_disk_table_test>)
set_tests_properties(ninfer_ple_disk_table_test PROPERTIES TIMEOUT 30)

add_executable(ninfer_ple_disk_prepare_test "${CMAKE_CURRENT_LIST_DIR}/ple/disk/test_prepare.cpp")
ninfer_test_includes(ninfer_ple_disk_prepare_test)
target_link_libraries(ninfer_ple_disk_prepare_test PRIVATE ninfer_model_execution)
ninfer_test_interpose(ninfer_ple_disk_prepare_test cudaEventRecord)
foreach(mode IN ITEMS host gpu)
  add_test(NAME ninfer_ple_disk_prepare_${mode}_test
    COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/ple/disk/test_prepare.py"
      $<TARGET_FILE:ninfer_ple_disk_prepare_test> --${mode})
  set_tests_properties(ninfer_ple_disk_prepare_${mode}_test PROPERTIES TIMEOUT 30)
endforeach()
set_tests_properties(ninfer_ple_disk_prepare_gpu_test PROPERTIES SKIP_RETURN_CODE 77)
set_tests_properties(ninfer_ple_disk_prepare_host_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=")

ninfer_add_test(ninfer_ple_upload_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/ple/test_upload.cpp"
  LIBRARIES ninfer_model_execution)
set_tests_properties(ninfer_ple_upload_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 30)
ninfer_add_test(ninfer_qwen4_exp_loading_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_real.cpp"
  LIBRARIES ninfer_model_execution)
set_tests_properties(ninfer_qwen4_exp_loading_real_test PROPERTIES
  SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real")
ninfer_add_test(ninfer_qwen4_exp_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp"
  LIBRARIES ninfer_model_loading)
ninfer_add_test(ninfer_qwen4_exp_state_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_state.cpp"
  LIBRARIES ninfer_model_execution)
set_tests_properties(ninfer_qwen4_exp_state_test PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")

ninfer_add_test(ninfer_qwen4_exp_reclaim_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_reclaim_real.cpp"
  LIBRARIES ninfer_model_execution ninfer_model_loading ninfer_core)
set_tests_properties(ninfer_qwen4_exp_reclaim_real_test PROPERTIES
  SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real")

ninfer_add_test(ninfer_qwen4_exp_score_state_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_score_state_real.cpp"
  LIBRARIES ninfer_model_execution)
set_tests_properties(ninfer_qwen4_exp_score_state_real_test PROPERTIES
  SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real")
