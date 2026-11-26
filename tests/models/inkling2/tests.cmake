ninfer_add_test(ninfer_inkling2_frontend_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend.cpp"
  LIBRARIES ninfer_model_control ninfer_jinja ninfer_grammar ninfer::json
  NEEDS_SOURCE_DIR)
set_tests_properties(ninfer_inkling2_frontend_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=")

add_executable(ninfer_inkling2_loading_test "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp")
ninfer_test_includes(ninfer_inkling2_loading_test)
target_link_libraries(ninfer_inkling2_loading_test PRIVATE ninfer_model_loading)
target_compile_definitions(ninfer_inkling2_loading_test PRIVATE NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
foreach(mode IN ITEMS host gpu)
  add_test(NAME ninfer_inkling2_loading_${mode}_test
    COMMAND ninfer_inkling2_loading_test --${mode})
  set_tests_properties(ninfer_inkling2_loading_${mode}_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
endforeach()
set_tests_properties(ninfer_inkling2_loading_host_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=")
set_tests_properties(ninfer_inkling2_loading_gpu_test PROPERTIES LABELS "gpu")

add_executable(ninfer_inkling2_execution_test "${CMAKE_CURRENT_LIST_DIR}/test_execution.cpp")
ninfer_test_includes(ninfer_inkling2_execution_test)
target_link_libraries(ninfer_inkling2_execution_test PRIVATE ninfer_model_execution ninfer_runtime_support)
foreach(mode IN ITEMS host gpu)
  add_test(NAME ninfer_inkling2_execution_${mode}_test
    COMMAND ninfer_inkling2_execution_test --${mode})
  set_tests_properties(ninfer_inkling2_execution_${mode}_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
endforeach()
set_tests_properties(ninfer_inkling2_execution_host_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=")
set_tests_properties(ninfer_inkling2_execution_gpu_test PROPERTIES LABELS "gpu")
