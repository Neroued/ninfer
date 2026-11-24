ninfer_add_test(ninfer_glm5next_frontend_resources_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_resources.cpp"
  LIBRARIES ninfer_model_loading ninfer::json)

ninfer_add_test(ninfer_glm5next_chat_template_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_chat_template.cpp"
  LIBRARIES ninfer_model_control ninfer_jinja ninfer::json
  NEEDS_SOURCE_DIR)

ninfer_add_test(ninfer_glm5next_preparation_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_preparation.cpp"
  LIBRARIES ninfer_model_control ninfer::json
  NEEDS_SOURCE_DIR)

ninfer_add_test(ninfer_glm5next_output_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_output.cpp"
  LIBRARIES ninfer_model_control ninfer::json
  NEEDS_SOURCE_DIR)

ninfer_add_test(ninfer_glm5next_tool_constraints_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tool_constraints.cpp"
  LIBRARIES ninfer_model_control ninfer_grammar ninfer::json
  NEEDS_SOURCE_DIR)

add_test(NAME ninfer_glm5next_tool_schema_oracle_test
  COMMAND ${CMAKE_COMMAND} -E env
    "NINFER_GLM_TOOL_PROBE=$<TARGET_FILE:ninfer_glm5next_tool_constraints_test>"
    ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/test_tool_schema.py")

ninfer_add_test(ninfer_glm5next_media_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_media.cpp"
  LIBRARIES ninfer_model_control ninfer::json
  NEEDS_SOURCE_DIR)

ninfer_add_test(ninfer_glm5next_protocol_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_protocol.cpp"
  LIBRARIES ninfer_model_control ninfer_serve ninfer::json
  NEEDS_SOURCE_DIR)

add_executable(ninfer_glm5next_loading_test "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp")
ninfer_test_includes(ninfer_glm5next_loading_test)
target_link_libraries(ninfer_glm5next_loading_test PRIVATE ninfer_model_execution)
foreach(mode IN ITEMS host gpu)
  add_test(NAME ninfer_glm5next_loading_${mode}_test
    COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/test_loading.py"
      $<TARGET_FILE:ninfer_glm5next_loading_test> --${mode})
  set_tests_properties(ninfer_glm5next_loading_${mode}_test PROPERTIES TIMEOUT 60)
endforeach()
set_tests_properties(ninfer_glm5next_loading_gpu_test PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")
set_tests_properties(ninfer_glm5next_loading_host_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=")

add_executable(ninfer_glm5next_execution_resources_test "${CMAKE_CURRENT_LIST_DIR}/test_execution_resources.cpp")
ninfer_test_includes(ninfer_glm5next_execution_resources_test)
target_link_libraries(ninfer_glm5next_execution_resources_test PRIVATE ninfer_model_execution)
foreach(mode IN ITEMS host gpu)
  add_test(NAME ninfer_glm5next_execution_resources_${mode}_test
    COMMAND ninfer_glm5next_execution_resources_test --${mode})
endforeach()
set_tests_properties(ninfer_glm5next_execution_resources_host_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=")
set_tests_properties(ninfer_glm5next_execution_resources_gpu_test PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")

if(NINFER_TARGET STREQUAL "dgx_spark")
  ninfer_add_test(ninfer_glm5next_vision_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_vision.cpp"
    LIBRARIES ninfer_model_execution)
  set_tests_properties(ninfer_glm5next_vision_test PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")
endif()

add_executable(ninfer_glm5next_session_test "${CMAKE_CURRENT_LIST_DIR}/test_session.cpp")
ninfer_test_includes(ninfer_glm5next_session_test)
target_link_libraries(ninfer_glm5next_session_test PRIVATE ninfer_model_execution)
target_compile_definitions(ninfer_glm5next_session_test PRIVATE NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
add_test(NAME ninfer_glm5next_session_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/test_loading.py"
    $<TARGET_FILE:ninfer_glm5next_session_test>)
set_tests_properties(ninfer_glm5next_session_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=")
