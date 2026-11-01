ninfer_add_test(ninfer_jinja_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_jinja.cpp"
  LIBRARIES ninfer_jinja ninfer::json)

add_test(NAME ninfer_chat_templates_test
  COMMAND ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tests/text/test_chat_templates.py
          $<TARGET_FILE:ninfer_jinja_test>)

add_executable(ninfer_tokenizer_probe "${CMAKE_CURRENT_LIST_DIR}/tokenizer_probe.cpp")
ninfer_internal_includes(ninfer_tokenizer_probe)
target_link_libraries(ninfer_tokenizer_probe PRIVATE ninfer_text ninfer::json)
add_test(NAME ninfer_tokenizer_oracle_test
  COMMAND ${Python3_EXECUTABLE} -B "${CMAKE_CURRENT_LIST_DIR}/tokenizer_oracle.py"
          --probe $<TARGET_FILE:ninfer_tokenizer_probe>)
