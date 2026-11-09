# Structured output (constrained decoding): CPU tests of the grammar library, from
# Neroued/ninfer 77c2e5e2 (tests/cmake/RuntimeTests.cmake). The oracle scripts validate the
# native results with the independent `jsonschema` package (tests/text/requirements.txt).

ninfer_add_test(ninfer_grammar_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_grammar.cpp"
  LIBRARIES ninfer_grammar)

ninfer_add_test(ninfer_regex_choice_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../text/test_regex_choice.cpp"
  LIBRARIES ninfer_grammar ninfer::json)

add_test(NAME ninfer_regex_choice_oracle_test
  COMMAND ${CMAKE_COMMAND} -E env
    "NINFER_REGEX_PROBE=$<TARGET_FILE:ninfer_regex_choice_test>"
    ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tests/text/test_regex_choice.py)

ninfer_add_test(ninfer_json_schema_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../text/test_json_schema.cpp"
  LIBRARIES ninfer_grammar ninfer::json)

add_test(NAME ninfer_json_schema_oracle_test
  COMMAND ${CMAKE_COMMAND} -E env
    "NINFER_SCHEMA_PROBE=$<TARGET_FILE:ninfer_json_schema_test>"
    ${Python3_EXECUTABLE} -B ${PROJECT_SOURCE_DIR}/tests/text/test_json_schema.py)

# capy-flash structured output limits: input limits, compilation deadline, empty languages, matcher
# time budgets, failed compilations kept out of the vendor cache (support/thread_safe_cache.h).
ninfer_add_test(ninfer_constraint_limits_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../text/test_constraint_limits.cpp"
  LIBRARIES ninfer_grammar ninfer_xgrammar ninfer::json)
target_include_directories(ninfer_constraint_limits_test PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xgrammar/cpp)

# Frontend matcher on the real Qwen3.8 tokenizer (structured output P3). Point
# NINFER_TEST_QWEN_TOKENIZER_DIR at a Qwen3.8 checkpoint directory (tokenizer.json,
# tokenizer_config.json, generation_config.json); without it the test skips.
set(NINFER_TEST_QWEN_TOKENIZER_DIR "" CACHE PATH
  "Qwen3.8 checkpoint directory with tokenizer files for the output constraint test")
ninfer_add_test(ninfer_output_constraint_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../models/qwen3_5/test_output_constraint.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core ninfer::json)
set_tests_properties(ninfer_output_constraint_test PROPERTIES SKIP_RETURN_CODE 77)
if(NINFER_TEST_QWEN_TOKENIZER_DIR)
  set_tests_properties(ninfer_output_constraint_test PROPERTIES
    ENVIRONMENT "NINFER_TEST_TOKENIZER_DIR=${NINFER_TEST_QWEN_TOKENIZER_DIR}")
endif()

# capy-flash: client JSON at --max-request-json-depth through the whole prompt path (real Qwen3.8
# template and tokenizer); one level more is a 400. Skips without NINFER_TEST_QWEN_TOKENIZER_DIR.
ninfer_add_test(ninfer_request_json_depth_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../test_request_json_depth.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES ninfer_engine ninfer_core ninfer_serve ninfer::json)
set_tests_properties(ninfer_request_json_depth_test PROPERTIES SKIP_RETURN_CODE 77)
if(NINFER_TEST_QWEN_TOKENIZER_DIR)
  set_tests_properties(ninfer_request_json_depth_test PROPERTIES
    ENVIRONMENT "NINFER_TEST_TOKENIZER_DIR=${NINFER_TEST_QWEN_TOKENIZER_DIR}")
endif()

# Engine on a real artifact (structured output P5/P6): phases and flags in the source header.
# Without arguments it skips. To run it in ctest, give the ninfer-serve flags of each profile
# (space separated), e.g.
#   -DNINFER_TEST_CONSTRAINT_ARGS="/path/27b.ninfer --tp 2 --spec mtp ..."
#   -DNINFER_TEST_CONSTRAINT_FLASHNEXT_ARGS="/path/flash_next.ninfer --profile flash-next ..."
# ctest then runs the `suite` phase on each profile (GPU: run ctest inside gpu-lease).
set(NINFER_TEST_CONSTRAINT_ARGS "" CACHE STRING
  "ninfer-serve flags of the profile for ninfer_engine_constraint_real_test (suite phase)")
set(NINFER_TEST_CONSTRAINT_FLASHNEXT_ARGS "" CACHE STRING
  "ninfer-serve flags of a second profile (Flash-Next) for the same test")
ninfer_add_test(ninfer_engine_constraint_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/../models/qwen3_5/test_engine_constraint_real.cpp"
  LIBRARIES ninfer_engine ninfer_core ninfer_serve ninfer::json)
set_tests_properties(ninfer_engine_constraint_real_test PROPERTIES SKIP_RETURN_CODE 77)
if(NINFER_TEST_CONSTRAINT_ARGS)
  set_tests_properties(ninfer_engine_constraint_real_test PROPERTIES TIMEOUT 3600
    ENVIRONMENT "NINFER_TEST_CONSTRAINT_SERVE_ARGS=${NINFER_TEST_CONSTRAINT_ARGS}")
endif()
if(NINFER_TEST_CONSTRAINT_FLASHNEXT_ARGS)
  add_test(NAME ninfer_engine_constraint_real_flashnext_test
    COMMAND ninfer_engine_constraint_real_test)
  set_tests_properties(ninfer_engine_constraint_real_flashnext_test PROPERTIES
    SKIP_RETURN_CODE 77 TIMEOUT 3600
    ENVIRONMENT "NINFER_TEST_CONSTRAINT_SERVE_ARGS=${NINFER_TEST_CONSTRAINT_FLASHNEXT_ARGS}")
endif()
