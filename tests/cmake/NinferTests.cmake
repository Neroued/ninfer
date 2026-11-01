# Test declarations are included from tests/CMakeLists.txt so executable paths and
# CTest working directories stay under <build-dir>/tests.
function(ninfer_test_includes target)
  ninfer_internal_includes(${target})
  target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
endfunction()

# Export only the replacements used by this executable's fault-injection cases.
# Production shared libraries resolve these symbols through the normal ELF lookup.
function(ninfer_test_interpose target)
  target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
  foreach(symbol IN LISTS ARGN)
    target_link_options(${target} PRIVATE "LINKER:--export-dynamic-symbol=${symbol}")
  endforeach()
endfunction()

function(ninfer_add_test name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "NEEDS_SOURCE_DIR" "" "SOURCES;LIBRARIES")
  add_executable(${name} ${arg_SOURCES})
  ninfer_test_includes(${name})
  target_link_libraries(${name} PRIVATE ${arg_LIBRARIES})
  if(arg_NEEDS_SOURCE_DIR)
    target_compile_definitions(${name} PRIVATE
      NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
      NINFER_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")
  endif()
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# Apply these to the translation unit containing the oracle, including shared
# test support libraries. Executable options do not propagate into those libraries.
function(ninfer_op_oracle_options target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math>
      $<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)
  endif()
endfunction()

function(ninfer_add_op_test name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES;LIBRARIES;FORMATS")
  add_executable(${name} ${arg_SOURCES})
  ninfer_test_includes(${name})
  target_link_libraries(${name} PRIVATE ${arg_LIBRARIES} ninfer_op_test_support)
  ninfer_op_oracle_options(${name})
  if(arg_FORMATS)
    string(REGEX REPLACE "_test$" "" prefix "${name}")
    foreach(format IN LISTS arg_FORMATS)
      set(test "${prefix}_${format}_test")
      add_test(NAME ${test} COMMAND ${name} --format ${format})
      set_tests_properties(${test} PROPERTIES SKIP_RETURN_CODE 77 LABELS "ops")
    endforeach()
  else()
    add_test(NAME ${name} COMMAND ${name})
    set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77 LABELS "ops")
  endif()
endfunction()
