function(ninfer_add_op_bench name)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "SOURCES")
  add_executable(${name} ${arg_SOURCES})
  ninfer_internal_includes(${name})
  target_include_directories(${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/ops)
  target_link_libraries(${name} PRIVATE ninfer_ops)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()
