# End-to-end product benchmark: public Engine API and native .ninfer artifacts only.
add_executable(ninfer_bench
  "${CMAKE_CURRENT_LIST_DIR}/ninfer_bench.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/options.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/report.cpp")
ninfer_internal_includes(ninfer_bench)
target_include_directories(ninfer_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ninfer_bench PRIVATE ninfer_engine ninfer_product_config
  ninfer_product_prompt_input ninfer::json CUDA::cudart)
