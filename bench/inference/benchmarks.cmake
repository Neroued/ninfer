# End-to-end product benchmark: public Engine API and native .ninfer artifacts only.
add_executable(ninfer_bench
  "${CMAKE_CURRENT_LIST_DIR}/ninfer_bench.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/ninfer_bench_support.cpp")
ninfer_internal_includes(ninfer_bench)
target_include_directories(ninfer_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(ninfer_bench PRIVATE NINFER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(ninfer_bench PRIVATE ninfer_engine CUDA::cudart)
