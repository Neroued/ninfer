add_library(ninfer_hc_test_support STATIC
  "${CMAKE_CURRENT_LIST_DIR}/fixtures.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/oracle.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/runner.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/validation.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/mix/fixtures.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/mix/oracle.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/mix/validation.cpp")
ninfer_test_includes(ninfer_hc_test_support)
ninfer_op_oracle_options(ninfer_hc_test_support)
target_link_libraries(ninfer_hc_test_support PUBLIC ninfer_ops)
foreach(form IN ITEMS expand_norm grouped_rmsnorm combine combine_norm)
  ninfer_add_op_test(ninfer_hc_${form}_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_${form}.cpp"
    LIBRARIES ninfer_hc_test_support)
endforeach()
ninfer_add_op_test(ninfer_hc_mix_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mix.cpp"
  LIBRARIES ninfer_hc_test_support)
