if(NINFER_TARGET STREQUAL "dgx_spark")
  ninfer_add_op_bench(ninfer_mhc_bench SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cu")
endif()
