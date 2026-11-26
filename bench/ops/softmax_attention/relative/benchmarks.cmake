ninfer_add_op_bench(ninfer_relative_attention_prepare_bench
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/prepare.cu")

if(NINFER_TARGET STREQUAL "rtx5090")
  ninfer_add_op_bench(ninfer_relative_softmax_attention_bench
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/attention.cu")
endif()
