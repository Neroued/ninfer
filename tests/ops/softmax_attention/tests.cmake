ninfer_add_op_test(ninfer_sliding_window_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/sliding_window_attention.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_softmax_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/causal/cache.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/causal/device_cache.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/causal/oracle.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/causal/runner.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/causal/cases.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/plain_and_packed.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/context.cpp"
  LIBRARIES ninfer_ops
  FORMATS bf16 int8 fp8 nvfp4 k8v4)

include("${CMAKE_CURRENT_LIST_DIR}/qsa/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kpool_mla/tests.cmake")
