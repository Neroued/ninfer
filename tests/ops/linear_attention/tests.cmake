ninfer_add_op_test(ninfer_kimi_delta_attention_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/kimi_delta_attention.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_causal_conv1d_silu_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/causal_conv1d_silu.cpp"
  LIBRARIES ninfer_ops)

include("${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/control/tests.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gated_delta_net/input_conv/tests.cmake")
