ninfer_add_op_test(ninfer_sequence_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/ragged_prefix.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/masked_block.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/autoregressive_round.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/verify.cpp"
  LIBRARIES ninfer_ops)
