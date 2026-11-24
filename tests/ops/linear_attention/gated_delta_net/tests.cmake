ninfer_add_op_test(ninfer_gated_delta_net_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_recurrence.cpp"
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gated_delta_net_replay_record_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_record.cpp"
  LIBRARIES ninfer_ops)
ninfer_add_op_test(ninfer_gdn_replay_fold_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fold.cpp"
  LIBRARIES ninfer_ops)
