if(NINFER_TARGET STREQUAL "rtx5090")
  ninfer_add_op_test(ninfer_loop_message_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loop_message.cpp"
    LIBRARIES ninfer_ops)
endif()
