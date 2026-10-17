ninfer_add_op_test(ninfer_sampling_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/sampling.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/candidate_selector.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/accept_greedy.cpp"
          "${CMAKE_CURRENT_LIST_DIR}/accept_sparse.cpp"
  LIBRARIES ninfer_ops)
