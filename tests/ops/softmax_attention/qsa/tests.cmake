ninfer_add_op_test(ninfer_qsa_tail_extract_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tail_extract.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_qsa_prepare_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prepare.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_qsa_indexer_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_indexer.cpp"
  LIBRARIES ninfer_ops)

ninfer_add_op_test(ninfer_qsa_sparse_gqa_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sparse_gqa.cpp"
  LIBRARIES ninfer_ops
  FORMATS bf16 int8 fp8 nvfp4 k8v4)
