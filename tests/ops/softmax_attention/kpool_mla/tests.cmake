if(NINFER_TARGET STREQUAL "dgx_spark")
  ninfer_add_op_test(ninfer_kpool_chain_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_chain.cpp"
    LIBRARIES ninfer_ops)
  ninfer_add_op_test(ninfer_kpool_sparse_mla_test
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_sparse_mla.cpp"
    LIBRARIES ninfer_ops)
  ninfer_add_op_test(ninfer_kpool_indexer_test
    PROCESSORS 2 COST 80
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_indexer.cpp"
    LIBRARIES ninfer_ops CUDA::cuda_driver)
  ninfer_add_op_test(ninfer_kpool_prepare_test
    PROCESSORS 2 COST 80
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prepare.cpp"
    LIBRARIES ninfer_ops)
endif()
ninfer_add_op_test(ninfer_kpool_tail_extract_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tail_extract.cpp"
  LIBRARIES ninfer_ops)
