target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/n12288_k4096.cu")
target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/n12288_k4096_tma.cu")
target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/n4096_k6144.cu")
target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/n4096_k6144_tma.cu")
