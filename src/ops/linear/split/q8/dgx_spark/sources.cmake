target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k2560.cu")
target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/n16384_k2560_k32.cu")
