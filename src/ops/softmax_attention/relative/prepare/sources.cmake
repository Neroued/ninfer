target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/entry.cpp")

# Keep the fused kernel's register allocation through final linkage.
target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/launch.cu")
