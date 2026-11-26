target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/entry.cpp")
target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/launch.cu")
target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/fused_attn_res.cu")
