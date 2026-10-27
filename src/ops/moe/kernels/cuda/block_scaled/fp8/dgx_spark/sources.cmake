target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/launch.cu")

target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../block64/dgx_spark/launch.cu")
