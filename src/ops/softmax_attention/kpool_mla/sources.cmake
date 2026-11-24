target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/layout.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/prepare/prepare.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/indexer/indexer.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sparse_mla/sparse_mla.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/tail_extract.cu")
if(NINFER_TARGET STREQUAL "dgx_spark")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/prepare/dgx_spark/select.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/indexer/dgx_spark/select.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/sparse_mla/dgx_spark/select.cpp")
  target_sources(ninfer_ops_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/prepare/dgx_spark/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/indexer/dgx_spark/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/sparse_mla/dgx_spark/launch.cu")
else()
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/prepare/unsupported.cpp")
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/indexer/unsupported.cpp")
  target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/sparse_mla/unsupported.cpp")
endif()
