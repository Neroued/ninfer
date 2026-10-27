target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/tail_extract.cu"
  "${CMAKE_CURRENT_LIST_DIR}/prepare/prepare.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/prepare/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/indexer/indexer.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/indexer/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/sparse_gqa.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/launch.cu"
)

target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/indexer/${NINFER_TARGET}/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/${NINFER_TARGET}/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/${NINFER_TARGET}/launch.cu"
)

if(NINFER_TARGET STREQUAL "dgx_spark")
  target_sources(ninfer_ops_non_rdc PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/indexer/dgx_spark/launch.cu"
    "${CMAKE_CURRENT_LIST_DIR}/sparse_gqa/dgx_spark/grouped.cu")
else()
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/indexer/${NINFER_TARGET}/launch.cu")
endif()
