target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n248320_k2560.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n124160_k2560.cu"
)
target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/n77440_k4096.cu")
