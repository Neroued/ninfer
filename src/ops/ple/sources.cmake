target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/lookup.cu"
  "${CMAKE_CURRENT_LIST_DIR}/operands.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/gate.cu"
  "${CMAKE_CURRENT_LIST_DIR}/conv.cu"
  "${CMAKE_CURRENT_LIST_DIR}/state_extract.cu")
