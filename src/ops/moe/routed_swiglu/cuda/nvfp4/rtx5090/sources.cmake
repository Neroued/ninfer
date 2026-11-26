target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/shapes/h2560_i640_e512_k10/dispatch.cpp")
target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/shapes/h2304_i1152_e128_k8/dispatch.cpp")
target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/softcap.cu")
target_sources(ninfer_ops_non_rdc PRIVATE "${CMAKE_CURRENT_LIST_DIR}/softcap_tma.cu")
