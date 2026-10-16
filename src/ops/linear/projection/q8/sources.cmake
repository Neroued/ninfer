target_sources(ninfer_ops PRIVATE "${CMAKE_CURRENT_LIST_DIR}/q8_dispatch.cpp")
include("${CMAKE_CURRENT_LIST_DIR}/${NINFER_TARGET}/sources.cmake")
