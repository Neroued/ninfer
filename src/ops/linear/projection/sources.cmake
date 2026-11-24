target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/linear.cpp"
)

include("${CMAKE_CURRENT_LIST_DIR}/bf16/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/fp8/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/nvfp4/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/q4/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/q5/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/q6/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/q8/sources.cmake")
