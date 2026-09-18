# Immutable loading and mutable execution have separate compile owners.
add_library(ninfer_model_loading STATIC registry.cpp)
ninfer_internal_includes(ninfer_model_loading)
target_link_libraries(ninfer_model_loading PUBLIC ninfer_artifact ninfer_text ninfer_ops)

add_library(ninfer_model_runtime STATIC)
ninfer_internal_includes(ninfer_model_runtime)
target_link_libraries(ninfer_model_runtime
  PUBLIC ninfer_model_loading ninfer_ops ninfer_runtime_support
  PRIVATE ninfer_media_decode ninfer_jinja)

include(qwen3_5/loading_sources.cmake)
include(qwen3_5/frontend_sources.cmake)
include(qwen3_5/execution_sources.cmake)
include(qwen3_5/program_sources.cmake)
