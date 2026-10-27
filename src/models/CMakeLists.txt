# Immutable loading and mutable execution have separate compile owners.
add_library(ninfer_model_loading STATIC registry.cpp common/bindings.cpp common/weights.cpp)
if(NOT WIN32)
  target_sources(ninfer_model_loading PRIVATE
    qwen4_exp/ple/parameters.cpp
    qwen4_exp/ple/disk/table.cpp)
endif()
ninfer_internal_includes(ninfer_model_loading)
target_link_libraries(ninfer_model_loading PUBLIC ninfer_artifact ninfer_text ninfer_ops)

add_library(ninfer_model_runtime STATIC)
if(NOT WIN32)
  target_sources(ninfer_model_runtime PRIVATE
    qwen4_exp/ple/hash.cpp
    qwen4_exp/ple/upload.cpp
    qwen4_exp/ple/disk/row_reader.cpp
    qwen4_exp/ple/disk/prepare.cpp
  )
endif()
ninfer_internal_includes(ninfer_model_runtime)
target_link_libraries(ninfer_model_runtime
  PUBLIC ninfer_model_loading ninfer_ops ninfer_runtime_support
  PRIVATE ninfer_media_decode ninfer_jinja ninfer_grammar ninfer_xgrammar)
target_include_directories(ninfer_model_runtime PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xgrammar/cpp)
target_include_directories(ninfer_model_runtime SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xgrammar/3rdparty/picojson)

target_sources(ninfer_model_loading PRIVATE common/moe/weights.cpp)
target_sources(ninfer_model_runtime PRIVATE
  common/context/state_store.cpp
  common/context/logical_kv_store.cpp
  common/context/host_kv_store.cpp
  common/context/kv_history.cpp
  common/context/work.cpp
  common/context/migration.cpp
  common/moe/graph.cpp
  common/moe/plan.cpp
  common/moe/execution.cpp
  common/moe/resources.cpp)

include(qwen3_5/loading_sources.cmake)
if(NOT WIN32)
  include(qwen4_exp/loading_sources.cmake)
endif()
include(common/qwen/sources.cmake)
include(qwen3_5/execution_sources.cmake)
if(NOT WIN32)
  include(qwen4_exp/execution_sources.cmake)
endif()
include(qwen3_5/program_sources.cmake)
