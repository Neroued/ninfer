# Immutable loading and mutable execution have separate compile owners.
add_library(ninfer_model_loading STATIC registry.cpp common/bindings.cpp common/weights.cpp)
target_sources(ninfer_model_loading PRIVATE
  qwen4_exp/ple/parameters.cpp
  qwen4_exp/ple/disk/table.cpp)
ninfer_internal_includes(ninfer_model_loading)
target_link_libraries(ninfer_model_loading PUBLIC ninfer_artifact ninfer_text ninfer_ops)

add_library(ninfer_model_runtime STATIC)
target_sources(ninfer_model_runtime PRIVATE
  qwen4_exp/ple/hash.cpp
  qwen4_exp/ple/upload.cpp
  qwen4_exp/ple/disk/row_reader.cpp
  qwen4_exp/ple/disk/prepare.cpp
)
ninfer_internal_includes(ninfer_model_runtime)
target_compile_definitions(ninfer_model_runtime PUBLIC
  NINFER_SHARED_PHYSICAL_MEMORY=${NINFER_SHARED_PHYSICAL_MEMORY})
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
  common/moe/execution.cpp
  common/moe/schedule.cpp)

if(NINFER_TARGET STREQUAL "rtx5090")
  target_compile_definitions(ninfer_model_runtime PRIVATE NINFER_EXPERT_OFFLOAD=1)
  target_link_libraries(ninfer_model_runtime PRIVATE ninfer_cpu_executor)
  target_sources(ninfer_model_runtime PRIVATE
    common/moe/offload/plan.cpp
    common/moe/offload/resources.cpp
    common/moe/offload/execution.cpp
    common/moe/offload/graph.cpp
    common/moe/offload/schedule.cpp)
else()
  target_compile_definitions(ninfer_model_runtime PRIVATE NINFER_EXPERT_OFFLOAD=0)
endif()

include(qwen3_5/loading_sources.cmake)
include(qwen4_exp/loading_sources.cmake)
include(common/qwen/sources.cmake)
include(qwen3_5/execution_sources.cmake)
include(qwen4_exp/execution_sources.cmake)
include(qwen3_5/program_sources.cmake)
