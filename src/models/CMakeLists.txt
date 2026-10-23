# Definitions and immutable weights are independent of execution ownership.
add_library(ninfer_model_loading STATIC
  registry.cpp
  common/frontend/resources.cpp
  glm5next/frontend/resources.cpp
  glm5next/frontend/processor_config.cpp
  common/bindings.cpp
  common/parallel.cpp
  common/weights.cpp
  common/moe/weights.cpp
  qwen4_exp/ple/parameters.cpp
  qwen4_exp/ple/disk/table.cpp)
ninfer_internal_includes(ninfer_model_loading)
target_link_libraries(ninfer_model_loading PUBLIC ninfer_artifact ninfer_text ninfer_ops ninfer_deployment)

# Coordinator-owned sessions, context directories and frontend policy.
add_library(ninfer_model_control STATIC
  common/frontend/chat_template.cpp
  common/frontend/prompt.cpp
  glm5next/frontend/chat_template.cpp
  glm5next/frontend/prompt_layout.cpp
  glm5next/frontend/frontend.cpp
  glm5next/frontend/output.cpp
  glm5next/frontend/tool_contract.cpp
  glm5next/frontend/tool_decoder.cpp
  glm5next/frontend/tool_grammar.cpp
  glm5next/prefix_identity.cpp
  common/frontend/output_resources.cpp
  common/frontend/output_session.cpp
  common/frontend/tools.cpp
  common/frontend/image_transform.cpp
  glm5next/frontend/processor.cpp
  common/frontend/media_cache.cpp
  common/frontend/digest.cpp
  common/context/directory.cpp
  common/context/directory/records.cpp
  common/context/directory/state.cpp
  common/context/directory/pages.cpp
  common/context/directory/history.cpp
  common/context/work.cpp)
ninfer_internal_includes(ninfer_model_control)
target_link_libraries(ninfer_model_control
  PUBLIC ninfer_model_loading ninfer_runtime_support
  PRIVATE ninfer_media_decode ninfer_jinja ninfer_grammar ninfer_xgrammar)
target_include_directories(ninfer_model_control PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xgrammar/cpp)
target_include_directories(ninfer_model_control SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/xgrammar/3rdparty/picojson)

# Rank-owned storage, mathematical execution and graph preparation.
add_library(ninfer_model_execution STATIC
  common/context/storage.cpp
  common/context/transfer.cpp
  common/moe/execution.cpp
  common/moe/schedule.cpp
  qwen4_exp/ple/hash.cpp
  qwen4_exp/ple/upload.cpp
  qwen4_exp/ple/disk/row_reader.cpp
  qwen4_exp/ple/disk/prepare.cpp)
ninfer_internal_includes(ninfer_model_execution)
target_link_libraries(ninfer_model_execution
  PUBLIC ninfer_model_loading ninfer_model_control ninfer_ops ninfer_runtime_support)

if(NINFER_TARGET STREQUAL "rtx5090")
  target_link_libraries(ninfer_model_execution PRIVATE ninfer_cpu_executor)
  target_sources(ninfer_model_execution PRIVATE
    common/moe/offload/plan.cpp
    common/moe/offload/resources.cpp
    common/moe/offload/execution.cpp
    common/moe/offload/graph.cpp
    common/moe/offload/schedule.cpp)
endif()

include(qwen3_5/loading_sources.cmake)
include(qwen4_exp/loading_sources.cmake)
include(common/qwen/sources.cmake)
include(qwen3_5/execution_sources.cmake)
include(qwen4_exp/execution_sources.cmake)
include(qwen3_5/program_sources.cmake)
include(qwen3_5/session_sources.cmake)
include(qwen4_exp/session_sources.cmake)
