add_library(ninfer_runtime_support STATIC
  engine/context_cache/context_cost.cpp
  engine/context_cache/context_cost_defaults.cpp
  engine/kv_capacity.cpp
  engine/startup_memory.cpp
  engine/public_types.cpp
)
ninfer_internal_includes(ninfer_runtime_support)
target_link_libraries(ninfer_runtime_support PUBLIC ninfer_core PRIVATE ninfer::json)

# Public Engine and common contracts; model execution remains in the model layer.
add_library(ninfer_engine STATIC
  engine/startup.cpp
  contract/sampling.cpp
  engine/engine.cpp
)
ninfer_internal_includes(ninfer_engine)
target_include_directories(ninfer_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ninfer_engine PRIVATE
  ninfer_deployment
  ninfer_model_execution
  ninfer_artifact
  ninfer_core
  ninfer_runtime_support
  ninfer_ops
  ninfer_text
  ninfer_media_decode)
add_library(ninfer::engine ALIAS ninfer_engine)

if(NINFER_ENABLE_DISTRIBUTED)
  add_library(ninfer_runtime_distributed STATIC executor/protocol.cpp executor/resources.cpp)
  ninfer_internal_includes(ninfer_runtime_distributed)
  target_link_libraries(ninfer_runtime_distributed PUBLIC ninfer_distributed_control)
  target_link_libraries(ninfer_engine PRIVATE ninfer_runtime_distributed)
endif()
