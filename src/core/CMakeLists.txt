add_library(ninfer_cpu STATIC
  cpu/system.cpp
  cpu/task_pool.cpp
)
ninfer_internal_includes(ninfer_cpu)
target_link_libraries(ninfer_cpu PRIVATE PkgConfig::HWLOC Threads::Threads)

if(NINFER_TARGET STREQUAL "rtx5090")
  add_library(ninfer_cpu_executor STATIC cpu/executor.cpp cpu/x86.cpp)
  ninfer_internal_includes(ninfer_cpu_executor)
  target_link_libraries(ninfer_cpu_executor PUBLIC ninfer_cpu)
endif()

add_library(ninfer_core STATIC
  arena.cpp
  buffer.cpp
  buffer.cu
  memory.cpp
  transfer.cpp
  transfer.cu
  device.cu
  device_memory.cpp
  dtype.cpp
  nf4.cpp
  io/file.cpp
  io/read_queue.cpp
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_context_arena.cpp
  host_kv_arena.cpp
  paged_kv_cache.cpp
  linear_attention_state.cpp
  state_image.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  weight_bank.cpp
  expert_cache.cpp
  decode_graph.cpp
)
ninfer_internal_includes(ninfer_core)
target_compile_definitions(ninfer_core PRIVATE
  NINFER_SHARED_PHYSICAL_MEMORY=${NINFER_SHARED_PHYSICAL_MEMORY})
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC ninfer_cpu CUDA::cudart CUDA::nvtx3 Threads::Threads)
target_link_libraries(ninfer_core PRIVATE PkgConfig::URING)
