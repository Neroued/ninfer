if(NOT WIN32)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
     NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
    message(FATAL_ERROR "NInfer CPU execution currently requires Linux x86-64 or Windows")
  endif()
endif()

add_library(ninfer_cpu STATIC
  cpu/task_pool.cpp
  cpu/executor.cpp
)
if(WIN32)
  target_sources(ninfer_cpu PRIVATE cpu/system_win.cpp)
else()
  target_sources(ninfer_cpu PRIVATE cpu/system.cpp)
endif()
ninfer_internal_includes(ninfer_cpu)
if(WIN32)
  target_link_libraries(ninfer_cpu PRIVATE Threads::Threads)
else()
  target_link_libraries(ninfer_cpu PRIVATE PkgConfig::HWLOC Threads::Threads)
endif()

add_library(ninfer_core STATIC
  arena.cpp
  buffer.cpp
  buffer.cu
  memory.cpp
  transfer.cpp
  transfer.cu
  device.cu
  dtype.cpp
  nf4.cpp
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
if(WIN32)
  target_sources(ninfer_core PRIVATE io/file_win.cpp)
else()
  target_sources(ninfer_core PRIVATE io/file.cpp io/read_queue.cpp)
endif()
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC ninfer_cpu CUDA::cudart CUDA::nvtx3 Threads::Threads)
if(NOT WIN32)
  target_link_libraries(ninfer_core PRIVATE PkgConfig::URING)
endif()