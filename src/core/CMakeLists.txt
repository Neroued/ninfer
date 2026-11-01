add_library(ninfer_core STATIC
  arena.cu
  device.cu
  dtype.cpp
  host_worker_pool.cpp
  gdn_replay_records.cpp
  cyclic_kv_cache.cpp
  host_kv_arena.cpp
  disk_kv_cache.cpp
  paged_kv_cache.cpp
  linear_attention_state.cpp
  layout.cpp
  tensor.cpp
  weight_view.cpp
  decode_graph.cpp
  expert_cache_policy.cpp
  expert_cache.cu
  expert_staging.cu
  elastic_arena.cu
  uva_copy.cu
)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC CUDA::cudart CUDA::nvtx3 Threads::Threads PRIVATE CUDA::cuda_driver)
