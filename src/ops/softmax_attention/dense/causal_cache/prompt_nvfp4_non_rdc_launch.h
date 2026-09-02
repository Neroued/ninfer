#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void causal_attention_prompt_nvfp4_kernel_launch(const Tensor& q, const Tensor& positions,
                                                 float scale, const PagedKVLayerView& cache,
                                                 Tensor& out, cudaStream_t stream);

void causal_attention_prompt_nvfp4_batch_kernel_launch(const Tensor& q, const Tensor& positions,
                                                       const Tensor& valid_columns,
                                                       const Tensor& table_rows, float scale,
                                                       const PagedKVBatchLayerView& cache,
                                                       Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
