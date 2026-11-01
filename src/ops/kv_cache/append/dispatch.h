#pragma once

#include "ninfer/ops/kv_cache_append.h"

namespace ninfer::ops::detail {

void kv_cache_append_dispatch(const Tensor& k, const Tensor& v, const Tensor& positions,
                              PagedKVLayerView cache, cudaStream_t stream);
void kv_cache_append_prefix_dispatch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                     const Tensor& counts, const Tensor& table_rows,
                                     PagedKVBatchLayerView cache,
                                     KVCacheAppendPrefixExecutionEnvelope envelope,
                                     cudaStream_t stream);
void kv_cache_append_prefix_dispatch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                     const Tensor& counts, CyclicKVCacheLayerView cache,
                                     KVCacheAppendPrefixExecutionEnvelope envelope,
                                     cudaStream_t stream);

} // namespace ninfer::ops::detail
