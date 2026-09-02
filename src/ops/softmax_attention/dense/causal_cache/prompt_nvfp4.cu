// ninfer::ops::detail - public-op composition for group-16 NVFP4 causal prompt attention.
#include "ops/softmax_attention/dense/causal_cache/launch.h"

#include "ops/kv_cache/append/launch.h"
#include "ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc_launch.h"

namespace ninfer::ops::detail {

void causal_attention_prompt_nvfp4_attention_launch(const Tensor& q, const Tensor& positions,
                                                    float scale, const PagedKVLayerView& cache,
                                                    Tensor& out, cudaStream_t stream) {
    causal_attention_prompt_nvfp4_kernel_launch(q, positions, scale, cache, out, stream);
}

void causal_attention_prompt_nvfp4_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                          const Tensor& positions, const Tensor& valid_columns,
                                          const Tensor& table_rows, float scale,
                                          PagedKVBatchLayerView cache, Tensor& out,
                                          cudaStream_t stream) {
    kv_cache_append_batch_launch(k, v, positions, valid_columns, table_rows, cache, stream);
    causal_attention_prompt_nvfp4_batch_kernel_launch(q, positions, valid_columns, table_rows,
                                                      scale, cache, out, stream);
}

} // namespace ninfer::ops::detail
