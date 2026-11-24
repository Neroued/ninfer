// Non-RDC compilation is required for the producer/consumer register redistribution.
#include "ops/softmax_attention/dense/causal_cache/ragged_launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/tiled_launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/instances.h"
#include "ops/softmax_attention/kernels/causal/nvfp4/tiled_launch.cuh"

namespace ninfer::ops::detail {
void nvfp4_kv_tiled_attention(const CausalAttentionOperands& p, Nvfp4KvReadView cache,
                              cudaStream_t stream) {
    if (p.query_heads == 24)
        launch_nvfp4_kv_tiled_mma<CausalD256H24Kv4, Nvfp4KvTiledInstance>(p, cache, stream);
    else
        launch_nvfp4_kv_tiled_mma<CausalD256H16Kv2, Nvfp4KvTiledInstance>(p, cache, stream);
}

void launch_nvfp4_ragged_tiled(const CausalAttentionOperands& p, ConstPagedKVBatchLayerView cache,
                               CausalTiledMetadata metadata, CausalKvPartition, CausalPartialView,
                               cudaStream_t stream) {
    const auto invoke = [&]<class G>() {
        const auto view = make_quantized_causal_cache_view<Nvfp4KvReadView>(cache);
        launch_nvfp4_kv_tiled_mma<G, Nvfp4KvTiledInstance>(p, view, stream, metadata);
    };
    if (p.query_heads == 24)
        invoke.template operator()<CausalD256H24Kv4>();
    else
        invoke.template operator()<CausalD256H16Kv2>();
}
} // namespace ninfer::ops::detail
