// Non-RDC compilation is required for the producer/consumer register redistribution.
#include "ops/softmax_attention/dense/causal_cache/nvfp4/tiled_launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/instances.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/tiled_launch.cuh"
#include "ops/softmax_attention/common/causal_tiled_merge.cuh"

namespace ninfer::ops::detail {
void nvfp4_kv_tiled_attention(const CausalAttentionOperands& p, Nvfp4KvReadView cache,
                              CausalKvPartition partition, WorkspaceArena& workspace,
                              cudaStream_t stream) {
    auto scope = workspace.scope();
    const auto partial =
        allocate_causal_partials(workspace, p.query_heads, p.width, partition.capacity, 1);
    const auto invoke = [&]<class G>() {
        launch_nvfp4_kv_tiled_mma<G, Nvfp4KvTiledInstance>(p, cache, partition, partial.view(),
                                                          stream);
        launch_causal_tiled_merge<G, true>(p, cache.valid_columns, partition, partial.view(),
                                           stream);
    };
    if (p.query_heads == 24)
        invoke.template operator()<CausalD256H24Kv4>();
    else
        invoke.template operator()<CausalD256H16Kv2>();
}
} // namespace ninfer::ops::detail
