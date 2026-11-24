#include "ops/softmax_attention/dense/causal_cache/ragged_launch.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/tiled_launch.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/instances.h"
#include "ops/softmax_attention/kernels/causal/tile_io.cuh"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include "ops/softmax_attention/kernels/causal/mxfp8_tiled_launch.cuh"
#include "ops/softmax_attention/kernels/causal/tiled_merge.cuh"

namespace ninfer::ops::detail {
void fp8_kv_tiled_attention(const CausalAttentionOperands& p, Fp8KvReadView cache,
                            CausalKvPartition partition, MemoryArena& workspace,
                            cudaStream_t stream) {
    auto scope = workspace.scope();
    const auto partial =
        allocate_causal_partials(workspace, p.query_heads, p.width, partition.capacity, 1);
    const auto invoke = [&]<class G>() {
        launch_mxfp8_kv_tiled_mma<G, Fp8KvTiledInstance, Fp8KvTiledValues>(p, cache, partition,
                                                                           partial.view(), stream);
        launch_causal_tiled_merge<G, false>(p, cache.valid_columns, partition, partial.view(),
                                            stream);
    };
    if (p.query_heads == 24)
        invoke.template operator()<CausalD256H24Kv4>();
    else
        invoke.template operator()<CausalD256H16Kv2>();
}

void launch_fp8_ragged_tiled(const CausalAttentionOperands& p, ConstPagedKVBatchLayerView cache,
                             CausalTiledMetadata metadata, CausalKvPartition partition,
                             CausalPartialView partial, cudaStream_t stream) {
    const auto invoke = [&]<class G>() {
        const auto view = make_quantized_causal_cache_view<Fp8KvReadView>(cache);
        if (metadata.tile_rows == kRaggedNarrowTile)
            launch_mxfp8_kv_tiled_mma<G, Fp8KvTiledMmaSchedule<kRaggedNarrowTile, 64>,
                                      Fp8KvTiledValues>(p, view, partition, partial, stream,
                                                        metadata);
        else
            launch_mxfp8_kv_tiled_mma<G, Fp8KvTiledInstance, Fp8KvTiledValues>(
                p, view, partition, partial, stream, metadata);
        if (partition.capacity > 1)
            launch_causal_tiled_merge<G, false>(p, nullptr, partition, partial, stream, metadata);
    };
    if (p.query_heads == 24)
        invoke.template operator()<CausalD256H24Kv4>();
    else
        invoke.template operator()<CausalD256H16Kv2>();
}
} // namespace ninfer::ops::detail
