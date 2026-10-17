#include "ops/softmax_attention/qsa/indexer/plan.h"
#include "ops/common/host/launch.h"
#include "ops/softmax_attention/qsa/indexer/score.cuh"
#include "ops/softmax_attention/qsa/indexer/select.cuh"

namespace ninfer::ops::detail::qsa {
namespace {
template <int QueryTile, bool VectorCache>
void launch_score(IndexerArguments a, IndexerWindowPlan plan, cudaStream_t stream) {
    constexpr int threads = QueryTile > 16 ? QueryTile * 8 : 128;
    constexpr int bytes   = (QueryTile * 4 + kIndexerKeyTile) * 128 * sizeof(__nv_bfloat16);
    constexpr auto kernel = indexer_score_mma_kernel<QueryTile, VectorCache>;
    const int shared      = common::prepare_shared<bytes, kernel, true>();
    kernel<<<plan.score_workers, threads, shared, stream>>>(a);
}

template <bool VectorCache>
void launch_score(IndexerArguments a, IndexerWindowPlan plan, cudaStream_t stream) {
    if (plan.query_tile == 4)
        launch_score<4, VectorCache>(a, plan, stream);
    else if (plan.query_tile == 8)
        launch_score<8, VectorCache>(a, plan, stream);
    else if (plan.query_tile == 16)
        launch_score<16, VectorCache>(a, plan, stream);
    else
        launch_score<32, VectorCache>(a, plan, stream);
}
} // namespace

void launch_indexer_window(IndexerArguments a, IndexerWindowPlan plan, bool vector_cache,
                           CudaExecutionView execution) {
    if (vector_cache)
        launch_score<true>(a, plan, execution.stream());
    else
        launch_score<false>(a, plan, execution.stream());
    indexer_partition_kernel<kIndexerPartition>
        <<<plan.select_workers, 256, 0, execution.stream()>>>(a);
    indexer_finish_kernel<kIndexerPartition><<<a.width * a.batch, 256, 0, execution.stream()>>>(a);
}
} // namespace ninfer::ops::detail::qsa
