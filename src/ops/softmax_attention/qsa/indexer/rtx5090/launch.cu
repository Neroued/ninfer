#include "ops/softmax_attention/qsa/indexer/plan.h"
#include "ops/common/host/launch.h"
#include "ops/common/host/operands.h"
#include "ops/softmax_attention/qsa/indexer/score.cuh"
#include "ops/softmax_attention/qsa/indexer/select.cuh"

namespace ninfer::ops::detail::qsa {
namespace {
template <int QueryTile, bool VectorCache, int MaxQueries, bool BatchedLoop>
void launch_score(IndexerArguments a, IndexerWindowPlan plan, cudaStream_t stream) {
    constexpr int threads = QueryTile > 16 ? QueryTile * 8 : 128;
    constexpr int bytes   = (QueryTile * 4 + kIndexerKeyTile) * 128 * sizeof(__nv_bfloat16);
    constexpr auto kernel =
        indexer_score_mma_kernel<QueryTile, VectorCache, MaxQueries, BatchedLoop>;
    const int shared = common::prepare_shared<bytes, kernel, true>();
    kernel<<<plan.score_workers, threads, shared, stream>>>(a);
}

template <bool VectorCache, int MaxQueries, bool BatchedLoop>
void launch_score(IndexerArguments a, IndexerWindowPlan plan, cudaStream_t stream) {
    if (plan.query_tile == 4)
        launch_score<4, VectorCache, MaxQueries, BatchedLoop>(a, plan, stream);
    else if (plan.query_tile == 8)
        launch_score<8, VectorCache, MaxQueries, BatchedLoop>(a, plan, stream);
    else if (plan.query_tile == 16)
        launch_score<16, VectorCache, MaxQueries, BatchedLoop>(a, plan, stream);
    else
        launch_score<32, VectorCache, MaxQueries, BatchedLoop>(a, plan, stream);
}

void launch_window(IndexerArguments a, IndexerWindowPlan plan, bool vector_cache,
                   CudaExecutionView execution) {
    const auto stream = execution.stream();
    if (a.partitions == 0) {
        launch_indexer_select_all(a, plan.select_workers, stream);
        return;
    }
    const auto submit = [&]<int MaxQueries, bool BatchedLoop>() {
        if (vector_cache)
            launch_score<true, MaxQueries, BatchedLoop>(a, plan, stream);
        else
            launch_score<false, MaxQueries, BatchedLoop>(a, plan, stream);
        if (a.partitions == 1)
            launch_indexer_single_select<8, 8>(a, plan.select_workers, stream);
        else {
            indexer_partition_kernel<kIndexerPartition, 8, 8, 1, MaxQueries, false, BatchedLoop>
                <<<plan.select_workers, 256, 0, stream>>>(a);
            launch_indexer_finish<8, 8>(a, stream);
        }
    };
    const int slots = a.width * a.batch;
    if (slots > kIndexerMaxWindowQueries)
        submit.template operator()<kIndexerMaxWindowQueries, true>();
    else if (slots > 64)
        submit.template operator()<kIndexerMaxWindowQueries, false>();
    else
        submit.template operator()<64, false>();
}
} // namespace

void launch_indexer_windows(IndexerArguments input, IndexerPlan plan, std::size_t,
                            std::size_t key_alignment, CudaExecutionView execution) {
    const bool vector_cache = common::operand::aligned(input.keys, 16) &&
                              input.key_row_stride % 8 == 0 && key_alignment % 16 == 0;
    for_each_indexer_window(input, plan, [&](IndexerArguments a) {
        const auto window =
            select_indexer_window(plan, a.width, a.batch, execution.multiprocessor_count(), false);
        launch_window(a, window, vector_cache, execution);
    });
}
} // namespace ninfer::ops::detail::qsa
