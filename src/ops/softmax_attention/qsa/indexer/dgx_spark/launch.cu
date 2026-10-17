#include "ops/softmax_attention/qsa/indexer/plan.h"
#include "ops/common/host/launch.h"
#include "ops/softmax_attention/qsa/indexer/dgx_spark/score.cuh"
#include "ops/softmax_attention/qsa/indexer/dgx_spark/simt.cuh"
#include "ops/softmax_attention/qsa/indexer/select.cuh"

namespace ninfer::ops::detail::qsa {
namespace {
template <int Tile, bool Vector, int Stages>
void launch_score(IndexerArguments args, IndexerWindowPlan plan, cudaStream_t stream) {
    constexpr int threads = Tile > 16 ? Tile * 8 : 128;
    constexpr auto kernel = indexer_score_pipeline_kernel<Tile, Vector, Stages>;
    const int shared = common::prepare_shared<Stages * kIndexerKeyTile * 128 * 2, kernel, true>();
    kernel<<<plan.score_workers, threads, shared, stream>>>(args);
}

template <bool Vector>
void launch_score(IndexerArguments args, IndexerWindowPlan plan, cudaStream_t stream) {
    if (plan.simt) {
        indexer_score_simt_kernel<16, 128, Vector><<<plan.score_workers, 128, 0, stream>>>(args);
        return;
    }
    const auto submit = [&]<int Tile>() {
        if (plan.stages == 2) {
            launch_score<Tile, Vector, 2>(args, plan, stream);
            return;
        }
        launch_score<Tile, Vector, 1>(args, plan, stream);
    };
    switch (plan.query_tile) {
    case 4:
        submit.template operator()<4>();
        break;
    case 8:
        submit.template operator()<8>();
        break;
    case 16:
        submit.template operator()<16>();
        break;
    case 32:
        submit.template operator()<32>();
        break;
    case 64:
        submit.template operator()<64>();
        break;
    }
}
} // namespace

void launch_indexer_window(IndexerArguments args, IndexerWindowPlan plan, bool vector_cache,
                           CudaExecutionView execution) {
    if (vector_cache)
        launch_score<true>(args, plan, execution.stream());
    else
        launch_score<false>(args, plan, execution.stream());
    const auto select = [&]<int Bits, int Copies>() {
        indexer_partition_kernel<kIndexerPartition, Bits, Copies, 4>
            <<<plan.select_workers, 256, 0, execution.stream()>>>(args);
        indexer_finish_kernel<kIndexerPartition, Bits, Copies>
            <<<args.width * args.batch, 256, 0, execution.stream()>>>(args);
    };
    if (plan.compact_radix)
        select.template operator()<6, 2>();
    else
        select.template operator()<8, 8>();
}
} // namespace ninfer::ops::detail::qsa
