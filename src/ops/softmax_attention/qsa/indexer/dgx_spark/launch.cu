#include "ops/softmax_attention/qsa/indexer/plan.h"
#include "ops/common/host/launch.h"
#include "ops/softmax_attention/qsa/indexer/dgx_spark/score.cuh"
#include "ops/softmax_attention/qsa/indexer/dgx_spark/simt.cuh"
#include "ops/softmax_attention/qsa/indexer/select.cuh"

namespace ninfer::ops::detail::qsa {
namespace {
void launch_finish(IndexerArguments args, cudaStream_t stream) {
    const auto submit = [&]<int Parts>() {
        indexer_finish_kernel<kIndexerPartition, 6, 2, Parts>
            <<<args.width * args.batch, 256, 0, stream>>>(args);
    };
    switch (args.partitions) {
    case 0:
    case 1:
        return submit.template operator()<1>();
    case 2:
        return submit.template operator()<2>();
    case 3:
        return submit.template operator()<3>();
    case 4:
        return submit.template operator()<4>();
    case 5:
        return submit.template operator()<5>();
    case 6:
        return submit.template operator()<6>();
    case 7:
        return submit.template operator()<7>();
    case 8:
        return submit.template operator()<8>();
    case 9:
        return submit.template operator()<9>();
    case 10:
        return submit.template operator()<10>();
    case 11:
        return submit.template operator()<11>();
    case 12:
        return submit.template operator()<12>();
    case 13:
        return submit.template operator()<13>();
    case 14:
        return submit.template operator()<14>();
    case 15:
        return submit.template operator()<15>();
    default:
        return submit.template operator()<16>();
    }
}

template <int Tile, bool Vector, int Stages, bool RetainKeys = false>
void launch_score(IndexerArguments args, IndexerWindowPlan plan, cudaStream_t stream) {
    constexpr int threads = Tile > 16 ? Tile * 8 : 128;
    constexpr auto kernel = indexer_score_pipeline_kernel<Tile, Vector, Stages, RetainKeys>;
    const int shared = common::prepare_shared<Stages * kIndexerKeyTile * 128 * 2, kernel, true>();
    kernel<<<plan.score_workers, threads, shared, stream>>>(args);
}

template <bool Vector>
void launch_score(IndexerArguments args, IndexerWindowPlan plan, cudaStream_t stream) {
    if (plan.score == IndexerScore::Simt16) {
        indexer_score_simt_kernel<16, 128, Vector><<<plan.score_workers, 128, 0, stream>>>(args);
        return;
    }
    if (plan.score == IndexerScore::Simt8) {
        indexer_score_simt_kernel<8, 128, Vector><<<plan.score_workers, 128, 0, stream>>>(args);
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
    case 128:
        if constexpr (Vector) {
            if (plan.retain_keys) {
                launch_score<128, true, 2, true>(args, plan, stream);
                break;
            }
        }
        launch_score<128, Vector, 2>(args, plan, stream);
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
    const auto partition = [&]<int Queries>() {
        if (plan.reverse_selection)
            indexer_partition_kernel<kIndexerPartition, 6, 2, 4, Queries, true>
                <<<plan.select_workers, 256, 0, execution.stream()>>>(args);
        else
            indexer_partition_kernel<kIndexerPartition, 6, 2, 4, Queries>
                <<<plan.select_workers, 256, 0, execution.stream()>>>(args);
    };
    if (args.width * args.batch <= 64)
        partition.template operator()<64>();
    else
        partition.template operator()<128>();
    launch_finish(args, execution.stream());
}
} // namespace ninfer::ops::detail::qsa
