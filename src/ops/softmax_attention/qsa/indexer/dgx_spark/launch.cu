#include "ops/softmax_attention/qsa/indexer/plan.h"
#include "ops/common/host/launch.h"
#include "ops/common/host/operands.h"
#include "ops/softmax_attention/qsa/indexer/dgx_spark/score.cuh"
#include "ops/softmax_attention/qsa/indexer/dgx_spark/score_tma.cuh"
#include "ops/softmax_attention/qsa/indexer/dgx_spark/simt.cuh"
#include "ops/softmax_attention/qsa/indexer/select.cuh"

#include <limits>
#include <optional>

namespace ninfer::ops::detail::qsa {
namespace {
template <int Tile, bool Vector, int Stages, bool RetainKeys = false, int MapCapacity = 32>
void launch_score(IndexerArguments args, IndexerWindowPlan plan, cudaStream_t stream) {
    constexpr int threads = Tile > 16 ? Tile * 8 : 128;
    constexpr auto kernel =
        indexer_score_pipeline_kernel<Tile, Vector, Stages, RetainKeys, MapCapacity>;
    const int shared = common::prepare_shared<Stages * kIndexerKeyTile * 128 * 2, kernel, true>();
    kernel<<<plan.score_workers, threads, shared, stream>>>(args);
}

template <bool Vector>
void launch_score(IndexerArguments args, IndexerWindowPlan plan, cudaStream_t stream) {
    if (plan.score == IndexerScore::Simt16) {
        const int slots = args.width * args.batch;
        if (slots > kIndexerMaxWindowQueries)
            indexer_score_simt_kernel<16, 128, Vector, kIndexerMaxWindowQueries, true>
                <<<plan.score_workers, 128, 0, stream>>>(args);
        else if (slots > 64)
            indexer_score_simt_kernel<16, 128, Vector, kIndexerMaxWindowQueries, false>
                <<<plan.score_workers, 128, 0, stream>>>(args);
        else
            indexer_score_simt_kernel<16, 128, Vector, 64, false>
                <<<plan.score_workers, 128, 0, stream>>>(args);
        return;
    }
    if (plan.score == IndexerScore::Simt8) {
        indexer_score_simt_kernel<8, 128, Vector><<<plan.score_workers, 128, 0, stream>>>(args);
        return;
    }
    const auto submit = [&]<int Tile>() {
        if constexpr (Tile == 4) {
            if (args.batch > 32) {
                if (plan.stages == 2)
                    launch_score<Tile, Vector, 2, false, kIndexerMaxWindowQueries>(args, plan,
                                                                                   stream);
                else
                    launch_score<Tile, Vector, 1, false, kIndexerMaxWindowQueries>(args, plan,
                                                                                   stream);
                return;
            }
        }
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

void launch_window(IndexerArguments args, IndexerWindowPlan plan, bool vector_cache,
                   const CUtensorMap* map, CudaExecutionView execution) {
    const auto stream = execution.stream();
    if (args.partitions == 0) {
        launch_indexer_select_all(args, plan.select_workers, stream);
        return;
    }
    if (plan.score == IndexerScore::Tma) {
        const int shared =
            common::prepare_shared<kIndexerTmaSharedBytes, indexer_score_tma_kernel, true>();
        indexer_score_tma_kernel<<<plan.score_workers, 128, shared, stream>>>(args, *map);
    } else if (vector_cache)
        launch_score<true>(args, plan, stream);
    else
        launch_score<false>(args, plan, stream);
    if (args.partitions == 1) {
        launch_indexer_single_select<6, 2>(args, plan.select_workers, stream);
        return;
    }
    const auto partition = [&]<int Queries, bool BatchedLoop>() {
        if (plan.reverse_selection)
            indexer_partition_kernel<kIndexerPartition, 6, 2, 4, Queries, true, BatchedLoop>
                <<<plan.select_workers, 256, 0, stream>>>(args);
        else
            indexer_partition_kernel<kIndexerPartition, 6, 2, 4, Queries, false, BatchedLoop>
                <<<plan.select_workers, 256, 0, stream>>>(args);
    };
    const int slots = args.width * args.batch;
    if (slots <= 64)
        partition.template operator()<64, false>();
    else if (slots <= kIndexerMaxWindowQueries)
        partition.template operator()<kIndexerMaxWindowQueries, false>();
    else
        partition.template operator()<kIndexerMaxWindowQueries, true>();
    launch_indexer_finish<6, 2>(args, stream);
}
} // namespace

void launch_indexer_windows(IndexerArguments input, IndexerPlan plan, std::size_t key_bytes,
                            std::size_t key_alignment, CudaExecutionView execution) {
    const bool vector_cache = common::operand::aligned(input.keys, 16) &&
                              input.key_row_stride % 8 == 0 && key_alignment % 16 == 0;
    const bool tma_cache = vector_cache && common::operand::aligned(input.keys, 128) &&
                           key_alignment % 128 == 0 &&
                           key_bytes / 128 <= std::size_t(std::numeric_limits<int>::max());
    std::optional<CUtensorMap> map;
    for_each_indexer_window(input, plan, [&](IndexerArguments a) {
        const auto window = select_indexer_window(plan, a.width, a.batch,
                                                  execution.multiprocessor_count(), tma_cache);
        if (window.score == IndexerScore::Tma && !map)
            map.emplace(make_indexer_key_map(input, key_bytes));
        launch_window(a, window, vector_cache, map ? &*map : nullptr, execution);
    });
}
} // namespace ninfer::ops::detail::qsa
