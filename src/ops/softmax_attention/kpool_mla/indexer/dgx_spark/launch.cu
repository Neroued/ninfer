#include "ops/softmax_attention/kpool_mla/indexer/plan.h"
#include "ops/softmax_attention/kpool_mla/indexer/score_bf16.cuh"
#include "ops/softmax_attention/kpool_mla/indexer/score_fp8.cuh"
#include "ops/softmax_attention/kpool_mla/indexer/query_fp8.cuh"
#include "ops/softmax_attention/kpool_mla/indexer/select.cuh"
#include "ops/common/host/launch.h"

namespace ninfer::ops::detail::kpool {
namespace {
__global__ void indexer_idle_kernel() {}

__global__ void indexer_all_kernel(const __grid_constant__ IndexerArguments a) {
    for (std::int64_t slot = blockIdx.x; slot < a.width * a.batch; slot += gridDim.x) {
        const int end = a.end(int(slot)), count = end / 4;
        const auto column = a.column(int(slot));
        if (threadIdx.x == 0) {
            a.ends[column]   = end;
            a.counts[column] = count;
        }
        auto* ids = a.ids + column * kIndexSelected;
        if (reinterpret_cast<std::uintptr_t>(ids) % alignof(int4) == 0) {
            auto* vectors = reinterpret_cast<int4*>(ids);
            for (int v = threadIdx.x; v < count / 4; v += blockDim.x) {
                const int i = v * 4;
                vectors[v]  = make_int4(i, i + 1, i + 2, i + 3);
            }
            const int i = count / 4 * 4 + threadIdx.x;
            if (i < count) ids[i] = i;
        } else {
            for (int i = threadIdx.x; i < count; i += blockDim.x) ids[i] = i;
        }
    }
}

void select_all(IndexerArguments a, int workers, cudaStream_t stream) {
    indexer_all_kernel<<<std::min(a.width * a.batch, workers), 128, 0, stream>>>(a);
    for (int i = 0; i < 3; ++i) indexer_idle_kernel<<<1, 1, 0, stream>>>();
}

template <int Q, int G, int Stages>
void score(IndexerArguments a, IndexerSchedule s, cudaStream_t stream) {
    constexpr auto kernel = indexer_score_kernel<Q, G, Stages>;
    const int shared      = common::prepare_shared<Stages * G * 128 * 2, kernel, true>();
    kernel<<<s.score_workers, (Q < 4 ? 4 : Q) * 32, shared, stream>>>(a);
}

void launch_selection(IndexerArguments a, int workers, bool resident, cudaStream_t stream) {
    const auto partition = [&]<int Items>() {
        if (resident)
            indexer_partition_words_kernel<4, Items><<<workers, 256, 0, stream>>>(a);
        else
            indexer_partition_words_kernel<1, Items><<<workers, 256, 0, stream>>>(a);
    };
    switch (indexer_selection_items(a.columns)) {
    case 4:
        partition.template operator()<4>();
        break;
    case 8:
        partition.template operator()<8>();
        break;
    default:
        partition.template operator()<16>();
        break;
    }
    if (a.partitions > kIndexFanIn)
        indexer_merge_kernel<false><<<a.width * a.batch * a.merge_groups, 256, 0, stream>>>(a);
    else
        indexer_idle_kernel<<<1, 1, 0, stream>>>();
    if (a.partitions > 1)
        indexer_merge_kernel<true><<<a.width * a.batch, 256, 0, stream>>>(a);
    else
        indexer_idle_kernel<<<1, 1, 0, stream>>>();
}

void launch_indexer_window(IndexerArguments a, IndexerSchedule s, cudaStream_t stream) {
    const auto submit = [&]<int Q, int G>() {
        if (s.stages == 2)
            score<Q, G, 2>(a, s, stream);
        else
            score<Q, G, 1>(a, s, stream);
    };
    switch (s.queries) {
    case 1:
        submit.template operator()<1, 64>();
        break;
    case 2:
        submit.template operator()<2, 64>();
        break;
    default:
        submit.template operator()<4, 32>();
        break;
    }
    launch_selection(a, s.select_workers, s.resident_selection, stream);
}

template <Fp8IndexerRoute Route>
void score_fp8(Fp8IndexerArguments a, Fp8IndexerSchedule s, cudaStream_t stream) {
    constexpr auto c      = fp8_indexer_config(Route);
    constexpr auto kernel = indexer_score_fp8_kernel<c.queries, c.keys, c.stages, c.blocks,
                                                     c.warp_queries, c.epilogue_prefetch>;
    constexpr int warps   = c.queries / c.warp_queries;
    const int shared      = common::prepare_shared<c.stages * c.keys * 129, kernel, true>();
    kernel<<<s.score_workers, (warps < 4 ? 4 : warps) * 32, shared, stream>>>(a);
}

void launch_fp8_window(Fp8IndexerArguments a, Fp8IndexerSchedule s, cudaStream_t stream) {
    switch (s.route) {
    case Fp8IndexerRoute::Q1K64S2:
        score_fp8<Fp8IndexerRoute::Q1K64S2>(a, s, stream);
        break;
    case Fp8IndexerRoute::Q2K64S2:
        score_fp8<Fp8IndexerRoute::Q2K64S2>(a, s, stream);
        break;
    case Fp8IndexerRoute::Q4K32S1:
        score_fp8<Fp8IndexerRoute::Q4K32S1>(a, s, stream);
        break;
    case Fp8IndexerRoute::Q4K32S2:
        score_fp8<Fp8IndexerRoute::Q4K32S2>(a, s, stream);
        break;
    case Fp8IndexerRoute::Q4K64S3:
        score_fp8<Fp8IndexerRoute::Q4K64S3>(a, s, stream);
        break;
    case Fp8IndexerRoute::Q8K16Prefetch:
        score_fp8<Fp8IndexerRoute::Q8K16Prefetch>(a, s, stream);
        break;
    }
    launch_selection(a.index, s.select_workers, s.resident_selection, stream);
}

void pad_windows(int submitted, int slots, cudaStream_t stream) {
    for (int slot = submitted; slot < slots; ++slot)
        for (int stage = 0; stage < 4; ++stage) indexer_idle_kernel<<<1, 1, 0, stream>>>();
}

IndexerArguments request_group(IndexerArguments a, int begin, int count) {
    const auto column = std::int64_t(begin) * a.tokens;
    a.query += column * 4096;
    a.coefficients += column * 32;
    a.positions += column;
    a.valid += begin;
    a.table_rows += begin;
    a.ids += column * kIndexSelected;
    a.counts += column;
    a.ends += column;
    a.batch = count;
    return a;
}

void launch_request_group(IndexerArguments group, int sms, cudaStream_t stream) {
    const int slots    = indexer_graph_slots(group.tokens, group.tokens, group.batch);
    const auto windows = indexer_windows(group.tokens, group.batch, group.columns);
    int submitted      = 0;
    for (int b = 0; b < group.batch;) {
        const int take  = std::min(windows.group_size, group.batch - b);
        auto a          = request_group(group, b, take);
        const int count = windows.count(a.tokens, take);
        for (int slot = 0; slot < count; ++slot) {
            const auto window = indexer_window(a.tokens, count, slot, windows.tile);
            a.begin           = window.begin;
            a.width           = window.width;
            if (a.columns <= kIndexSelected)
                select_all(a, 4 * sms, stream);
            else
                launch_indexer_window(a, select_indexer(a, sms), stream);
        }
        submitted += count;
        b += take;
    }
    pad_windows(submitted, slots, stream);
}

void launch_fp8_request_group(Fp8IndexerArguments group, int prepared_batch, int sms,
                              cudaStream_t stream) {
    const auto windows =
        indexer_windows(group.index.tokens, group.index.batch, group.index.columns, prepared_batch);
    const bool all = group.index.columns <= kIndexSelected;
    for (int begin = 0; begin < group.index.tokens;) {
        const int width   = std::min(kIndexPreparedQueries, group.index.tokens - begin);
        group.query_begin = begin;
        group.query_width = width;
        if (all)
            indexer_idle_kernel<<<1, 1, 0, stream>>>();
        else
            indexer_query_fp8_kernel<<<width * group.index.batch * 8, 128, 0, stream>>>(group);

        int submitted = 0;
        for (int b = 0; b < group.index.batch;) {
            const int take = std::min(windows.group_size, group.index.batch - b);
            auto f         = group;
            f.index        = request_group(group.index, b, take);
            if (!all) {
                // Prepared Q is shared across all score windows in the chunk.
                f.query += std::int64_t(b) * width * 4096;
                f.coefficients += b * width * 32;
            }
            const int count = windows.count(width, take);
            for (int slot = 0; slot < count; ++slot) {
                const auto window = indexer_window(width, count, slot, windows.tile);
                f.index.begin     = begin + window.begin;
                f.index.width     = window.width;
                if (all)
                    select_all(f.index, 4 * sms, stream);
                else
                    launch_fp8_window(f, select_fp8_indexer(f.index, sms), stream);
            }
            submitted += count;
            b += take;
        }
        pad_windows(
            submitted,
            indexer_graph_slots(group.index.tokens, width, group.index.batch, prepared_batch),
            stream);
        begin += width;
    }
}
} // namespace

void launch_indexer(IndexerArguments a, int sms, cudaStream_t stream) {
    for (int begin = 0; begin < a.batch;) {
        const int count = indexer_group_size(a.tokens, a.batch - begin);
        launch_request_group(request_group(a, begin, count), sms, stream);
        begin += count;
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_fp8_indexer(Fp8IndexerArguments f, int sms, cudaStream_t stream) {
    const int prepared_batch = indexer_group_size(f.index.tokens, f.index.batch);
    for (int begin = 0; begin < f.index.batch;) {
        const int count = std::min(prepared_batch, f.index.batch - begin);
        auto group      = f;
        group.index     = request_group(f.index, begin, count);
        launch_fp8_request_group(group, prepared_batch, sms, stream);
        begin += count;
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_indexer_segment(IndexerArguments a, int sms, cudaStream_t stream) {
    launch_indexer_window(a, select_indexer(a, sms), stream);
}

void launch_fp8_indexer_segment(Fp8IndexerArguments f, int sms, cudaStream_t stream) {
    const auto& a = f.index;
    f.query_begin = a.begin;
    f.query_width = a.width;
    indexer_query_fp8_kernel<<<a.width * 8, 128, 0, stream>>>(f);
    launch_fp8_window(f, select_fp8_indexer(a, sms), stream);
}
} // namespace ninfer::ops::detail::kpool
