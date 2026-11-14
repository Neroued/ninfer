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
    for (int slot = blockIdx.x; slot < a.width * a.batch; slot += gridDim.x) {
        const int end = a.end(slot), count = end / 4;
        const auto column = a.column(slot);
        if (threadIdx.x == 0) {
            a.ends[column]   = end;
            a.counts[column] = count;
        }
        for (int i = threadIdx.x; i < count; i += blockDim.x)
            a.ids[column * kIndexSelected + i] = i;
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
    if (a.columns <= kIndexSelected) {
        select_all(a, s.score_workers, stream);
        return;
    }
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
    if (a.index.columns <= kIndexSelected) {
        select_all(a.index, s.score_workers, stream);
        return;
    }
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

int launch_fp8_windows(Fp8IndexerArguments f, int begin, int count, int windows, int sms,
                       cudaStream_t stream) {
    for (int slot = 0; slot < windows; ++slot) {
        const auto window = indexer_window(count, windows, slot, 8);
        f.index.begin     = begin + window.begin;
        f.index.width     = window.width;
        launch_fp8_window(f, select_fp8_indexer(f.index, sms), stream);
    }
    return windows;
}

int launch_windows(IndexerArguments a, int windows, int sms, cudaStream_t stream) {
    for (int slot = 0; slot < windows; ++slot) {
        const auto window = indexer_window(a.tokens, windows, slot);
        a.begin           = window.begin;
        a.width           = window.width;
        launch_indexer_window(a, select_indexer(a, sms), stream);
    }
    return windows;
}

void launch_request_group(IndexerArguments a, int windows, int sms, cudaStream_t stream) {
    const int slots = indexer_graph_slots(a.tokens, a.batch);
    int submitted   = 0;
    if (separate_indexer_requests(a)) {
        // Request windows share the group's score/candidate storage in stream order.
        // Tensor and selection strides stay T.
        const int rows            = indexer_window_width(a.tokens, windows) * a.batch;
        const int request_windows = a.tokens / rows + (a.tokens % rows != 0);
        auto request              = a;
        request.batch             = 1;
        for (int b = 0; b < a.batch; ++b) {
            submitted += launch_windows(request, request_windows, sms, stream);
            request.query += std::int64_t(a.tokens) * 4096;
            request.coefficients += std::int64_t(a.tokens) * 32;
            request.positions += a.tokens;
            ++request.valid;
            ++request.table_rows;
            request.ids += std::int64_t(a.tokens) * kIndexSelected;
            request.counts += a.tokens;
            request.ends += a.tokens;
        }
    } else {
        submitted = launch_windows(a, windows, sms, stream);
    }
    // Four nodes per fixed Graph slot, including when envelope growth changes
    // request grouping.
    for (int slot = submitted; slot < slots; ++slot)
        for (int stage = 0; stage < 4; ++stage) indexer_idle_kernel<<<1, 1, 0, stream>>>();
    CUDA_CHECK(cudaGetLastError());
}

void launch_fp8_request_group(Fp8IndexerArguments f, int sms, cudaStream_t stream) {
    for (int begin = 0; begin < f.index.tokens;) {
        const int width      = std::min(kIndexPreparedQueries, f.index.tokens - begin);
        f.query_begin        = begin;
        f.query_width        = width;
        const int query_jobs = f.index.columns > kIndexSelected ? width * f.index.batch * 8 : 1;
        if (f.index.columns > kIndexSelected)
            indexer_query_fp8_kernel<<<query_jobs, 128, 0, stream>>>(f);
        else
            indexer_idle_kernel<<<1, 1, 0, stream>>>();
        const auto windows = fp8_indexer_windows(width, f.index.batch, f.index.columns);
        int submitted      = 0;
        auto request       = f;
        for (int b = 0; b < f.index.batch; b += windows.group_size) {
            const int take      = std::min(windows.group_size, f.index.batch - b);
            request.index.batch = take;
            submitted +=
                launch_fp8_windows(request, begin, width, windows.count(width, take), sms, stream);
            if (b + take == f.index.batch) break;
            // Scratch is reused only after the group's final merge. Prepared Q
            // survives all groups; its stride is the outer window, not tokens.
            request.query += std::int64_t(take) * width * 4096;
            request.coefficients += std::int64_t(take) * width * 32;
            auto& a = request.index;
            a.query += std::int64_t(take) * a.tokens * 4096;
            a.coefficients += std::int64_t(take) * a.tokens * 32;
            a.positions += take * a.tokens;
            a.valid += take;
            a.table_rows += take;
            a.ids += std::int64_t(take) * a.tokens * kIndexSelected;
            a.counts += take * a.tokens;
            a.ends += take * a.tokens;
        }
        // Q preparation has one node per fixed outer window. Only the four score /
        // selection stages occupy the envelope-independent inner Graph slots.
        const int slots = fp8_indexer_graph_slots(width, f.index.batch);
        for (int slot = submitted; slot < slots; ++slot)
            for (int stage = 0; stage < 4; ++stage) indexer_idle_kernel<<<1, 1, 0, stream>>>();
        begin += width;
    }
    CUDA_CHECK(cudaGetLastError());
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
} // namespace

void launch_indexer(IndexerArguments a, int sms, cudaStream_t stream) {
    for (int begin = 0; begin < a.batch;) {
        auto group         = request_group(a, begin, indexer_group_size(a.tokens, a.batch - begin));
        const auto windows = indexer_windows(a.tokens, a.tokens, group.batch, a.columns);
        launch_request_group(group, windows.count, sms, stream);
        begin += group.batch;
    }
}

void launch_fp8_indexer(Fp8IndexerArguments f, int sms, cudaStream_t stream) {
    for (int begin = 0; begin < f.index.batch;) {
        auto group  = f;
        group.index = request_group(f.index, begin,
                                    indexer_group_size(f.index.tokens, f.index.batch - begin));
        // Prepared Q and score scratch are reused only after this group's final selection.
        launch_fp8_request_group(group, sms, stream);
        begin += group.index.batch;
    }
}
} // namespace ninfer::ops::detail::kpool
