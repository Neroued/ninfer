#include "ops/softmax_attention/kpool_mla/indexer/plan.h"
#include "ops/softmax_attention/kpool_mla/indexer/score_bf16.cuh"
#include "ops/softmax_attention/kpool_mla/indexer/score_fp8.cuh"
#include "ops/softmax_attention/kpool_mla/indexer/query_fp8.cuh"
#include "ops/softmax_attention/kpool_mla/indexer/select.cuh"
#include "ops/common/host/launch.h"

namespace ninfer::ops::detail::kpool {
namespace {
__global__ void indexer_idle_kernel() {}

template <int Q, int G, int Stages>
void score(IndexerArguments a, IndexerSchedule s, cudaStream_t stream) {
    constexpr auto kernel = indexer_score_kernel<Q, G, Stages>;
    const int shared      = common::prepare_shared<Stages * G * 128 * 2, kernel, true>();
    kernel<<<s.score_workers, (Q < 4 ? 4 : Q) * 32, shared, stream>>>(a);
}

void launch_selection(IndexerArguments a, int workers, cudaStream_t stream) {
    indexer_partition_kernel<<<workers, 256, 0, stream>>>(a);
    indexer_merge_kernel<false>
        <<<std::max(1, a.width * a.batch * a.merge_groups), 256, 0, stream>>>(a);
    indexer_merge_kernel<true><<<a.width * a.batch, 256, 0, stream>>>(a);
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
    launch_selection(a, s.select_workers, stream);
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
    if (s.resident_selection)
        indexer_partition_fp8_kernel<4><<<s.select_workers, 256, 0, stream>>>(a.index);
    else
        indexer_partition_fp8_kernel<1><<<s.select_workers, 256, 0, stream>>>(a.index);
    indexer_merge_kernel<false, 8, 2, true>
        <<<std::max(1, a.index.width * a.index.batch * a.index.merge_groups), 256, 0, stream>>>(
            a.index);
    indexer_merge_kernel<true, 8, 2, true>
        <<<a.index.width * a.index.batch, 256, 0, stream>>>(a.index);
}

int launch_fp8_windows(Fp8IndexerArguments f, int begin, int count, int windows, int sms,
                       cudaStream_t stream) {
    const int width = count / windows, larger = count % windows;
    for (int slot = 0; slot < windows; ++slot) {
        f.index.begin = begin + slot * width + std::min(slot, larger);
        f.index.width = width + (slot < larger);
        launch_fp8_window(f, select_fp8_indexer(f.index, sms), stream);
    }
    return windows;
}

int launch_windows(IndexerArguments a, int windows, int sms, cudaStream_t stream) {
    const int width = a.tokens / windows, larger = a.tokens % windows;
    for (int slot = 0; slot < windows; ++slot) {
        a.begin = slot * width + std::min(slot, larger);
        a.width = width + (slot < larger);
        launch_indexer_window(a, select_indexer(a, sms), stream);
    }
    return windows;
}
} // namespace

void launch_indexer(IndexerArguments a, int windows, int sms, cudaStream_t stream) {
    const int stride = kIndexGraphRows / a.batch;
    const auto slots = (std::int64_t(a.tokens) + stride - 1) / stride;
    int submitted    = 0;
    if (separate_indexer_requests(a)) {
        // Reuse the caller's existing score/candidate storage. Each request window
        // fits the original maximum row count; its final merge finishes before
        // the next request overwrites scratch. Tensor and selection strides stay T.
        const int rows            = (a.tokens / windows + (a.tokens % windows != 0)) * a.batch;
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
    // request grouping. The grouped route has T>=64, B>=3 and at most
    // B*ceil(windows/B) active slots, which fits this original slot budget.
    for (int slot = submitted; slot < slots; ++slot)
        for (int stage = 0; stage < 4; ++stage) indexer_idle_kernel<<<1, 1, 0, stream>>>();
    CUDA_CHECK(cudaGetLastError());
}

void launch_fp8_indexer(Fp8IndexerArguments f, int sms, cudaStream_t stream) {
    const int stride = kIndexGraphRows / f.index.batch;
    for (int begin = 0; begin < f.index.tokens;) {
        const int width      = std::min(kIndexPreparedQueries, f.index.tokens - begin);
        f.query_begin        = begin;
        f.query_width        = width;
        const int query_jobs = f.index.columns > kIndexSelected ? width * f.index.batch * 8 : 1;
        indexer_query_fp8_kernel<<<query_jobs, 128, 0, stream>>>(f);
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
        const int slots = (width + stride - 1) / stride;
        for (int slot = submitted; slot < slots; ++slot)
            for (int stage = 0; stage < 4; ++stage) indexer_idle_kernel<<<1, 1, 0, stream>>>();
        begin += width;
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::kpool
