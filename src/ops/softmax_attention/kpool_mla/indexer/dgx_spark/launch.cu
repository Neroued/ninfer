#include "ops/softmax_attention/kpool_mla/indexer/plan.h"
#include "ops/softmax_attention/kpool_mla/indexer/score.cuh"
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
    indexer_partition_kernel<<<s.select_workers, 256, 0, stream>>>(a);
    indexer_merge_kernel<false>
        <<<std::max(1, a.width * a.batch * a.merge_groups), 256, 0, stream>>>(a);
    indexer_merge_kernel<true><<<a.width * a.batch, 256, 0, stream>>>(a);
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
} // namespace ninfer::ops::detail::kpool
