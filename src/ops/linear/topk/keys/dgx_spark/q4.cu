#include "ops/linear/topk/keys/epilogue.cuh"
#include "ops/linear/kernels/q4/q4_sliced_k_launch.cuh"
#include "ops/linear/kernels/q4/q4_a16_tma_prefetch.cuh"

namespace ninfer::ops::detail::topk_keys {
namespace {
using Small = Q4A16SlicedKMmaSchedule<16, 16, 8, 2, Cache::cg, Cache::ca, 2, 4096, 16>;

template <class S, int Panel = 64, bool Sliced = false>
void run(const Tensor& x, const Weight& w, int begin, int domain, const Workspace& ws,
         cudaStream_t stream) {
    if constexpr (Sliced)
        launch_q4_a16_sliced_k_mma<S>(q4_linear_operands(x, w),
                                      output(ws.candidates, begin, domain),
                                      LinearIdentityEpilogue{}, stream);
    else
        launch_q4_a16_tma_prefetch<S>(q4_linear_operands(x, w),
                                      output(ws.candidates, begin, domain),
                                      TopKPanelEpilogue<Panel>{}, stream);
}

using Mid16              = Q4A16TmaPrefetchSchedule<64, 16, 16, 8, 2, 4096, 1, 16, 4, 1, 128, 2>;
using Mid32              = Q4A16TmaPrefetchSchedule<64, 32, 32, 8, 2, 4096, 1, 16, 4, 1, 128, 2>;
using Mid64              = Q4A16TmaPrefetchSchedule<128, 64, 128, 8, 2, 4096, 1, 16, 4, 1, 64, 1>;
using Mid96              = Q4A16TmaPrefetchSchedule<64, 96, 32, 24, 2, 4096, 1, 16, 4, 1, 128, 1>;
using Large              = Q4A16TmaPrefetchSchedule<64, 128, 64, 16, 2, 4096, 1>;
constexpr int kScoreRows = 256;

struct ScoreOutput {
    float* data;
    int columns;

    __device__ void store(int row, int column, float score) const {
        data[(std::int64_t(row / kScoreRows) * columns + column) * kScoreRows + row % kScoreRows] =
            score;
    }
};

using Bulk = Q4A16TmaPrefetchSchedule<64, 256, 64, 32, 2, 4096, 1, 16, 4, 1, 64, 1>;

__global__ __launch_bounds__(256) void select_scores(const float* scores, Output output, int rows,
                                                     int columns) {
    const int group = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int column = blockIdx.y * 8 + warp;
    if (column >= columns) return;
    const int row = group * kScoreRows + lane * 8;
    std::uint64_t keys[8]{};
    // The Q4 GEMM consumes complete 64-row tiles, including the final vocabulary window.
    if (row < rows) {
        const auto* source =
            scores + (std::int64_t(group) * columns + column) * kScoreRows + lane * 8;
        const float4 first  = *reinterpret_cast<const float4*>(source);
        const float4 second = *reinterpret_cast<const float4*>(source + 4);
        const float values[]{first.x,  first.y,  first.z,  first.w,
                             second.x, second.y, second.z, second.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) keys[i] = output.key(values[i], row + i, rows);
    }
    const auto chosen = topk_warp_rank(keys);
    if (lane < 16) output.publish(group, column, lane, chosen);
}

void materialized(const Tensor& x, const Weight& w, int begin, int domain, const Workspace& ws,
                  cudaStream_t stream) {
    launch_q4_a16_tma_prefetch<Bulk>(q4_linear_operands(x, w),
                                     ScoreOutput{static_cast<float*>(ws.scores.data), x.ne[1]},
                                     LinearIdentityEpilogue{}, stream);
    select_scores<<<dim3(ws.candidates.producer_groups, common::div_up(x.ne[1], 8)), 256, 0,
                    stream>>>(static_cast<const float*>(ws.scores.data),
                              output(ws.candidates, begin, domain), w.n, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

Plan select_q4(int columns) {
    if (columns > 256) return {materialized, kScoreRows, false, nullptr, 8192, true};
    if (columns <= 8) return {run<Small, 16, true>, Small::kBlockRows};
    if (columns <= 16) return {run<Mid16, 16>, Mid16::kBlockRows};
    if (columns <= 32) return {run<Mid32>, Mid32::kBlockRows};
    if (columns <= 64) return {run<Mid64, 16>, Mid64::kBlockRows};
    if (columns <= 96) return {run<Mid96, 128>, Mid96::kBlockRows};
    return {run<Large, 128>, Large::kBlockRows};
}
} // namespace ninfer::ops::detail::topk_keys
