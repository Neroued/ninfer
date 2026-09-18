#include "core/weight.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_ksplit_mma.cuh"
#include "ops/linear/q4/q4_ksplit_strided_store.cuh"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kParentRows = 7168;
constexpr std::int32_t kSplitRow   = 6144;
constexpr std::int32_t kHidden     = 5120;

using Q4AttnSimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4AttnSimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                    cudaStream_t stream) {
    using Schedule = Q4GemvR1Q8DirectSchedule;
    const dim3 grid(static_cast<unsigned>(div_up(kParentRows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Schedule, true, kSplitRow><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(key.data), kParentRows, kHidden);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, bool Full>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                    cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(kParentRows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full, true, kSplitRow>
        <<<grid, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
            static_cast<__nv_bfloat16*>(key.data), q.ne[0], key.ne[0], kParentRows, kHidden, cols,
            weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_q4_simt_route(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                          cudaStream_t stream) {
    const bool full = (kParentRows % Schedule::kRowsPerCta) == 0 &&
                      ((kHidden / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    if (full) {
        launch_q4_simt<Schedule, true>(x, weight, q, key, stream);
    } else {
        launch_q4_simt<Schedule, false>(x, weight, q, key, stream);
    }
}

template <std::int32_t Capacity>
void launch_q4_ksplit_exact(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                            cudaStream_t stream) {
    using Geometry = Q4LinearGeometry<kParentRows, kHidden>;
    constexpr std::int32_t kTileCols = (Capacity + 7) / 8 * 8;
    using Store = Q4KSplitStridedStore<true, kSplitRow>;
    // The store's live-column count is the problem's actual column count: Capacity only sizes the
    // tile, and the K-split kernel stages exactly the columns it is told are live.
    const Store store{static_cast<__nv_bfloat16*>(q.data), q.ne[0],
                      static_cast<__nv_bfloat16*>(key.data), key.ne[0], x.ne[1]};
    q4_ksplit_mma_kernel<Geometry, kTileCols, Capacity, Store, Q4KSplitIdentityRows, true>
        <<<kParentRows / Q4KSplitMmaSchedule::kRowsPerCta, Q4KSplitMmaSchedule::kThreads, 0,
           stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                     static_cast<const std::uint8_t*>(weight.qdata),
                     static_cast<const std::uint8_t*>(weight.scales),
                     static_cast<__nv_bfloat16*>(q.data), store, {}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q4_ksplit_band(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key,
                           cudaStream_t stream) {
    if (weight.padded_shape[1] != kHidden) {
        throw std::invalid_argument("attention Q4 K-split requires padded K == hidden");
    }
    switch (x.ne[1]) {
    case 7:
        launch_q4_ksplit_exact<7>(x, weight, q, key, stream);
        return;
    case 8:
        launch_q4_ksplit_exact<8>(x, weight, q, key, stream);
        return;
    case 9:
        launch_q4_ksplit_exact<9>(x, weight, q, key, stream);
        return;
    case 10:
        launch_q4_ksplit_exact<10>(x, weight, q, key, stream);
        return;
    case 11:
        launch_q4_ksplit_exact<11>(x, weight, q, key, stream);
        return;
    case 12:
        launch_q4_ksplit_exact<12>(x, weight, q, key, stream);
        return;
    default:
        throw std::invalid_argument("attention Q4 K-split band covers T in [7,12]");
    }
}

void launch_q4(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key, cudaStream_t stream) {
    switch (x.ne[1]) {
    case 1:
        launch_q4_gemv(x, weight, q, key, stream);
        return;
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
        // K-split for the Q4 parent across the whole parent-split range. Complete-op measurement
        // (both parents launched, all four outputs, one graph, one probe run per column count):
        // 73.0-77.6 us at T=9..12 against 100.1-105.7 us for the row-split SIMT that R0 used there,
        // and 107.8-108.3 us for the grouped form the resolver switches to at T=13. The resolver
        // boundary at 13 is right for the grouped-vs-row-split question, but it hid this: the split
        // form with a K-split Q4 parent is 24-29% faster than both. T=2..6 keep the SIMT tile, where
        // a 16-wide K-split tile would waste more MMA work than it saves.
        launch_q4_ksplit_band(x, weight, q, key, stream);
        return;
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
        launch_q4_simt_route<Q4AttnSimtR8C4Schedule>(x, weight, q, key, stream);
        return;
    default:
        throw std::invalid_argument("attention Q4 split-output requires T in [1,12]");
    }
}

void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    constexpr int kRowsPerBlock = 16;
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    constexpr int kGrid         = kParentRows / kRowsPerBlock;
    q5_rowsplit_gemv_kernel<kParentRows, kHidden, kRowsPerBlock, 2, true, true, kSplitRow>
        <<<kGrid, kBlockThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                              static_cast<const std::uint8_t*>(weight.qdata),
                                              static_cast<const std::uint8_t*>(weight.qhigh),
                                              static_cast<const std::uint8_t*>(weight.scales),
                                              static_cast<__nv_bfloat16*>(gate.data),
                                              static_cast<__nv_bfloat16*>(value.data));
    CUDA_CHECK(cudaGetLastError());
}

template <int Cols>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                      cudaStream_t stream) {
    constexpr int kThreads = 4 * 32;
    const dim3 grid(static_cast<unsigned>(kParentRows), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, 5, kHidden, true, kSplitRow>
        <<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                        static_cast<const std::uint8_t*>(weight.qdata),
                                        static_cast<const std::uint8_t*>(weight.qhigh),
                                        static_cast<const std::uint8_t*>(weight.scales),
                                        static_cast<__nv_bfloat16*>(gate.data),
                                        static_cast<__nv_bfloat16*>(value.data), kParentRows,
                                        gate.ne[0], kHidden, Cols, weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q5_split4_exact(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                            cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2:
        launch_q5_split4<2>(x, weight, gate, value, stream);
        return;
    case 3:
        launch_q5_split4<3>(x, weight, gate, value, stream);
        return;
    case 4:
        launch_q5_split4<4>(x, weight, gate, value, stream);
        return;
    case 5:
        launch_q5_split4<5>(x, weight, gate, value, stream);
        return;
    case 6:
        launch_q5_split4<6>(x, weight, gate, value, stream);
        return;
    case 7:
        launch_q5_split4<7>(x, weight, gate, value, stream);
        return;
    case 8:
        launch_q5_split4<8>(x, weight, gate, value, stream);
        return;
    case 9:
        launch_q5_split4<9>(x, weight, gate, value, stream);
        return;
    default:
        throw std::invalid_argument("attention Q5 split4 requires T in [2,9]");
    }
}

template <int ColsPerTile>
void launch_q5_simt(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    constexpr int kRowsPerBlock = 8;
    constexpr int kStages       = 2;
    constexpr int kThreads      = kRowsPerBlock * 32;
    const std::int32_t cols     = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(kParentRows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, ColsPerTile)), 1u);
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, ColsPerTile, kRowsPerBlock, kStages, true,
                                 kSplitRow><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(value.data), kParentRows, gate.ne[0], kHidden, cols,
        weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q5(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q5_gemv(x, weight, gate, value, stream);
        return;
    }
    if (x.ne[1] <= 9) {
        // Split4: one CTA owns one output row, its four warps split the K dimension and reduce
        // their partial sums through shared memory, with the column count as a compile-time
        // template argument so the kernel covers exactly the live columns. The fused projections
        // use the same shape up to 6, so the Q5 parent has one mechanism from 2 to 9.
        launch_q5_split4_exact(x, weight, gate, value, stream);
        return;
    }
    if (x.ne[1] <= 12) {
        // c4 SIMT: one output row per warp, up to four columns per column tile, with the quantized
        // weight planes staged in shared memory and activations read from the input tensor.
        // Retained in this interval on complete-Op measurements; both shapes are legal at every
        // count in [2,12].
        launch_q5_simt<4>(x, weight, gate, value, stream);
        return;
    }
    throw std::invalid_argument("attention Q5 split-output requires T in [1,12]");
}

} // namespace

void q4_q5_attn_input_small_t_launch(const Tensor& x, const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_q4(x, query_key_weight, q, k, stream);
    launch_q5(x, gate_value_weight, gate, v, stream);
}

} // namespace ninfer::ops::detail
