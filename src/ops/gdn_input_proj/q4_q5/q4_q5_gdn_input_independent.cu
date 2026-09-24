#include "core/weight.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "core/pdl.cuh"
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

constexpr std::int32_t kQkRows     = 4096;
constexpr std::int32_t kValueRows  = 6144;
constexpr std::int32_t kZRows      = 6144;
constexpr std::int32_t kValueZRows = kValueRows + kZRows;
constexpr std::int32_t kHidden     = 5120;

using Q4GdnSimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4GdnSimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Q4GemvR1Q8DirectSchedule;
    const dim3 grid(static_cast<unsigned>(div_up(kQkRows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Schedule><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        nullptr, kQkRows, kHidden);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, bool Full>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t cols   = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(kQkRows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full><<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        nullptr, out_ld, 0, kQkRows, kHidden, cols, weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_q4_simt_route(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const bool full = (kQkRows % Schedule::kRowsPerCta) == 0 &&
                      ((kHidden / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    if (full) {
        launch_q4_simt<Schedule, true>(x, weight, out, stream);
    } else {
        launch_q4_simt<Schedule, false>(x, weight, out, stream);
    }
}

template <std::int32_t Capacity>
void launch_q4_ksplit_exact(const Tensor& x, const Weight& weight, Tensor& out,
                            cudaStream_t stream) {
    using Geometry = Q4LinearGeometry<kQkRows, kHidden>;
    constexpr std::int32_t kTileCols = (Capacity + 7) / 8 * 8;
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    // The store's live-column count is the problem's actual column count, not the tile capacity:
    // Capacity only sizes the tile, and the K-split kernel stages exactly the columns it is told are
    // live, so the masked tail can never be written.
    const Q4KSplitStridedStore<false, 0> store{static_cast<__nv_bfloat16*>(out.data), out_ld,
                                               nullptr, 0, x.ne[1]};
    q4_ksplit_mma_kernel<Geometry, kTileCols, Capacity, Q4KSplitStridedStore<false, 0>,
                         Q4KSplitIdentityRows, true>
        <<<kQkRows / Q4KSplitMmaSchedule::kRowsPerCta, Q4KSplitMmaSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(out.data), store, {}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q4_ksplit_band(const Tensor& x, const Weight& weight, Tensor& out,
                           cudaStream_t stream) {
    if (weight.padded_shape[1] != kHidden) {
        throw std::invalid_argument("Q4/Q5 GDN K-split requires padded K == hidden");
    }
    switch (x.ne[1]) {
    case 7:
        launch_q4_ksplit_exact<7>(x, weight, out, stream);
        return;
    case 8:
        launch_q4_ksplit_exact<8>(x, weight, out, stream);
        return;
    case 9:
        launch_q4_ksplit_exact<9>(x, weight, out, stream);
        return;
    case 10:
        launch_q4_ksplit_exact<10>(x, weight, out, stream);
        return;
    case 11:
        launch_q4_ksplit_exact<11>(x, weight, out, stream);
        return;
    case 12:
        launch_q4_ksplit_exact<12>(x, weight, out, stream);
        return;
    default:
        throw std::invalid_argument("Q4/Q5 GDN K-split band covers T in [7,12]");
    }
}

void launch_q4(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    switch (x.ne[1]) {
    case 1:
        launch_q4_gemv(x, weight, out, stream);
        return;
    case 2:
    case 3:
    case 4:
        launch_q4_simt_route<Q4GdnSimtR8C4Schedule>(x, weight, out, stream);
        return;
    case 7:
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
        // The K-split MMA arms all 8 warps of a CTA onto K instead of waiting out the weight stream
        // of a row. Complete-op measurement (both parents, one graph, one probe run per column count)
        // at T=9..12: the split form with this parent is 101.6-105.7 us against 120.1 us for the
        // grouped kernel R6 chose, while at T=13 the grouped kernel wins again (120.1 against 126.2),
        // so the band ends at 12.
        launch_q4_ksplit_band(x, weight, out, stream);
        return;
    default:
        if (x.ne[1] <= 15) {
            launch_q4_simt_route<Q4GdnSimtR8C8Schedule>(x, weight, out, stream);
            return;
        }
        throw std::invalid_argument("Q4/Q5 GDN independent launch requires T in [1,15]");
    }
}

void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                    cudaStream_t stream) {
    constexpr int kRowsPerBlock = 16;
    constexpr int kThreads      = kRowsPerBlock * 32;
    q5_rowsplit_gemv_kernel<kValueZRows, kHidden, kRowsPerBlock, 2, true, true, kValueRows>
        <<<kValueZRows / kRowsPerBlock, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const std::uint8_t*>(weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data));
    CUDA_CHECK(cudaGetLastError());
}

template <int Cols>
void launch_q5_split4(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                      cudaStream_t stream) {
    constexpr int kThreads    = 4 * 32;
    const std::int32_t out_ld = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(kValueZRows), 1u, 1u);
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Cols, 5, kHidden, true, kValueRows>
        <<<grid, kThreads, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                        static_cast<const std::uint8_t*>(weight.qdata),
                                        static_cast<const std::uint8_t*>(weight.qhigh),
                                        static_cast<const std::uint8_t*>(weight.scales),
                                        static_cast<__nv_bfloat16*>(value.data),
                                        static_cast<__nv_bfloat16*>(z.data), kValueZRows, out_ld,
                                        kHidden, Cols, weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q5_split4_exact(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                            cudaStream_t stream) {
    switch (x.ne[1]) {
    case 2:
        launch_q5_split4<2>(x, weight, value, z, stream);
        return;
    case 3:
        launch_q5_split4<3>(x, weight, value, z, stream);
        return;
    case 4:
        launch_q5_split4<4>(x, weight, value, z, stream);
        return;
    case 5:
        launch_q5_split4<5>(x, weight, value, z, stream);
        return;
    case 6:
        launch_q5_split4<6>(x, weight, value, z, stream);
        return;
    case 7:
        launch_q5_split4<7>(x, weight, value, z, stream);
        return;
    case 8:
        launch_q5_split4<8>(x, weight, value, z, stream);
        return;
    case 9:
        launch_q5_split4<9>(x, weight, value, z, stream);
        return;
    case 10:
        launch_q5_split4<10>(x, weight, value, z, stream);
        return;
    default:
        throw std::invalid_argument("GDN Q5 split4 requires T in [2,10]");
    }
}

template <int kColsPerTile>
void launch_q5_simt_cols(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
                         cudaStream_t stream) {
    constexpr int kRowsPerBlock = 8;
    constexpr int kStages       = 2;
    constexpr int kThreads      = kRowsPerBlock * 32;
    const std::int32_t cols     = x.ne[1];
    const std::int32_t out_ld   = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(kValueZRows, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(cols, kColsPerTile)), 1u);
    q5_rowsplit_gemm_simt_kernel<Q5RowSplitSimtSchedule, kColsPerTile, kRowsPerBlock, kStages, true,
                                 kValueRows><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(value.data),
        static_cast<__nv_bfloat16*>(z.data), kValueZRows, out_ld, kHidden, cols,
        weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
}

void launch_q5(const Tensor& x, const Weight& weight, Tensor& value, Tensor& z,
               cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q5_gemv(x, weight, value, z, stream);
        return;
    }
    if (x.ne[1] <= 10) {
        // Split4: one CTA owns one output row, its four warps split the K dimension and reduce
        // their partial sums through shared memory. The column count is a compile-time template
        // argument, so the kernel covers exactly the live columns. The fused projections below
        // (T=2..6) use the same shape, so the Q5 parent has one mechanism from 2 to 10; the band
        // end is the measured crossover against the c4 tile at 11..12, not a shape limit.
        launch_q5_split4_exact(x, weight, value, z, stream);
        return;
    }
    if (x.ne[1] <= 12) {
        // c4 SIMT: one output row per warp, up to four columns per column tile, with the quantized
        // weight planes staged in shared memory and activations read from the input tensor.
        // Retained in this interval on complete-Op measurements; both shapes are legal at every
        // count in [2,15].
        launch_q5_simt_cols<4>(x, weight, value, z, stream);
        return;
    }
    if (x.ne[1] <= 15) {
        launch_q5_simt_cols<8>(x, weight, value, z, stream);
        return;
    }
    throw std::invalid_argument("Q4/Q5 GDN independent launch requires T in [1,15]");
}

void launch_t4_pdl(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                   Tensor& qk, Tensor& value, Tensor& z, cudaStream_t stream) {
    using Q4Schedule         = Q4GdnSimtR8C4Schedule;
    constexpr int kQ5Threads = 4 * 32;
    const dim3 q4_grid(kQkRows / Q4Schedule::kRowsPerCta, 1u, 1u);
    const dim3 q5_grid(kValueZRows, 1u, 1u);
    const std::int32_t q4_out_ld = static_cast<std::int32_t>(qk.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t q5_out_ld = static_cast<std::int32_t>(value.nb[1] / sizeof(__nv_bfloat16));

    // Q5 and Q4 publish disjoint row ranges. Q4 can execute while Q5 drains and joins Q5 only at
    // exit, before the following convolution/snapshot kernel becomes runnable.
    q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, 4, 5, kHidden, true, kValueRows,
                                        Q5Split4StoreEpilogue, true, false>
        <<<q5_grid, kQ5Threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            kValueZRows, q5_out_ld, kHidden, 4, value_z_weight.padded_shape[1], 5);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(pdl::launch_dependent(
        {q4_grid, dim3(Q4Schedule::kThreads), 0, stream},
        q4_rowsplit_gemm_simt_kernel<Q4Schedule, true, false, 0, Q4SimtStoreEpilogue, false, true>,
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const std::uint8_t*>(qk_weight.qdata),
        static_cast<const std::uint8_t*>(qk_weight.scales), static_cast<__nv_bfloat16*>(qk.data),
        nullptr, q4_out_ld, 0, kQkRows, kHidden, 4, qk_weight.padded_shape[1],
        Q4SimtStoreEpilogue{}));
}

} // namespace

void q4_q5_gdn_input_independent_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& value_z_weight, Tensor& qk, Tensor& value,
                                        Tensor& z, cudaStream_t stream) {
    if (x.ne[1] == 4) {
        launch_t4_pdl(x, qk_weight, value_z_weight, qk, value, z, stream);
        return;
    }
    launch_q4(x, qk_weight, qk, stream);
    launch_q5(x, value_z_weight, value, z, stream);
}

} // namespace ninfer::ops::detail
