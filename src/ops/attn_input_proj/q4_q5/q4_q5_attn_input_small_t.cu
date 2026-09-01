#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
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
    using Schedule = Q4GemvR1W8DirectSchedule;
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

void launch_q4(const Tensor& x, const Weight& weight, Tensor& q, Tensor& key, cudaStream_t stream) {
    switch (x.ne[1]) {
    case 1:
        launch_q4_gemv(x, weight, q, key, stream);
        return;
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13:
    case 14:
    case 15:
        launch_q4_simt_route<Q4AttnSimtR8C4Schedule>(x, weight, q, key, stream);
        return;
    case 8:
    case 16:
        launch_q4_simt_route<Q4AttnSimtR8C8Schedule>(x, weight, q, key, stream);
        return;
    default:
#ifdef NINFER_VOLTA_BUILD
        // GroupedHomogeneousPairMma* (T>16) need Ampere+ mma/ldmatrix, trap-stubbed on sm_70.
        // launch_q4_simt_route already takes cols as a runtime grid parameter (the switch above
        // is only picking a tile-size schedule for tuning, not a kernel limit), so it
        // generalizes to any T unchanged. See docs/v100.md.
        if (x.ne[1] > 16) {
            launch_q4_simt_route<Q4AttnSimtR8C8Schedule>(x, weight, q, key, stream);
            return;
        }
#endif
        throw std::invalid_argument("attention Q4 split-output requires T in [1,16]");
    }
}

void launch_q5_gemv(const Tensor& x, const Weight& weight, Tensor& gate, Tensor& value,
                    cudaStream_t stream) {
    constexpr int kRowsPerBlock = 16;
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    constexpr int kGrid         = kParentRows / kRowsPerBlock;
    q5_rowsplit_gemv_kernel<kParentRows, kHidden, kRowsPerBlock, 2, true, false, true, kSplitRow>
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
    default:
        throw std::invalid_argument("attention Q5 split4 requires T in [2,6]");
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
    if (x.ne[1] <= 6) {
        launch_q5_split4_exact(x, weight, gate, value, stream);
        return;
    }
    if (x.ne[1] <= 16) {
        launch_q5_simt<4>(x, weight, gate, value, stream);
        return;
    }
#ifdef NINFER_VOLTA_BUILD
    // Same reasoning as launch_q4 above: launch_q5_simt<4> already takes cols as a runtime grid
    // parameter, so it generalizes past T=16 once GroupedHomogeneousPairMma* is unavailable.
    launch_q5_simt<4>(x, weight, gate, value, stream);
    return;
#else
    throw std::invalid_argument("attention Q5 split-output requires T in [1,16]");
#endif
}

} // namespace

void q4_q5_attn_input_small_t_launch(const Tensor& x, const Weight& query_key_weight,
                                     const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    launch_q4(x, query_key_weight, q, k, stream);
    launch_q5(x, gate_value_weight, gate, v, stream);
}

} // namespace ninfer::ops::detail
