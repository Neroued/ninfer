#include "ops/linear_pair/w8/w8_pair_kernels.h"
#include "ops/linear_pair/w8/w8_pair_plan.h"

#include "core/device.h"
#include "ops/linear/w8/w8_rowsplit_gemm_exact_t_splitk.cuh"
#include "ops/linear/w8/w8_rowsplit_gemm_medium_t_splitk.cuh"

#include <cuda_bf16.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows        = 1024;
constexpr int kHidden      = 2048;
constexpr int kRowsPerCta  = 8;
constexpr int kFirstExactT = 2;
constexpr int kLastExactT  = 32;
using PairOutput           = W8SplitOutput2<kRows, kRows>;
using PairLauncher         = void (*)(const Tensor&, const Weight&, const Weight&, Tensor&, Tensor&,
                              cudaStream_t);

struct W8PairExactTRows {
    static constexpr int kOutputRowsPerCta = kRowsPerCta;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & (kRowsPerCta - 1)) +
               (local_row >= kRowsPerCta ? kRows : 0);
    }
};

struct W8PairExactTEpilogue {
    __nv_bfloat16* first;
    __nv_bfloat16* second;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, float (&projected)[ActiveCols]) const {
        constexpr unsigned kPairMask = 0x0000ffffu;
        const int lane               = static_cast<int>(threadIdx.x) & 31;
        const int output_row         = row - lane + (lane & (kRowsPerCta - 1));
#pragma unroll
        for (int token = 0; token < ActiveCols; ++token) {
            const float second_value =
                __shfl_sync(kPairMask, projected[token], (lane & (kRowsPerCta - 1)) + kRowsPerCta);
            if (lane < kRowsPerCta) {
                const std::int64_t offset = static_cast<std::int64_t>(token) * kRows + output_row;
                first[offset]             = __float2bfloat16_rn(projected[token]);
                second[offset]            = __float2bfloat16_rn(second_value);
            }
        }
    }
};

template <int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                        Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    const auto* first_codes  = static_cast<const std::uint8_t*>(first_weight.qdata);
    const auto* first_scales = static_cast<const std::uint8_t*>(first_weight.scales);
    if (static_cast<const std::uint8_t*>(second_weight.qdata) != first_codes + kRows * kHidden ||
        static_cast<const std::uint8_t*>(second_weight.scales) !=
            first_scales + kRows * (kHidden / 32) * 2) {
        throw std::invalid_argument("W8 exact pair requires adjacent K/V row views");
    }

    const W8ContiguousOutput ignored{static_cast<__nv_bfloat16*>(first_out.data), kRows};
    const W8PairExactTEpilogue epilogue{static_cast<__nv_bfloat16*>(first_out.data),
                                        static_cast<__nv_bfloat16*>(second_out.data)};
    w8_rowsplit_exact_t_splitk_kernel<kHidden, TileCols, ActiveCols, W8ContiguousOutput,
                                      W8PairExactTEpilogue, W8PairExactTRows>
        <<<kRows / kRowsPerCta, 8 * 32, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data),
                                                     first_codes, first_scales, ignored, epilogue,
                                                     W8PairExactTRows{});
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<PairLauncher, sizeof...(Offsets)>{
        &launch_active_cols<kFirstExactT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kLastExactT - kFirstExactT + 1>{});

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_medium(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                   Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    const auto* first_codes  = static_cast<const std::uint8_t*>(first_weight.qdata);
    const auto* first_scales = static_cast<const std::uint8_t*>(first_weight.scales);
    if (static_cast<const std::uint8_t*>(second_weight.qdata) != first_codes + kRows * kHidden ||
        static_cast<const std::uint8_t*>(second_weight.scales) !=
            first_scales + kRows * (kHidden / 32) * 2) {
        throw std::invalid_argument("W8 medium pair requires adjacent K/V row views");
    }
    const PairOutput output{static_cast<__nv_bfloat16*>(first_out.data),
                            static_cast<__nv_bfloat16*>(second_out.data)};
    w8_rowsplit_medium_t_splitk_kernel<kHidden, TileCols, KSplits, NGroups, MinBlocks>
        <<<(2 * kRows) / 16, KSplits * NGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), first_codes, first_scales, output, x.ne[1]);
}

} // namespace

void w8_pair_splitk_exact_t_launch(const Tensor& x, const Weight& first_weight,
                                   const Weight& second_weight, Tensor& first_out,
                                   Tensor& second_out, cudaStream_t stream) {
    if (x.ne[0] != kHidden || x.ne[1] < kFirstExactT || x.ne[1] > kLastExactT ||
        first_out.ne[0] != kRows || first_out.ne[1] != x.ne[1] || second_out.ne[0] != kRows ||
        second_out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("W8 exact pair requires [1024,2048] and T=2..32");
    }
    kLaunchers[x.ne[1] - kFirstExactT](x, first_weight, second_weight, first_out, second_out,
                                       stream);
    CUDA_CHECK(cudaGetLastError());
}

void w8_pair_splitk_medium_launch(W8PairScheduleId schedule, const Tensor& x,
                                  const Weight& first_weight, const Weight& second_weight,
                                  Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    if (x.ne[0] != kHidden || x.ne[1] < 33 || first_out.ne[0] != kRows ||
        first_out.ne[1] != x.ne[1] || second_out.ne[0] != kRows || second_out.ne[1] != x.ne[1]) {
        throw std::invalid_argument("W8 medium pair requires [1024,2048] and T>=33");
    }
    switch (schedule) {
    case W8PairScheduleId::DualSplitKMediumC48:
        if (x.ne[1] <= 48) {
            launch_medium<48, 4, 2, 3>(x, first_weight, second_weight, first_out, second_out,
                                       stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        break;
    case W8PairScheduleId::DualSplitKMediumC64:
        if (x.ne[1] <= 64) {
            launch_medium<64, 4, 2, 2>(x, first_weight, second_weight, first_out, second_out,
                                       stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        break;
    case W8PairScheduleId::DualSplitKMediumC80:
        if (x.ne[1] <= 80) {
            launch_medium<80, 4, 2, 1>(x, first_weight, second_weight, first_out, second_out,
                                       stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        break;
    case W8PairScheduleId::DualSplitKMediumC88:
        if (x.ne[1] <= 88) {
            launch_medium<88, 4, 1, 1>(x, first_weight, second_weight, first_out, second_out,
                                       stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        break;
    case W8PairScheduleId::DualSplitKMediumC96:
        if (x.ne[1] <= 96) {
            launch_medium<96, 4, 1, 1>(x, first_weight, second_weight, first_out, second_out,
                                       stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        break;
    case W8PairScheduleId::DualSplitKMediumC104:
        if (x.ne[1] <= 104) {
            launch_medium<104, 4, 1, 1>(x, first_weight, second_weight, first_out, second_out,
                                        stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        break;
    case W8PairScheduleId::DualSplitKMediumC112:
        if (x.ne[1] <= 112) {
            launch_medium<112, 4, 1, 1>(x, first_weight, second_weight, first_out, second_out,
                                        stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        break;
    case W8PairScheduleId::DualSplitKMediumC128:
        if (x.ne[1] <= 128) {
            launch_medium<128, 2, 4, 2>(x, first_weight, second_weight, first_out, second_out,
                                        stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        break;
    case W8PairScheduleId::DualSplitKMediumC160:
        if (x.ne[1] <= 160) {
            launch_medium<160, 2, 5, 2>(x, first_weight, second_weight, first_out, second_out,
                                        stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        break;
    case W8PairScheduleId::DualSplitKMediumC192:
        if (x.ne[1] <= 192) {
            launch_medium<192, 2, 6, 2>(x, first_weight, second_weight, first_out, second_out,
                                        stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        break;
    case W8PairScheduleId::DualSplitKMediumC224:
        if (x.ne[1] <= 224) {
            launch_medium<224, 2, 7, 2>(x, first_weight, second_weight, first_out, second_out,
                                        stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        break;
    case W8PairScheduleId::DualSplitKMediumC256:
        if (x.ne[1] <= 256) {
            launch_medium<256, 2, 8, 1>(x, first_weight, second_weight, first_out, second_out,
                                        stream);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        break;
    default:
        break;
    }
    throw std::invalid_argument("W8 medium pair schedule does not cover this T");
}

} // namespace ninfer::ops::detail
