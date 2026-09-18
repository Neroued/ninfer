#include "core/weight.h"
#include "ops/linear_add/q8/q8_linear_add_kernels.h"

#include "core/device.h"
#include "ops/linear/q8/q8_ksplit_mma.cuh"
#include "ops/linear/q8/q8_ksplit_grouped_mma.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows           = 2048;
constexpr int kRowsPerCta     = 16;
constexpr int kFirstExactCols = 2;
constexpr int kLastExactCols  = 48;
using ProjectionLauncher      = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int Hidden, int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& weight, Tensor& residual_out,
                        cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8    ? 8
                             : ActiveCols <= 16 ? 16
                             : ActiveCols <= 24 ? 24
                             : ActiveCols <= 32 ? 32
                             : ActiveCols <= 40 ? 40
                                                : 48;
    constexpr int KWarps =
        Hidden == 4096 ? (ActiveCols <= 12 ? 16 : 8) : (ActiveCols <= 32 ? 8 : 4);
    constexpr int MinBlocks = Hidden == 4096 ? (KWarps == 16 ? 1 : 2) : (ActiveCols <= 32 ? 2 : 3);
    constexpr auto ScaleAccess =
        ActiveCols > 4 ? Q8KSplitScaleAccess::Shared : Q8KSplitScaleAccess::Direct;
    constexpr auto ActivationCache =
        Hidden == 4096 && (ActiveCols == 4 || (ActiveCols >= 27 && ActiveCols <= 40)) ? Cache::cg
                                                                                      : Cache::ca;
    using Geometry = Q8LinearGeometry<kRows, Hidden>;
    using Schedule = Q8KSplitSchedule<KWarps, TileCols, MinBlocks, ScaleAccess, ActivationCache>;
    static_assert((kRows % kRowsPerCta) == 0);
    auto* residual = static_cast<__nv_bfloat16*>(residual_out.data);
    const Q8ContiguousOutput output{residual, kRows};
    q8_ksplit_mma_kernel<Geometry, ActiveCols, Schedule, Q8ContiguousOutput,
                         Q8KSplitResidualEpilogue>
        <<<kRows / kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, Q8KSplitResidualEpilogue{});
}

template <int Hidden, std::size_t... Offsets>
constexpr auto make_projection_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<Hidden, kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kK4096ProjectionLaunchers = make_projection_launchers<4096>(
    std::make_index_sequence<kLastExactCols - kFirstExactCols + 1>{});
constexpr auto kK6144ProjectionLaunchers = make_projection_launchers<6144>(
    std::make_index_sequence<kLastExactCols - kFirstExactCols + 1>{});

template <int Hidden, int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_medium(const Tensor& x, Tensor& residual_out, const Weight& weight,
                   cudaStream_t stream) {
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(residual_out.data), kRows};
    q8_ksplit_grouped_mma_kernel<Hidden, TileCols, KSplits, NGroups, MinBlocks, Q8ContiguousOutput,
                                 true><<<kRows / kRowsPerCta, KSplits * NGroups * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
}

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void dispatch_medium_shape(const Tensor& x, const Weight& weight, Tensor& residual_out,
                           cudaStream_t stream) {
    if (weight.k == 4096) {
        launch_medium<4096, TileCols, KSplits, NGroups, MinBlocks>(x, residual_out, weight, stream);
    } else {
        launch_medium<6144, TileCols, KSplits, NGroups, MinBlocks>(x, residual_out, weight, stream);
    }
}

} // namespace

void q8_linear_add_splitk_mma_launch(const Tensor& x, const Weight& weight, Tensor& residual_out,
                                     cudaStream_t stream) {
    if (x.ne[1] < kFirstExactCols || x.ne[1] > kLastExactCols) {
        throw std::invalid_argument("Q8 linear_add split-K MMA requires exact T=2..48");
    }
    if (weight.k == 6144) {
        kK6144ProjectionLaunchers[x.ne[1] - kFirstExactCols](x, weight, residual_out, stream);
    } else {
        kK4096ProjectionLaunchers[x.ne[1] - kFirstExactCols](x, weight, residual_out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void q8_linear_add_medium_splitk_launch(const Tensor& x, const Weight& weight, Tensor& residual_out,
                                        cudaStream_t stream) {
    const std::int32_t t = x.ne[1];
    if ((weight.k != 4096 && weight.k != 6144) || t < 49 || t > 128) {
        throw std::invalid_argument("Q8 linear_add medium split-K requires T=49..128");
    }
    if (t <= 64) {
        dispatch_medium_shape<64, 8, 4, 1>(x, weight, residual_out, stream);
    } else if (t == 65) {
        dispatch_medium_shape<80, 8, 2, 1>(x, weight, residual_out, stream);
    } else if (t <= 72) {
        dispatch_medium_shape<72, 8, 3, 1>(x, weight, residual_out, stream);
    } else if (t <= 80) {
        dispatch_medium_shape<80, 8, 2, 1>(x, weight, residual_out, stream);
    } else if (t <= 96) {
        dispatch_medium_shape<96, 4, 6, 1>(x, weight, residual_out, stream);
    } else if (t <= 112) {
        dispatch_medium_shape<112, 4, 7, 1>(x, weight, residual_out, stream);
    } else if (t <= 120) {
        dispatch_medium_shape<120, 4, 5, 1>(x, weight, residual_out, stream);
    } else if (t <= 125) {
        dispatch_medium_shape<128, 4, 4, 1>(x, weight, residual_out, stream);
    } else {
        dispatch_medium_shape<128, 4, 8, 1>(x, weight, residual_out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
