#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include "ops/linear_swiglu/w8/w8_linear_swiglu_output.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kIntermediate = 6144;
constexpr int kHidden       = 2048;
constexpr int kFirstExactT  = 2;
constexpr int kLastExactT   = 48;
using ProjectionLauncher    = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int TileCols = ActiveCols <= 8    ? 8
                             : ActiveCols <= 16 ? 16
                             : ActiveCols <= 24 ? 24
                             : ActiveCols <= 32 ? 32
                             : ActiveCols <= 40 ? 40
                                                : 48;
    constexpr auto ScaleAccess =
        ActiveCols > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct;
    using Geometry  = W8LinearGeometry<2 * kIntermediate, kHidden>;
    using RowPolicy = W8SwiGluPairedRows<kIntermediate>;
    using Schedule =
        std::conditional_t<(ActiveCols <= 32), W8SmallTMmaDefaultSchedule<TileCols, ActiveCols>,
                           W8SmallTMmaSchedule<4, TileCols, 3, ScaleAccess>>;
    const W8ContiguousOutput ignored_output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const W8SwiGluDirectEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const RowPolicy row_policy{};
    w8_small_t_mma_kernel<Geometry, ActiveCols, Schedule, W8ContiguousOutput,
                          W8SwiGluDirectEpilogue, RowPolicy, true>
        <<<kIntermediate / RowPolicy::kOutputRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), ignored_output, epilogue, row_policy);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<kFirstExactT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kLastExactT - kFirstExactT + 1>{});

} // namespace

void w8_linear_swiglu_splitk_exact_t_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            cudaStream_t stream) {
    if (x.ne[1] < kFirstExactT || x.ne[1] > kLastExactT) {
        throw std::invalid_argument("W8 LinearSwiGLU exact split-K requires T=2..48");
    }
    kLaunchers[x.ne[1] - kFirstExactT](x, w, out, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
