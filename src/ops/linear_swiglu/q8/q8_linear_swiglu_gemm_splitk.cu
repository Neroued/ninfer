#include "ops/linear/q8/q8_geometry.h"
#include "ops/linear/q8/q8_instances.cuh"
#include "core/weight.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_output.cuh"

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
    constexpr int TileCols     = ActiveCols <= 8    ? 8
                                 : ActiveCols <= 16 ? 16
                                 : ActiveCols <= 24 ? 24
                                 : ActiveCols <= 32 ? 32
                                 : ActiveCols <= 40 ? 40
                                                    : 48;
    constexpr auto ScaleAccess = ActiveCols > 4 ? Q8ScaleAccess::Shared : Q8ScaleAccess::Direct;
    using Geometry             = Q8LinearGeometry<2 * kIntermediate, kHidden>;
    using RowPolicy            = Q8SwiGluPairedRows<kIntermediate>;
    using Schedule = std::conditional_t<(ActiveCols <= 32), Q8SlicedKDefault<TileCols, ActiveCols>,
                                        Q8A16SlicedKMmaSchedule<TileCols, 4, 1, 3, ScaleAccess>>;
    const LinearBf16Output ignored_output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const Q8SwiGluDirectEpilogue epilogue{};

    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, ActiveCols, true>,
        RowPolicy>(q8_linear_operands(x, w), ignored_output, epilogue, stream);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<kFirstExactT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kLastExactT - kFirstExactT + 1>{});

} // namespace

void q8_linear_swiglu_splitk_exact_t_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            cudaStream_t stream) {
    if (x.ne[1] < kFirstExactT || x.ne[1] > kLastExactT) {
        throw std::invalid_argument("Q8 LinearSwiGLU exact split-K requires T=2..48");
    }
    kLaunchers[x.ne[1] - kFirstExactT](x, w, out, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
