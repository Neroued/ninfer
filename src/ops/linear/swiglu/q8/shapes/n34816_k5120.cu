#include "ops/linear/swiglu/q8/launch.cuh"
#include "ops/linear/kernels/q8/q8_instances.cuh"
#include "ops/linear/kernels/q8/q8_geometry.h"
#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/reduce.cuh"

#include <array>
#include <utility>

namespace ninfer::ops::detail {
namespace {
using Geometry                       = Q8LinearGeometry<34816, 5120>;
constexpr std::int32_t kIntermediate = Geometry::kOutputRows / 2;
constexpr std::int32_t kLastSmallT   = 40;
using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int Capacity>
void launch_tile(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule =
        Q8A16SlicedKMmaSchedule<Capacity, Capacity == 24 ? 8 : 4, 1, 2, Q8ScaleAccess::Shared>;
    using RowPolicy = Q8SwiGluPairedRows<kIntermediate>;
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    static_assert((kIntermediate % RowPolicy::kOutputRowsPerCta) == 0);

    const LinearBf16Output ignored_output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const Q8SwiGluDirectEpilogue epilogue{};

    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, Capacity, false>, RowPolicy>(
        q8_linear_operands(x, weight), ignored_output, epilogue, stream);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_tile<8 * (1 + static_cast<int>(Offsets))>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<kLastSmallT / 8>{});

} // namespace

LinearSwiGluLaunch select_q8_swiglu_n34816_k5120(int tokens) {
    if (tokens <= 40) return kLaunchers[(tokens - 1) / 8];
    if (tokens <= 63) return launch_q8_swiglu_mma<Q8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>;
    if (tokens <= 64) return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>>;
    if (tokens <= 80) return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 80, 128, 64, 8, 1, 2>>;
    if (tokens <= 96) return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 96, 128, 64, 8, 1, 2>>;
    return launch_q8_swiglu_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}
} // namespace ninfer::ops::detail
