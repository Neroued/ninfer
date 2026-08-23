#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_small_t.cuh"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_output.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Geometry              = Fp8MlpGateUpGeometry;
using Launch                = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);
constexpr int kIntermediate = Geometry::kOutputRows / 2;

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename Fp8LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert((Schedule::kRowsPerWarp % 2) == 0);
    using Rows                = Fp8SwiGluRows<Schedule::kRowsPerWarp / 2, kIntermediate>;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    fp8_small_t_kernel<Geometry, ActiveTokens, Schedule, Fp8SwiGluOutput, Fp8IdentityEpilogue, Rows,
                       true><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales),
        Fp8SwiGluOutput{static_cast<__nv_bfloat16*>(out.data), kIntermediate}, {}, Rows{});
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<kFp8FirstSmallT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kFp8LinearSmallTMax<Geometry> - kFp8FirstSmallT + 1>{});

} // namespace

void fp8_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream) {
    if (x.ne[1] < kFp8FirstSmallT || x.ne[1] > kFp8LinearSmallTMax<Geometry>) {
        throw std::invalid_argument("fp8 linear_swiglu small-T: unsupported T");
    }
    kLaunchers[static_cast<std::size_t>(x.ne[1] - kFp8FirstSmallT)](x, weight, out, stream);
}

} // namespace ninfer::ops::detail
