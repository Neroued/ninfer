#include "core/weight.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear/fp8/fp8_simt.cuh"
#include "ops/linear_add/fp8/fp8_linear_add_epilogue.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// SIMT column extent controls the accumulator register count; retain exact small extents,
// with common register/load profiles and no per-shape cache hints or isolated token exceptions.
template <class Geometry, int ActiveTokens>
struct Fp8LinearAddSmallTProductionSchedule {
    static_assert(ActiveTokens >= 2 && ActiveTokens <= kFp8LinearAddChunkTokens);
    static constexpr int kWarpsPerCta   = ActiveTokens <= 19 ? 8 : 4;
    static constexpr int kRowsPerWarp   = ActiveTokens <= 5 ? 1 : 2;
    static constexpr int kValuesPerLane = ActiveTokens <= 19 ? 16 : 8;
    using Type = Fp8SimtSchedule<kWarpsPerCta, kRowsPerWarp, kValuesPerLane, ActiveTokens, 1,
                                 Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                 Fp8SimtBlockOrder::RowsContiguous, 1>;
};

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Schedule = typename Fp8LinearAddSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    auto* output              = static_cast<__nv_bfloat16*>(residual.data);
    fp8_simt_kernel<Geometry, ActiveTokens, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales),
        Fp8ContiguousOutput{output, Geometry::kOutputRows},
        Fp8AddResidualEpilogue{output, Geometry::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, 2 + static_cast<int>(Offsets)>...};
}

template <class Geometry>
const auto& launchers() {
    static constexpr auto kLaunchers =
        make_launchers<Geometry>(std::make_index_sequence<kFp8LinearAddChunkTokens - 2 + 1>{});
    return kLaunchers;
}

} // namespace

void fp8_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream) {
    if (x.ne[1] < 2 || x.ne[1] > kFp8LinearAddChunkTokens) {
        throw std::invalid_argument("fp8 linear_add small-T: unsupported T");
    }
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - 2);
    switch (resolve_fp8_geometry(weight.n, weight.k)) {
    case Fp8GeometryId::N5120K6144:
        launchers<Fp8N5120K6144>()[index](x, weight, residual, stream);
        return;
    case Fp8GeometryId::N5120K17408:
        launchers<Fp8N5120K17408>()[index](x, weight, residual, stream);
        return;
    case Fp8GeometryId::N14336K5120:
    case Fp8GeometryId::N16384K5120:
    case Fp8GeometryId::N34816K5120:
    case Fp8GeometryId::N248320K5120:
        break;
    }
    throw std::invalid_argument("fp8 linear_add small-T: unsupported problem");
}

} // namespace ninfer::ops::detail
