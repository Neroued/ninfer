#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear/fp8/fp8_small_t.cuh"
#include "ops/linear_add/fp8/fp8_linear_add_epilogue.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class Geometry, int ActiveTokens>
struct Fp8LinearAddSmallTProductionSchedule;

// LinearAdd's residual read and fused epilogue shift the register/occupancy crossovers relative to
// contiguous Linear. These are the measured RTX 5090 ranges for [5120,6144].
template <int ActiveTokens>
struct Fp8LinearAddSmallTProductionSchedule<Fp8Residual6144Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kFp8FirstSmallT && ActiveTokens <= kFp8LastSmallT);
    static constexpr int kWarpsPerCta =
        (ActiveTokens >= 15 && ActiveTokens <= 19) || ActiveTokens == 22 ? 4 : 8;
    static constexpr int kRowsPerWarp   = ActiveTokens <= 5 ? 1 : 2;
    static constexpr int kValuesPerLane = ActiveTokens >= 20 && ActiveTokens <= 23 ? 8 : 16;
    static constexpr int kTokenTile     = ActiveTokens == 24 ? 12 : ActiveTokens;
    static constexpr auto kCodeCache =
        (ActiveTokens >= 12 && ActiveTokens <= 14) || ActiveTokens == 23 ? Fp8CodeCache::Streaming
                                                                         : Fp8CodeCache::Default;
    static constexpr auto kBlockOrder = ActiveTokens == 24
                                            ? Fp8SmallTBlockOrder::TokenTilesContiguous
                                            : Fp8SmallTBlockOrder::RowsContiguous;
    using Type =
        Fp8SmallTSchedule<kWarpsPerCta, kRowsPerWarp, kValuesPerLane, kTokenTile, 1,
                          Fp8SmallTActivationAccess::TokenPacked, kCodeCache, 1, kBlockOrder, 1>;
};

// The longer K of [5120,17408] favors streaming code loads around its two register crossovers;
// keeping the default v16 schedule at T=9..16 avoids an isolated fast point followed by a cliff.
template <int ActiveTokens>
struct Fp8LinearAddSmallTProductionSchedule<Fp8Residual17408Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kFp8FirstSmallT && ActiveTokens <= kFp8LastSmallT);
    static constexpr int kWarpsPerCta   = ActiveTokens >= 21 && ActiveTokens <= 22 ? 4 : 8;
    static constexpr int kRowsPerWarp   = ActiveTokens <= 5 ? 1 : 2;
    static constexpr int kValuesPerLane = ActiveTokens >= 20 && ActiveTokens <= 23 ? 8 : 16;
    static constexpr int kTokenTile     = ActiveTokens == 24 ? 12 : ActiveTokens;
    static constexpr auto kCodeCache    = (ActiveTokens >= 6 && ActiveTokens <= 8) ||
                                               (ActiveTokens >= 17 && ActiveTokens <= 20) ||
                                               ActiveTokens == 23
                                              ? Fp8CodeCache::Streaming
                                              : Fp8CodeCache::Default;
    static constexpr auto kBlockOrder   = ActiveTokens == 24
                                              ? Fp8SmallTBlockOrder::TokenTilesContiguous
                                              : Fp8SmallTBlockOrder::RowsContiguous;
    using Type =
        Fp8SmallTSchedule<kWarpsPerCta, kRowsPerWarp, kValuesPerLane, kTokenTile, 1,
                          Fp8SmallTActivationAccess::TokenPacked, kCodeCache, 1, kBlockOrder, 1>;
};

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Schedule = typename Fp8LinearAddSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    auto* output              = static_cast<__nv_bfloat16*>(residual.data);
    fp8_small_t_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales),
            Fp8ContiguousOutput{output, Geometry::kOutputRows},
            Fp8AddResidualEpilogue{output, Geometry::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, kFp8FirstSmallT + static_cast<int>(Offsets)>...};
}

template <class Geometry>
const auto& launchers() {
    static constexpr auto kLaunchers =
        make_launchers<Geometry>(std::make_index_sequence<kFp8LastSmallT - kFp8FirstSmallT + 1>{});
    return kLaunchers;
}

} // namespace

void fp8_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream) {
    if (x.ne[1] < kFp8FirstSmallT || x.ne[1] > kFp8LastSmallT) {
        throw std::invalid_argument("fp8 linear_add small-T: unsupported T");
    }
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kFp8FirstSmallT);
    switch (resolve_fp8_problem(weight.n, weight.k)) {
    case Fp8Problem::Residual6144:
        launchers<Fp8Residual6144Geometry>()[index](x, weight, residual, stream);
        return;
    case Fp8Problem::Residual17408:
        launchers<Fp8Residual17408Geometry>()[index](x, weight, residual, stream);
        return;
    case Fp8Problem::AttnInput:
    case Fp8Problem::GdnInput:
    case Fp8Problem::MlpGateUp:
    case Fp8Problem::Vocabulary:
        break;
    }
    throw std::invalid_argument("fp8 linear_add small-T: unsupported problem");
}

} // namespace ninfer::ops::detail
