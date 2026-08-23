#include "ops/linear/fp8/fp8_launch.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear/fp8/fp8_small_t.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename Fp8LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    fp8_small_t_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, kFp8FirstSmallT + static_cast<int>(Offsets)>...};
}

template <class Geometry, std::int32_t LastToken = kFp8LinearSmallTMax<Geometry>>
const auto& launchers() {
    static constexpr auto kLaunchers =
        make_launchers<Geometry>(std::make_index_sequence<LastToken - kFp8FirstSmallT + 1>{});
    return kLaunchers;
}

template <class Geometry>
void launch_registered(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] < kFp8FirstSmallT || x.ne[1] > kFp8LinearSmallTMax<Geometry>) {
        throw std::invalid_argument("fp8 linear small-T: unsupported T for problem");
    }
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kFp8FirstSmallT);
    launchers<Geometry>()[index](x, weight, out, stream);
}

} // namespace

void launch_fp8_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    switch (resolve_fp8_problem(weight.n, weight.k)) {
    case Fp8Problem::AttnInput:
        launch_registered<Fp8AttnInputGeometry>(x, weight, out, stream);
        return;
    case Fp8Problem::GdnInput:
        launch_registered<Fp8GdnInputGeometry>(x, weight, out, stream);
        return;
    case Fp8Problem::MlpGateUp:
        launch_registered<Fp8MlpGateUpGeometry>(x, weight, out, stream);
        return;
    case Fp8Problem::Vocabulary:
#ifdef NINFER_VOLTA_BUILD
        launch_registered<Fp8VocabularyGeometry>(x, weight, out, stream);
        return;
#else
        break;
#endif
    case Fp8Problem::Residual6144:
        launch_registered<Fp8Residual6144Geometry>(x, weight, out, stream);
        return;
    case Fp8Problem::Residual17408:
        launch_registered<Fp8Residual17408Geometry>(x, weight, out, stream);
        return;
    }
    throw std::logic_error("FP8 vocabulary small-T uses its A16 MMA route");
}

} // namespace ninfer::ops::detail
