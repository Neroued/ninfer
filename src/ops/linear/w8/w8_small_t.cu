#include "ops/linear/w8/w8_launch.h"

#include "core/device.h"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename W8LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);

    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    w8_small_t_mma_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int First, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<W8Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, First + static_cast<int>(Offsets)>...};
}

template <int Capacity>
void launch_vocabulary_tile(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry = W8VocabularyProjectionGeometry;
    using Schedule = W8SmallTMmaSchedule<Capacity <= 32 ? 8 : 4, Capacity, 2,
                                         W8SmallTMmaScaleAccess::Shared>;
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    w8_small_t_mma_kernel<Geometry, Capacity, Schedule, W8ContiguousOutput,
                          W8SmallTMmaStoreEpilogue, W8SmallTMmaIdentityRows, false, true>
        <<<Geometry::kOutputRows / 16, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, W8SmallTMmaStoreEpilogue{},
            W8SmallTMmaIdentityRows{}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... I>
constexpr auto vocabulary_launchers(std::index_sequence<I...>) {
    return std::array<W8Launch, sizeof...(I)>{&launch_vocabulary_tile<8 * (static_cast<int>(I) + 1)>...};
}
constexpr auto kVocabularyLaunchers = vocabulary_launchers(std::make_index_sequence<5>{});
constexpr auto kMtpInputLaunchers =
    make_launchers<W8MtpInputProjectionGeometry, kW8MtpInputFirstSmallT>(
        std::make_index_sequence<kW8MtpInputLastSmallT - kW8MtpInputFirstSmallT + 1>{});
constexpr auto kMtpAttentionLaunchers =
    make_launchers<W8MtpAttentionProjectionGeometry, kW8MtpAttentionFirstSmallT>(
        std::make_index_sequence<kW8MtpAttentionLastSmallT - kW8MtpAttentionFirstSmallT + 1>{});
constexpr auto kMtpAttentionOutputLaunchers =
    make_launchers<W8MtpAttentionOutputGeometry, kW8MtpAttentionOutputFirstSmallT>(
        std::make_index_sequence<kW8MtpAttentionOutputLastSmallT -
                                 kW8MtpAttentionOutputFirstSmallT + 1>{});
constexpr auto kMtpGateUpLaunchers =
    make_launchers<W8MtpGateUpProjectionGeometry, kW8MtpGateUpFirstSmallT>(
        std::make_index_sequence<kW8MtpGateUpLastSmallT - kW8MtpGateUpFirstSmallT + 1>{});
constexpr auto kMtpDownLaunchers =
    make_launchers<W8MtpDownProjectionGeometry, kW8MtpDownFirstSmallT>(
        std::make_index_sequence<kW8MtpDownLastSmallT - kW8MtpDownFirstSmallT + 1>{});
constexpr auto k35bMtpProjectionLaunchers = make_launchers<W835bMtpProjectionGeometry,
                                                           kW835bMtpProjectionFirstSmallT>(
    std::make_index_sequence<kW835bMtpProjectionLastSmallT - kW835bMtpProjectionFirstSmallT + 1>{});
constexpr auto kDFlash2AttentionLaunchers = make_launchers<W8DFlash2AttentionProjectionGeometry,
                                                           kW8DFlash2AttentionFirstSmallT>(
    std::make_index_sequence<kW8DFlash2AttentionLastSmallT - kW8DFlash2AttentionFirstSmallT + 1>{});

} // namespace

void launch_w8_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.n == W8VocabularyProjectionGeometry::kOutputRows &&
        weight.k == W8VocabularyProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W8VocabularyProjectionGeometry::kInputRows &&
        x.ne[1] >= kW8VocabularyFirstSmallT && x.ne[1] <= kW8VocabularyLastSmallT) {
        const std::size_t index = static_cast<std::size_t>((x.ne[1] - 1) / 8);
        kVocabularyLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == W8MtpInputProjectionGeometry::kOutputRows &&
        weight.k == W8MtpInputProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W8MtpInputProjectionGeometry::kInputRows &&
        x.ne[1] >= kW8MtpInputFirstSmallT && x.ne[1] <= kW8MtpInputLastSmallT) {
        const std::size_t index = static_cast<std::size_t>(x.ne[1] - kW8MtpInputFirstSmallT);
        kMtpInputLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == W8MtpAttentionProjectionGeometry::kOutputRows &&
        weight.k == W8MtpAttentionProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W8MtpAttentionProjectionGeometry::kInputRows &&
        x.ne[1] >= kW8MtpAttentionFirstSmallT && x.ne[1] <= kW8MtpAttentionLastSmallT) {
        const std::size_t index = static_cast<std::size_t>(x.ne[1] - kW8MtpAttentionFirstSmallT);
        kMtpAttentionLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == W8MtpAttentionOutputGeometry::kOutputRows &&
        weight.k == W8MtpAttentionOutputGeometry::kInputRows &&
        weight.padded_shape[1] == W8MtpAttentionOutputGeometry::kInputRows &&
        x.ne[1] >= kW8MtpAttentionOutputFirstSmallT && x.ne[1] <= kW8MtpAttentionOutputLastSmallT) {
        const std::size_t index =
            static_cast<std::size_t>(x.ne[1] - kW8MtpAttentionOutputFirstSmallT);
        kMtpAttentionOutputLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == W8MtpGateUpProjectionGeometry::kOutputRows &&
        weight.k == W8MtpGateUpProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W8MtpGateUpProjectionGeometry::kInputRows &&
        x.ne[1] >= kW8MtpGateUpFirstSmallT && x.ne[1] <= kW8MtpGateUpLastSmallT) {
        const std::size_t index = static_cast<std::size_t>(x.ne[1] - kW8MtpGateUpFirstSmallT);
        kMtpGateUpLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == W8MtpDownProjectionGeometry::kOutputRows &&
        weight.k == W8MtpDownProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W8MtpDownProjectionGeometry::kInputRows &&
        x.ne[1] >= kW8MtpDownFirstSmallT && x.ne[1] <= kW8MtpDownLastSmallT) {
        const std::size_t index = static_cast<std::size_t>(x.ne[1] - kW8MtpDownFirstSmallT);
        kMtpDownLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == W835bMtpProjectionGeometry::kOutputRows &&
        weight.k == W835bMtpProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W835bMtpProjectionGeometry::kInputRows &&
        x.ne[1] >= kW835bMtpProjectionFirstSmallT && x.ne[1] <= kW835bMtpProjectionLastSmallT) {
        const std::size_t index =
            static_cast<std::size_t>(x.ne[1] - kW835bMtpProjectionFirstSmallT);
        k35bMtpProjectionLaunchers[index](x, weight, out, stream);
        return;
    }
    if (weight.n == W8DFlash2AttentionProjectionGeometry::kOutputRows &&
        weight.k == W8DFlash2AttentionProjectionGeometry::kInputRows &&
        weight.padded_shape[1] == W8DFlash2AttentionProjectionGeometry::kInputRows &&
        x.ne[1] >= kW8DFlash2AttentionFirstSmallT && x.ne[1] <= kW8DFlash2AttentionLastSmallT) {
        const std::size_t index =
            static_cast<std::size_t>(x.ne[1] - kW8DFlash2AttentionFirstSmallT);
        kDFlash2AttentionLaunchers[index](x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("W8 Linear small-T: unsupported exact problem");
}

} // namespace ninfer::ops::detail
