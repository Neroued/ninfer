#include "ops/linear/q4/q4_launch.h"

#include "core/device.h"
#include "ops/linear/q4/q4_small_t_mma.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kFirstSmallT    = 2;
constexpr int kLastFullT      = 8;
constexpr int kLastOptimizedT = 20;
using FullGeometry            = Q4DraftHeadGeometry<5120>;
using OptimizedGeometry       = Q4DraftHeadGeometry<2048>;

template <class Geometry, int TileTokens, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Q4DraftSmallTSchedule;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);

    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    q4_small_t_mma_kernel<Geometry, TileTokens, ActiveTokens>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int First, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Q4Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, ((First + static_cast<int>(Offsets) + 7) / 8) * 8,
                      First + static_cast<int>(Offsets)>...};
}

constexpr auto kFullLaunchers = make_launchers<FullGeometry, kFirstSmallT>(
    std::make_index_sequence<kLastFullT - kFirstSmallT + 1>{});
constexpr auto kOptimizedLaunchers = make_launchers<OptimizedGeometry, kFirstSmallT>(
    std::make_index_sequence<kLastOptimizedT - kFirstSmallT + 1>{});

template <class Geometry>
bool matches(const Tensor& x, const Weight& weight) {
    return weight.n == Geometry::kOutputRows && weight.k == Geometry::kInputRows &&
           weight.padded_shape[1] == Geometry::kInputRows && x.ne[1] >= kFirstSmallT;
}

} // namespace

void launch_q4_draft_head_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                  cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    // q4_small_t_mma_kernel traps below sm_80 (see q4_small_t_mma.cuh) — this is a plain,
    // unfused projection (draft-head logits), so unlike linear_swiglu's fused case, it maps
    // directly onto the already-validated generic q4 SIMT kernel with no composition needed.
    launch_q4_simt_r8_c8(x, weight, out, stream);
    return;
#endif
    if (matches<FullGeometry>(x, weight) && x.ne[1] <= kLastFullT) {
        kFullLaunchers[static_cast<std::size_t>(x.ne[1] - kFirstSmallT)](x, weight, out, stream);
        return;
    }
    if (matches<OptimizedGeometry>(x, weight) && x.ne[1] <= kLastOptimizedT) {
        kOptimizedLaunchers[static_cast<std::size_t>(x.ne[1] - kFirstSmallT)](x, weight, out,
                                                                              stream);
        return;
    }
    throw std::invalid_argument("Q4 Linear draft-head small-T: unsupported exact problem");
}

} // namespace ninfer::ops::detail
