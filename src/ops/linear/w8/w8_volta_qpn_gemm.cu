#include "core/device.h"
#include "ops/linear/w8/w8_launch.h"
#include "ops/linear/w8/w8_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
using S = W8VoltaQpnSchedule;
} // namespace

bool w8_volta_qpn_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    if (n <= 0 || t <= 0) { return false; }
    // Whole W8 groups per lane (the scale is per group and held across it), and one uint4 of
    // activations per mma pair.
    if (k % W8RowSplitStorage::kGroupK != 0 || k % 8 != 0) { return false; }
    // The CTA's four warps split K by whole groups, so every warp needs at least one.
    if (k / W8RowSplitStorage::kGroupK < S::kWarps) { return false; }
    // One 8-row A tile only. A two-tile form covering T=9..16 was measured and loses: it doubles
    // the MMA work for a second tile the shape does not need, reading 5115-5658us against the
    // 32x8 fused route's 5390-5502 over that span. The 32x8 geometry absorbs those rows for free,
    // which is exactly the crossover the Q4 sibling shows.
    return t <= S::kRowsPerTile;
}

void launch_w8_volta_qpn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_groups = w.padded_shape[1] / W8RowSplitStorage::kGroupK;

    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales = static_cast<const std::uint8_t*>(w.scales);
    auto* out_data     = static_cast<__nv_bfloat16*>(out.data);

    const dim3 grid(static_cast<unsigned>((n + S::kColsPerCta - 1) / S::kColsPerCta));
    // kBlk = 4 groups = 128 B, one cache line per lane per iteration; see the Q4 sibling for why
    // that, rather than an offline weight prepack, is what keeps the scattered per-lane row reads
    // off DRAM.
    w8_volta_qpn_gemm_kernel<1, 4><<<grid, S::kThreads, 0, stream>>>(
        codes, scales, static_cast<const __nv_bfloat16*>(x.data), out_data, n, k, t,
        padded_groups, out_ld);
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
