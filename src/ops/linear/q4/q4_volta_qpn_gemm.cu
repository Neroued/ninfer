#include "core/device.h"
#include "ops/linear/q4/q4_launch.h"
#include "ops/linear/q4/q4_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
using S = Q4VoltaQpnSchedule;
} // namespace

bool q4_volta_qpn_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    if (n <= 0 || t <= 0) { return false; }
    // Whole Q4 groups per lane (the scale is per group and held across it), and one uint4 of
    // activations per mma pair.
    if (k % Q4RowSplitStorage::kGroupK != 0 || k % 8 != 0) { return false; }
    // The CTA's four warps split K by whole groups, so every warp needs at least one.
    if (k / Q4RowSplitStorage::kGroupK < S::kWarps) { return false; }
    return t <= 2 * S::kRowsPerTile;
}

void launch_q4_volta_qpn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream,
                         std::int32_t weight_row_offset) {
    const std::int32_t n      = out.ne[0];
    const std::int32_t k      = x.ne[0];
    const std::int32_t t      = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const std::int32_t padded_groups = w.padded_shape[1] / Q4RowSplitStorage::kGroupK;

    const std::int64_t roff = static_cast<std::int64_t>(weight_row_offset) * padded_groups;
    const auto* codes =
        static_cast<const std::uint8_t*>(w.qdata) + roff * Q4RowSplitStorage::kCodeBytesPerGroup;
    const auto* scales =
        static_cast<const std::uint8_t*>(w.scales) + roff * Q4RowSplitStorage::kScaleBytesPerGroup;

    const dim3 grid(static_cast<unsigned>((n + S::kColsPerCta - 1) / S::kColsPerCta));
    // kBlk = 4 groups = 128 B, one cache line per lane per iteration; measured best at every
    // shape (see q4_volta_qpn_gemm.cuh and the V100 implementation).
#define NINFER_QPN_LAUNCH(TILES, BLK)                                                             \
    q4_volta_qpn_gemm_kernel<TILES, BLK><<<grid, S::kThreads, 0, stream>>>(                       \
        codes, scales, static_cast<const __nv_bfloat16*>(x.data),                                 \
        static_cast<__nv_bfloat16*>(out.data), n, k, t, padded_groups, out_ld)
    if (t <= S::kRowsPerTile) {
        NINFER_QPN_LAUNCH(1, 4);
    } else {
        NINFER_QPN_LAUNCH(2, 4);
    }
#undef NINFER_QPN_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
