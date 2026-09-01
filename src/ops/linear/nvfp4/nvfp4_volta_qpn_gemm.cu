#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
using S = Nvfp4VoltaQpnSchedule;
static_assert(kNvfp4VoltaQpnRowsPerTile == S::kRowsPerTile,
              "dispatch mirrors the tile height for host code");
} // namespace

bool nvfp4_volta_qpn_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    if (n <= 0 || t <= 0) { return false; }
    // The scale plane's BlockScaleK16M128x4 swizzle tiles on 128 output rows.
    if (n % 128 != 0) { return false; }
    // A lane consumes whole 64-k chunks (four NVFP4 groups, matching the swizzle's contiguous
    // scale-byte run), and the CTA's SPLITK warps split K by whole chunks. SPLITK=8 is the
    // production floor across every kTiles bucket (see launch_nvfp4_volta_qpn_with_output).
    if (k % S::kChunkK != 0) { return false; }
    if (k / S::kChunkK < 8) { return false; }
    // Prepacked runtime weights also use QPN2 at T=1; checkpoint-native weights keep the existing
    // SIMT decode route through the semantic-op plan.
    return t >= 1 && t <= 4 * S::kRowsPerTile;
}

void launch_nvfp4_volta_qpn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n                   = out.ne[0];
    const float inverse_weight_divisor = 1.0F / w.weight_scale_divisor;
    launch_nvfp4_volta_qpn_with_output(
        x, w, Nvfp4ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), n}, n,
        inverse_weight_divisor, stream);
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
