#include "core/device.h"
#include "ops/linear/fp8/fp8_launch.h"
#include "ops/linear/fp8/fp8_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

#ifdef NINFER_VOLTA_BUILD

namespace {
using S = Fp8VoltaQpnSchedule;
static_assert(kFp8VoltaQpnRowsPerTile == S::kRowsPerTile,
              "dispatch mirrors the tile height for host code");
} // namespace

bool fp8_volta_qpn_supported(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    if (n <= 0 || t <= 0) { return false; }
    // A lane consumes whole 128-byte lines, and the CTA's SPLITK warps split K by whole lines.
    // 8 is the floor every shipped SPLITK choice (8 or 16) divides down to (see the launcher).
    if (k % S::kKPerBlock != 0) { return false; }
    if (k / S::kKPerBlock < 8) { return false; }
    // Two 8-row A tiles cover T=9..16 in one pass over the weights. The W8 sibling rejected its
    // two-tile form because the 32x8 fused route already absorbed those rows for free; FP8 has no
    // 32x8 route, so here the alternative is chunking -- reading the whole weight twice -- and the
    // second tile wins easily.
    //
    // T=1 is excluded: the SIMT decode kernel reads 446.7 GB/s there against this kernel's 434.5,
    // because a single row leaves seven of the tile's eight A rows padded with zeros.
    return t >= 2 && t <= 4 * S::kRowsPerTile;
}

void launch_fp8_volta_qpn(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t n = out.ne[0];
    launch_fp8_volta_qpn_with_output(x, w, Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data),
                                                               n},
                                     n, stream);
}

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
