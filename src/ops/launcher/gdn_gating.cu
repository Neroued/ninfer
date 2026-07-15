// ninfer::ops - gdn_gating launcher: grid/block/stream configuration + kernel launch.
#include "ops/launcher/gdn_gating.h"

#include "ops/common/math.h"
#include "ops/kernel/gdn_gating.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

void gdn_gating_launch(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias,
                       Tensor& g, Tensor& beta, cudaStream_t stream) {
    const std::int64_t n = g.numel();
    constexpr int kBlock = 256;
    const int grid =
        static_cast<int>(std::max<std::int64_t>(1, div_up(n, static_cast<std::int64_t>(kBlock))));

    gdn_gating_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(a.data), static_cast<const __nv_bfloat16*>(b.data),
        static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
        static_cast<float*>(g.data), static_cast<float*>(beta.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
