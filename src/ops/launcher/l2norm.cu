// ninfer::ops - L2Norm launcher.
#include "ops/launcher/l2norm.h"

#include "ops/common/math.h"
#include "ops/kernel/l2norm.cuh"
#include "core/device.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

void l2norm_launch(const Tensor& x, float eps, Tensor& out, cudaStream_t stream) {
    const std::int32_t d = x.ne[0];
    if (d <= 0) { throw std::invalid_argument("l2norm: ne[0] must be positive"); }
    const std::int64_t rows = out.numel() / d;
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("l2norm: row count exceeds CUDA grid limit");
    }

    const auto x_addr   = reinterpret_cast<std::uintptr_t>(x.data);
    const auto o_addr   = reinterpret_cast<std::uintptr_t>(out.data);
    const bool aligned2 = ((x_addr | o_addr) & (alignof(__nv_bfloat162) - 1)) == 0;

    if (aligned2 && d >= 64 && d <= 256 && d % 64 == 0) {
        constexpr int kBlock         = 512;
        constexpr int kWarpsPerBlock = kBlock / kWarpSize;
        const auto blocks =
            static_cast<unsigned int>(div_up(rows, static_cast<std::int64_t>(kWarpsPerBlock)));
        l2norm_warp_bf16x2_kernel<kBlock>
            <<<blocks, kBlock, 0, stream>>>(static_cast<const __nv_bfloat162*>(x.data),
                                            static_cast<__nv_bfloat162*>(out.data), d, rows, eps);
    } else {
        constexpr int kBlock        = 512;
        constexpr int kRowsPerBlock = kBlock / kWarpSize;
        const auto blocks =
            static_cast<unsigned int>(div_up(rows, static_cast<std::int64_t>(kRowsPerBlock)));
        l2norm_generic_kernel<<<blocks, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data), d,
            rows, eps);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
