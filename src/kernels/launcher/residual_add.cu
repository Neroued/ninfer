// qus::kernels - residual_add launcher: grid/block/stream configuration + kernel launch.
// The only translation unit that includes this op's kernel header.
// See docs/l1-kernel-layering.md section 4.
#include "kernels/launcher/residual_add.h"

#include "kernels/common/math.h"
#include "kernels/kernel/residual_add.cuh"
#include "qus/core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>

namespace qus::kernels::detail {

void residual_add_launch(const Tensor& y, Tensor& x, cudaStream_t stream) {
    const std::int64_t n  = x.numel();
    constexpr int kBlock  = 128;
    const auto y_addr = reinterpret_cast<std::uintptr_t>(y.data);
    const auto x_addr = reinterpret_cast<std::uintptr_t>(x.data);
    if (((y_addr | x_addr) & (alignof(__nv_bfloat162) - 1)) != 0) {
        const int scalar_grid =
            static_cast<int>(div_up(n, static_cast<std::int64_t>(kBlock)));
        residual_add_scalar_kernel<<<scalar_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(y.data), static_cast<__nv_bfloat16*>(x.data), n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const std::int64_t n2 = n / 2;
    constexpr std::int64_t kPairsPerBlock = kBlock * kResidualAddPairsPerThread;
    const int grid =
        static_cast<int>(std::max<std::int64_t>(1, div_up(n2, kPairsPerBlock)));

    residual_add_kernel<<<grid, kBlock, 0, stream>>>(static_cast<const __nv_bfloat16*>(y.data),
                                                     static_cast<__nv_bfloat16*>(x.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
