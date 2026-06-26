// qus::kernels — silu_and_mul launcher: grid/block/stream configuration + kernel launch.
// The only translation unit that includes this op's kernel header.
// See docs/l1-kernel-layering.md §4.
#include "kernels/launcher/silu_and_mul.h"

#include "kernels/kernel/silu_and_mul.cuh"
#include "qus/core/device.h"  // CUDA_CHECK

#include <cstdint>

namespace qus::kernels::detail {

void silu_and_mul_launch(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    const std::int64_t n = out.numel();
    constexpr int kBlock = 1024;
    const int grid       = static_cast<int>((n + kBlock - 1) / kBlock);

    silu_and_mul_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate.data), static_cast<const __nv_bfloat16*>(up.data),
        static_cast<__nv_bfloat16*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
