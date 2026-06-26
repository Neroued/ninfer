// qus::kernels - rmsnorm launcher: grid/block/stream configuration + kernel launch.
#include "kernels/launcher/rmsnorm.h"

#include "kernels/kernel/rmsnorm.cuh"
#include "qus/core/device.h" // CUDA_CHECK

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace qus::kernels::detail {

void rmsnorm_launch(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                    const Tensor* z, Tensor& out, cudaStream_t stream) {
    constexpr int kBlock = 256;
    const std::int32_t d = x.ne[0];
    if (d <= 0) { throw std::invalid_argument("rmsnorm: ne[0] must be positive"); }
    const std::int64_t rows = out.numel() / d;
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("rmsnorm: row count exceeds CUDA grid limit");
    }
    const auto x_addr = reinterpret_cast<std::uintptr_t>(x.data);
    const auto w_addr = reinterpret_cast<std::uintptr_t>(weight.data);
    const auto o_addr = reinterpret_cast<std::uintptr_t>(out.data);
    const bool aligned2 = ((x_addr | w_addr | o_addr) & (alignof(__nv_bfloat162) - 1)) == 0;

    if (d == 5120 && z == nullptr && aligned2) {
        rmsnorm_d5120_kernel<<<static_cast<unsigned int>(rows), 512, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data), static_cast<__nv_bfloat16*>(out.data),
            rows, eps, unit_offset);
    } else {
        rmsnorm_kernel<<<static_cast<unsigned int>(rows), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data),
            z != nullptr ? static_cast<const __nv_bfloat16*>(z->data) : nullptr,
            static_cast<__nv_bfloat16*>(out.data), d, rows, eps, unit_offset);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
