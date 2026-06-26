// qus::kernels - linear Q6 launcher: GEMV/GEMM grid/block/stream setup.
#include "kernels/launcher/linear.h"

#include "kernels/kernel/linear_q6.cuh"
#include "qus/core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

constexpr int kBlock = 128;

int grid_for(std::int64_t n) {
    const std::int64_t grid = (n + kBlock - 1) / kBlock;
    if (grid > std::numeric_limits<int>::max()) {
        throw std::overflow_error("linear: Q6 launch grid exceeds CUDA limit");
    }
    return static_cast<int>(std::max<std::int64_t>(1, grid));
}

const std::uint8_t* payload_ptr(const Weight& weight) {
    const void* payload = weight.payload != nullptr ? weight.payload : weight.qdata;
    return static_cast<const std::uint8_t*>(payload);
}

} // namespace

void linear_q6_gemv_launch(const Tensor& x, const Weight& weight, Tensor& out,
                           cudaStream_t stream) {
    const std::int32_t n        = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t padded_k = weight.padded_shape[1];
    linear_q6_gemv_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), payload_ptr(weight),
        static_cast<__nv_bfloat16*>(out.data), n, k, padded_k);
    CUDA_CHECK(cudaGetLastError());
}

void linear_q6_gemm_launch(const Tensor& x, const Weight& weight, Tensor& out,
                           cudaStream_t stream) {
    const std::int32_t n        = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t t        = x.ne[1];
    const std::int32_t padded_k = weight.padded_shape[1];
    const std::int64_t elements = static_cast<std::int64_t>(n) * t;
    linear_q6_gemm_kernel<<<grid_for(elements), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), payload_ptr(weight),
        static_cast<__nv_bfloat16*>(out.data), n, k, t, padded_k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
