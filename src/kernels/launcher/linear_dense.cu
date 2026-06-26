// qus::kernels - linear dense launcher: GEMV/GEMM grid/block/stream setup.
#include "kernels/launcher/linear.h"

#include "kernels/kernel/linear_dense.cuh"
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
        throw std::overflow_error("linear: dense launch grid exceeds CUDA limit");
    }
    return static_cast<int>(std::max<std::int64_t>(1, grid));
}

bool is_fp32_weight(const Tensor& weight) { return weight.dtype == DType::FP32; }

} // namespace

void linear_dense_gemv_launch(const Tensor& x, const Tensor& weight, Tensor& out,
                              cudaStream_t stream) {
    const std::int32_t n = out.ne[0];
    const std::int32_t k = x.ne[0];
    linear_dense_gemv_kernel<<<grid_for(n), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), weight.data,
        static_cast<__nv_bfloat16*>(out.data), n, k, is_fp32_weight(weight));
    CUDA_CHECK(cudaGetLastError());
}

void linear_dense_gemm_launch(const Tensor& x, const Tensor& weight, Tensor& out,
                              cudaStream_t stream) {
    const std::int32_t n        = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t t        = x.ne[1];
    const std::int64_t elements = static_cast<std::int64_t>(n) * t;
    linear_dense_gemm_kernel<<<grid_for(elements), kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), weight.data,
        static_cast<__nv_bfloat16*>(out.data), n, k, t, is_fp32_weight(weight));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
