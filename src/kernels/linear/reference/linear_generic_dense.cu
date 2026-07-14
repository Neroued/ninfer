// ninfer::kernels::detail - generic dense linear GEMV/GEMM launchers.
#include "kernels/linear/reference/linear_generic.h"

#include "kernels/common/math.h"
#include "kernels/linear/reference/linear_generic_dense.cuh"
#include "ninfer/core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::kernels::detail {
namespace {

int checked_grid(std::int64_t blocks) {
    if (blocks <= 0) { return 1; }
    if (blocks > std::numeric_limits<int>::max()) {
        throw std::overflow_error("linear: dense launch grid exceeds CUDA limit");
    }
    return static_cast<int>(blocks);
}

dim3 tiled_gemm_grid_for(std::int32_t n, std::int32_t t) {
    const std::int64_t grid_m = div_up(static_cast<std::int64_t>(n),
                                       static_cast<std::int64_t>(kDenseGemmBlockRows));
    const std::int64_t grid_n = div_up(static_cast<std::int64_t>(t),
                                       static_cast<std::int64_t>(kDenseGemmBlockCols));
    if (grid_m > std::numeric_limits<unsigned>::max() ||
        grid_n > std::numeric_limits<unsigned>::max()) {
        throw std::overflow_error("linear: dense launch grid exceeds CUDA limit");
    }
    return dim3(static_cast<unsigned>(std::max<std::int64_t>(1, grid_m)),
                static_cast<unsigned>(std::max<std::int64_t>(1, grid_n)));
}

dim3 small_gemv_grid_for(std::int32_t n, std::int32_t t) {
    return dim3(static_cast<unsigned>(checked_grid(n)), static_cast<unsigned>(checked_grid(t)));
}

bool is_fp32_weight(const Tensor& weight) { return weight.dtype == DType::FP32; }

bool is_small_dense_prefill(std::int32_t n, std::int32_t t) {
    return n <= kDenseSmallGemvMaxN && t <= kDenseSmallGemvMaxT;
}

} // namespace

void linear_generic_dense_gemv_launch(const Tensor& x, const Tensor& weight, Tensor& out,
                                      cudaStream_t stream) {
    const std::int32_t n = out.ne[0];
    const std::int32_t k = x.ne[0];
    linear_generic_dense_gemv_kernel<<<checked_grid(n), kDenseGemvThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), weight.data,
        static_cast<__nv_bfloat16*>(out.data), n, k, is_fp32_weight(weight));
    CUDA_CHECK(cudaGetLastError());
}

void linear_generic_dense_gemm_launch(const Tensor& x, const Tensor& weight, Tensor& out,
                                      cudaStream_t stream) {
    const std::int32_t n   = out.ne[0];
    const std::int32_t k   = x.ne[0];
    const std::int32_t t   = x.ne[1];
    const bool weight_fp32 = is_fp32_weight(weight);
    if (is_small_dense_prefill(n, t)) {
        linear_generic_dense_small_gemv_kernel<<<small_gemv_grid_for(n, t), kDenseGemvThreads, 0,
                                                 stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), weight.data,
            static_cast<__nv_bfloat16*>(out.data), n, k, t, weight_fp32);
    } else {
        const dim3 block(static_cast<unsigned>(kDenseGemmBlockRows),
                         static_cast<unsigned>(kDenseGemmBlockCols));
        linear_generic_dense_gemm_kernel<<<tiled_gemm_grid_for(n, t), block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), weight.data,
            static_cast<__nv_bfloat16*>(out.data), n, k, t, weight_fp32);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::kernels::detail
