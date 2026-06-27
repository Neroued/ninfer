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

int checked_grid(std::int64_t blocks) {
    if (blocks <= 0) { return 1; }
    if (blocks > std::numeric_limits<int>::max()) {
        throw std::overflow_error("linear: dense launch grid exceeds CUDA limit");
    }
    return static_cast<int>(blocks);
}

int gemm_grid_for(std::int32_t n, std::int32_t t) {
    const std::int64_t tiles_n = (static_cast<std::int64_t>(n) + kDenseMmaM - 1) / kDenseMmaM;
    const std::int64_t tiles_t = (static_cast<std::int64_t>(t) + kDenseMmaN - 1) / kDenseMmaN;
    const std::int64_t tiles   = tiles_n * tiles_t;
    const std::int64_t grid    = (tiles + kDenseGemmWarps - 1) / kDenseGemmWarps;
    if (grid > std::numeric_limits<int>::max()) {
        throw std::overflow_error("linear: dense launch grid exceeds CUDA limit");
    }
    return static_cast<int>(std::max<std::int64_t>(1, grid));
}

dim3 tiled_gemm_grid_for(std::int32_t n, std::int32_t t) {
    const std::int64_t grid_m =
        (static_cast<std::int64_t>(n) + kDenseGemmBlockM - 1) / kDenseGemmBlockM;
    const std::int64_t grid_n =
        (static_cast<std::int64_t>(t) + kDenseGemmBlockN - 1) / kDenseGemmBlockN;
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

bool is_full_bf16_tiled_gemm(std::int32_t n, std::int32_t k, std::int32_t t, bool weight_fp32) {
    return !weight_fp32 && n >= kDenseGemmBlockM && t >= kDenseGemmBlockN &&
           (n % kDenseGemmBlockM) == 0 && (t % kDenseGemmBlockN) == 0 &&
           (k % kDenseGemmBlockK) == 0;
}

bool is_full_bf16_half_wide_gemm(std::int32_t n, std::int32_t k, std::int32_t t, bool weight_fp32) {
    return !weight_fp32 && n >= kDenseGemmBlockM && t >= kDenseGemmWideBlockN &&
           (n % kDenseGemmBlockM) == 0 && (t % kDenseGemmWideBlockN) == 0 &&
           (k % kDenseGemmHalfBlockK) == 0;
}

dim3 tiled_gemm_half_wide_grid_for(std::int32_t n, std::int32_t t) {
    const std::int64_t grid_m =
        (static_cast<std::int64_t>(n) + kDenseGemmHalfBlockM - 1) / kDenseGemmHalfBlockM;
    const std::int64_t grid_n =
        (static_cast<std::int64_t>(t) + kDenseGemmWideBlockN - 1) / kDenseGemmWideBlockN;
    if (grid_m > std::numeric_limits<unsigned>::max() ||
        grid_n > std::numeric_limits<unsigned>::max()) {
        throw std::overflow_error("linear: dense launch grid exceeds CUDA limit");
    }
    return dim3(static_cast<unsigned>(std::max<std::int64_t>(1, grid_m)),
                static_cast<unsigned>(std::max<std::int64_t>(1, grid_n)));
}

} // namespace

void linear_dense_gemv_launch(const Tensor& x, const Tensor& weight, Tensor& out,
                              cudaStream_t stream) {
    const std::int32_t n = out.ne[0];
    const std::int32_t k = x.ne[0];
    linear_dense_gemv_kernel<<<checked_grid(n), kDenseGemvThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), weight.data,
        static_cast<__nv_bfloat16*>(out.data), n, k, is_fp32_weight(weight));
    CUDA_CHECK(cudaGetLastError());
}

void linear_dense_gemm_launch(const Tensor& x, const Tensor& weight, Tensor& out,
                              cudaStream_t stream) {
    const std::int32_t n   = out.ne[0];
    const std::int32_t k   = x.ne[0];
    const std::int32_t t   = x.ne[1];
    const bool weight_fp32 = is_fp32_weight(weight);
    if (is_small_dense_prefill(n, t)) {
        linear_dense_small_gemv_kernel<<<small_gemv_grid_for(n, t), kDenseGemvThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), weight.data,
            static_cast<__nv_bfloat16*>(out.data), n, k, t, weight_fp32);
    } else if (weight_fp32) {
        linear_dense_gemm_tf32_kernel<<<tiled_gemm_grid_for(n, t), kDenseGemmThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const float*>(weight.data),
            static_cast<__nv_bfloat16*>(out.data), n, k, t);
    } else if (is_full_bf16_half_wide_gemm(n, k, t, weight_fp32)) {
        linear_dense_gemm_bf16_half_wide_full_kernel<<<tiled_gemm_half_wide_grid_for(n, t),
                                                       kDenseGemmHalfThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data), static_cast<__nv_bfloat16*>(out.data),
            n, k);
    } else if (is_full_bf16_tiled_gemm(n, k, t, weight_fp32)) {
        linear_dense_gemm_bf16_full_kernel<<<tiled_gemm_grid_for(n, t), kDenseGemmThreads, 0,
                                             stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.data), static_cast<__nv_bfloat16*>(out.data),
            n, k);
    } else if (n >= kDenseGemmBlockM && t >= kDenseGemmBlockN) {
        linear_dense_gemm_kernel<<<tiled_gemm_grid_for(n, t), kDenseGemmThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), weight.data,
            static_cast<__nv_bfloat16*>(out.data), n, k, t, weight_fp32);
    } else {
        linear_dense_gemm_warp_kernel<<<gemm_grid_for(n, t), kDenseGemmThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), weight.data,
            static_cast<__nv_bfloat16*>(out.data), n, k, t, weight_fp32);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
