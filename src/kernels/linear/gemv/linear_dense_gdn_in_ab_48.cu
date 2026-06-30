#include "kernels/linear/gemv/linear_dense_gdn_in_ab_48.cuh"

#include "qus/core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace qus::kernels::detail {
namespace {

constexpr int kN       = 48;
constexpr int kK       = 5120;
constexpr int kThreads = 256;

template <int Width = 32>
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset, Width);
    }
    return v;
}

template <int BlockSize>
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[BlockSize / 32];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) { warp_sums[warp] = v; }
    __syncthreads();

    v = (static_cast<int>(threadIdx.x) < (BlockSize / 32)) ? warp_sums[lane] : 0.0f;
    if (warp == 0) { v = warp_reduce_sum<BlockSize / 32>(v); }
    return v;
}

__device__ __forceinline__ float2 bf162_to_float2(__nv_bfloat162 v) {
    return __bfloat1622float2(v);
}

__global__ void linear_dense_gdn_in_ab_48_kernel(const __nv_bfloat16* x,
                                                 const __nv_bfloat16* a_weight,
                                                 const __nv_bfloat16* b_weight,
                                                 __nv_bfloat16* a_out,
                                                 __nv_bfloat16* b_out) {
    const int global_row = static_cast<int>(blockIdx.x);
    const bool is_b      = global_row >= kN;
    const int row        = is_b ? global_row - kN : global_row;
    const auto* weight   = is_b ? b_weight : a_weight;

    float acc                   = 0.0f;
    constexpr int kPairs        = kK / 2;
    const std::int64_t row_base = static_cast<std::int64_t>(row) * kK;
    const auto* x2              = reinterpret_cast<const __nv_bfloat162*>(x);
    const auto* w2 =
        reinterpret_cast<const __nv_bfloat162*>(weight + static_cast<std::int64_t>(row_base));
    for (int p = static_cast<int>(threadIdx.x); p < kPairs; p += static_cast<int>(blockDim.x)) {
        const float2 wf = bf162_to_float2(w2[p]);
        const float2 xf = bf162_to_float2(x2[p]);
        acc             = fmaf(wf.x, xf.x, acc);
        acc             = fmaf(wf.y, xf.y, acc);
    }

    acc = block_reduce_sum<kThreads>(acc);
    if (threadIdx.x == 0) {
        if (is_b) {
            b_out[row] = __float2bfloat16(acc);
        } else {
            a_out[row] = __float2bfloat16(acc);
        }
    }
}

void require_shape(const Weight& w, const char* name) {
    if (w.n != kN || w.k != kK || w.shape[0] != kN || w.shape[1] != kK) {
        throw std::invalid_argument(std::string("gdn_in_ab_decode: ") + name +
                                    " requires 48x5120 dense BF16");
    }
}

} // namespace

void linear_dense_gdn_in_ab_48_launch(const Tensor& x, const Weight& a_weight,
                                      const Weight& b_weight, Tensor& a_out, Tensor& b_out,
                                      cudaStream_t stream) {
    require_shape(a_weight, "a_weight");
    require_shape(b_weight, "b_weight");
    linear_dense_gdn_in_ab_48_kernel<<<2 * kN, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(a_weight.qdata),
        static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<__nv_bfloat16*>(a_out.data),
        static_cast<__nv_bfloat16*>(b_out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
