#include "ninfer/ops/rmsnorm_pack_tail.h"
#include "core/device.h"
#include "ops/common/host/operands.h"
#include "ops/common/cuda/reduce.cuh"
#include <cuda_bf16.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {
inline constexpr int kRmsnormPackTailRows        = 5120;
inline constexpr int kRmsnormPackTailPairsPerRow = kRmsnormPackTailRows / 2;
inline constexpr float kRmsnormPackTailEps       = 1.0e-6F;

template <int kBlock>
__device__ __forceinline__ float rmsnorm_pack_tail_inverse(float local_sum, float* warp_sums,
                                                           float* inverse) {
    constexpr int kWarps = kBlock / common::kWarpSize;
    const int lane       = static_cast<int>(threadIdx.x) & (common::kWarpSize - 1);
    const int warp       = static_cast<int>(threadIdx.x) / common::kWarpSize;

    local_sum = common::warp_reduce_sum(local_sum);
    if (lane == 0) { warp_sums[warp] = local_sum; }
    __syncthreads();
    if (warp == 0) {
        float block_sum = lane < kWarps ? warp_sums[lane] : 0.0F;
        block_sum       = common::warp_reduce_sum(block_sum);
        if (lane == 0) {
            *inverse = rsqrtf(block_sum * (1.0F / kRmsnormPackTailRows) + kRmsnormPackTailEps);
        }
    }
    __syncthreads();
    return *inverse;
}

// Grid x selects a mask column, grid y its request. Anchor columns are never read.
template <int kBlock>
__global__ __launch_bounds__(kBlock) void rmsnorm_pack_tail_kernel(
    const __nv_bfloat162* __restrict__ input, const __nv_bfloat162* __restrict__ weight,
    __nv_bfloat162* __restrict__ output, int width) {
    constexpr int kPairsPerThread = kRmsnormPackTailPairsPerRow / kBlock;
    static_assert(kRmsnormPackTailPairsPerRow % kBlock == 0);

    const int tail  = static_cast<int>(blockIdx.x);
    const int batch = static_cast<int>(blockIdx.y);
    const std::int64_t input_base =
        static_cast<std::int64_t>(batch * width + tail + 1) * kRmsnormPackTailPairsPerRow;
    const std::int64_t output_base =
        static_cast<std::int64_t>(batch * (width - 1) + tail) * kRmsnormPackTailPairsPerRow;

    __nv_bfloat162 values[kPairsPerThread];
    __nv_bfloat162 weights[kPairsPerThread];
    float local_sum = 0.0F;
#pragma unroll
    for (int item = 0; item < kPairsPerThread; ++item) {
        const int pair = static_cast<int>(threadIdx.x) + item * kBlock;
        values[item]   = input[input_base + pair];
        weights[item]  = weight[pair];
        const float2 x = __bfloat1622float2(values[item]);
        local_sum += x.x * x.x + x.y * x.y;
    }

    __shared__ float warp_sums[kBlock / common::kWarpSize];
    __shared__ float inverse;
    const float inv = rmsnorm_pack_tail_inverse<kBlock>(local_sum, warp_sums, &inverse);

#pragma unroll
    for (int item = 0; item < kPairsPerThread; ++item) {
        const int pair             = static_cast<int>(threadIdx.x) + item * kBlock;
        const float2 x             = __bfloat1622float2(values[item]);
        const float2 w             = __bfloat1622float2(weights[item]);
        output[output_base + pair] = __floats2bfloat162_rn(x.x * inv * w.x, x.y * inv * w.y);
    }
}

void launch_rmsnorm_pack_tail(const Tensor& input, const Tensor& weight, Tensor& output,
                              cudaStream_t stream) {
    constexpr int threads = 512;
    const dim3 grid(input.ne[1] - 1, input.ne[2]);
    rmsnorm_pack_tail_kernel<threads>
        <<<grid, threads, 0, stream>>>(static_cast<const __nv_bfloat162*>(input.data),
                                       static_cast<const __nv_bfloat162*>(weight.data),
                                       static_cast<__nv_bfloat162*>(output.data), input.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void require_bf16_contiguous_aligned(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::BF16 || !tensor.is_contiguous() || tensor.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(tensor.data) & 15U) != 0) {
        throw std::invalid_argument(std::string("rmsnorm_pack_tail: ") + name +
                                    " must be contiguous 16-byte-aligned BF16");
    }
}

} // namespace

void rmsnorm_pack_tail(const Tensor& input, const Tensor& weight, Tensor& output,
                       CudaExecutionView execution) {
    if (input.device != execution.device() || weight.device != execution.device() ||
        output.device != execution.device()) {
        throw std::invalid_argument(
            "rmsnorm_pack_tail: operands must belong to the execution device");
    }
    const auto stream = execution.stream();
    require_bf16_contiguous_aligned(input, "input");
    require_bf16_contiguous_aligned(weight, "weight");
    require_bf16_contiguous_aligned(output, "output");

    const std::int32_t batch = input.ne[2];
    const std::int32_t width = input.ne[1];
    if (input.ne[0] != 5120 || (width < 2 || width > 16) || batch < 1 || batch > 8 ||
        input.ne[3] != 1) {
        throw std::invalid_argument(
            "rmsnorm_pack_tail: input must have shape [5120,W,B], W=2..16, B=1..8");
    }
    if (weight.ne[0] != 5120 || weight.ne[1] != 1 || weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument("rmsnorm_pack_tail: weight must have shape [5120]");
    }
    if (output.ne[0] != 5120 || output.ne[1] != (width - 1) * batch || output.ne[2] != 1 ||
        output.ne[3] != 1) {
        throw std::invalid_argument("rmsnorm_pack_tail: output must have shape [5120,(W-1)B]");
    }
    const common::operand::Range ranges[]{
        common::operand::range(input.data, input.bytes(), "rmsnorm_pack_tail"),
        common::operand::range(weight.data, weight.bytes(), "rmsnorm_pack_tail"),
        common::operand::range(output.data, output.bytes(), "rmsnorm_pack_tail"),
    };
    common::operand::disjoint(ranges, "rmsnorm_pack_tail");

    launch_rmsnorm_pack_tail(input, weight, output, stream);
}

} // namespace ninfer::ops
