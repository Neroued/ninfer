#include "ops/common/host/launch.h"
#include "ninfer/ops/rmsnorm_pack_tail.h"
#include "core/device.h"
#include "ops/common/host/operands.h"
#include "ops/common/cuda/reduce.cuh"
#include <cmath>
#include <cuda_bf16.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops {
namespace {
template <int Hidden, int kBlock>
__device__ __forceinline__ float rmsnorm_pack_tail_inverse(float local_sum, float* warp_sums,
                                                           float* inverse, float epsilon) {
    constexpr int kWarps = kBlock / common::kWarpSize;
    const int lane       = static_cast<int>(threadIdx.x) & (common::kWarpSize - 1);
    const int warp       = static_cast<int>(threadIdx.x) / common::kWarpSize;

    local_sum = common::warp_reduce_sum(local_sum);
    if (lane == 0) { warp_sums[warp] = local_sum; }
    __syncthreads();
    if (warp == 0) {
        float block_sum = lane < kWarps ? warp_sums[lane] : 0.0F;
        block_sum       = common::warp_reduce_sum(block_sum);
        if (lane == 0) { *inverse = rsqrtf(block_sum * (1.0F / Hidden) + epsilon); }
    }
    __syncthreads();
    return *inverse;
}

// Grid x selects a mask column, grid y its request. Anchor columns are never read.
template <int Hidden, int kBlock>
__global__ __launch_bounds__(kBlock) void rmsnorm_pack_tail_kernel(
    const __nv_bfloat162* __restrict__ input, const __nv_bfloat162* __restrict__ weight,
    __nv_bfloat162* __restrict__ output, int width, float epsilon) {
    constexpr int kPairsPerThread = (Hidden / 2) / kBlock;
    static_assert((Hidden / 2) % kBlock == 0);

    const int tail  = static_cast<int>(blockIdx.x);
    const int batch = static_cast<int>(blockIdx.y);
    const std::int64_t input_base =
        (static_cast<std::int64_t>(batch) * width + tail + 1) * (Hidden / 2);
    const std::int64_t output_base =
        (static_cast<std::int64_t>(batch) * gridDim.x + tail) * (Hidden / 2);

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
    const float inv =
        rmsnorm_pack_tail_inverse<Hidden, kBlock>(local_sum, warp_sums, &inverse, epsilon);

#pragma unroll
    for (int item = 0; item < kPairsPerThread; ++item) {
        const int pair             = static_cast<int>(threadIdx.x) + item * kBlock;
        const float2 x             = __bfloat1622float2(values[item]);
        const float2 w             = __bfloat1622float2(weights[item]);
        output[output_base + pair] = __floats2bfloat162_rn(x.x * inv * w.x, x.y * inv * w.y);
    }
}

constexpr int kPackTailHidden  = 2304;
constexpr int kPackTailVectors = kPackTailHidden / 8;

__device__ __forceinline__ float2 unpack_bf16_pair(unsigned packed) {
    return {__uint_as_float(packed << 16), __uint_as_float(packed & 0xffff0000U)};
}

__device__ __forceinline__ float accumulate_pair_squares(float sum, unsigned packed) {
    const float2 x = unpack_bf16_pair(packed);
    return sum + x.x * x.x + x.y * x.y;
}

__device__ __forceinline__ unsigned scale_bf16_pair(unsigned input, unsigned weight, float inv) {
    const float2 x                  = unpack_bf16_pair(input);
    const float2 w                  = unpack_bf16_pair(weight);
    const __nv_bfloat162_raw result = __floats2bfloat162_rn(x.x * w.x * inv, x.y * w.y * inv);
    return __builtin_bit_cast(unsigned, result);
}

// One column per CTA and one aligned 128-bit vector per thread.
__global__ __launch_bounds__(kPackTailVectors) void rmsnorm_pack_tail_2304_kernel(
    const uint4* __restrict__ input, const uint4* __restrict__ weight, uint4* __restrict__ output,
    int width, float epsilon) {
    constexpr int kWarps           = kPackTailVectors / common::kWarpSize;
    constexpr int kPartialSumWidth = 16;
    const int thread               = static_cast<int>(threadIdx.x);
    const int lane                 = thread % common::kWarpSize;
    const int warp                 = thread / common::kWarpSize;
    const auto batch               = static_cast<std::int64_t>(blockIdx.y);
    const auto input_column        = batch * width + blockIdx.x + 1;
    const auto output_column       = batch * gridDim.x + blockIdx.x;

    const uint4 x = input[input_column * kPackTailVectors + thread];
    const uint4 w = weight[thread];
    float sum     = accumulate_pair_squares(0.0F, x.x);
    sum           = accumulate_pair_squares(sum, x.y);
    sum           = accumulate_pair_squares(sum, x.z);
    sum           = accumulate_pair_squares(sum, x.w);

    __shared__ float warp_sums[kWarps];
    sum = common::warp_reduce_sum(sum);
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    // Each half warp reduces the same nine partial sums, so one block barrier suffices.
    const int partial_lane = lane % kPartialSumWidth;
    const float partial    = partial_lane < kWarps ? warp_sums[partial_lane] : 0.0F;
    const float inv =
        rsqrtf(common::warp_sum<kPartialSumWidth>(partial) * (1.0F / kPackTailHidden) + epsilon);
    output[output_column * kPackTailVectors + thread] = {
        scale_bf16_pair(x.x, w.x, inv), scale_bf16_pair(x.y, w.y, inv),
        scale_bf16_pair(x.z, w.z, inv), scale_bf16_pair(x.w, w.w, inv)};
}

template <int Hidden, int Threads>
void launch_rmsnorm_pack_tail(const Tensor& input, const Tensor& weight, float epsilon,
                              Tensor& output, cudaStream_t stream) {
    using Packed = std::conditional_t<Hidden == kPackTailHidden, uint4, __nv_bfloat162>;
    constexpr int kItemsPerColumn = Hidden * sizeof(__nv_bfloat16) / sizeof(Packed);
    const int columns             = output.ne[1] / input.ne[2];
    common::for_each_token_slice(input.ne[2], 1, [&](int offset, int count) {
        const auto input_offset  = std::int64_t(offset) * input.ne[1] * kItemsPerColumn;
        const auto output_offset = std::int64_t(offset) * columns * kItemsPerColumn;
        const dim3 grid(columns, count);
        const auto* source = static_cast<const Packed*>(input.data) + input_offset;
        const auto* gain   = static_cast<const Packed*>(weight.data);
        auto* destination  = static_cast<Packed*>(output.data) + output_offset;
        if constexpr (Hidden == kPackTailHidden) {
            rmsnorm_pack_tail_2304_kernel<<<grid, Threads, 0, stream>>>(source, gain, destination,
                                                                        input.ne[1], epsilon);
        } else {
            rmsnorm_pack_tail_kernel<Hidden, Threads>
                <<<grid, Threads, 0, stream>>>(source, gain, destination, input.ne[1], epsilon);
        }
    });
}

void require_bf16_contiguous_aligned(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::BF16 || !tensor.is_contiguous() || tensor.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(tensor.data) & 15U) != 0) {
        throw std::invalid_argument(std::string("rmsnorm_pack_tail: ") + name +
                                    " must be contiguous 16-byte-aligned BF16");
    }
}

} // namespace

void rmsnorm_pack_tail(const Tensor& input, const Tensor& weight, float epsilon, Tensor& output,
                       CudaExecutionView execution) {
    if (input.device != execution.device() || weight.device != execution.device() ||
        output.device != execution.device()) {
        throw std::invalid_argument(
            "rmsnorm_pack_tail: operands must belong to the execution device");
    }
    if (!(epsilon > 0) || !std::isfinite(epsilon))
        throw std::invalid_argument("rmsnorm_pack_tail: epsilon must be positive and finite");
    const auto stream = execution.stream();
    require_bf16_contiguous_aligned(input, "input");
    require_bf16_contiguous_aligned(weight, "weight");
    require_bf16_contiguous_aligned(output, "output");

    const std::int32_t batch = input.ne[2];
    const std::int32_t width = input.ne[1];
    const int hidden         = input.ne[0];
    if ((hidden != kPackTailHidden && hidden != 4096 && hidden != 5120) ||
        (width < 2 || width > 16) || batch < 1 || input.ne[3] != 1) {
        throw std::invalid_argument(
            "rmsnorm_pack_tail: invalid [H,W,B] geometry or flattened output extent");
    }
    if (weight.ne[0] != hidden || weight.ne[1] != 1 || weight.ne[2] != 1 || weight.ne[3] != 1) {
        throw std::invalid_argument("rmsnorm_pack_tail: weight must have shape [H]");
    }
    if (output.ne[0] != hidden || output.ne[1] < batch || output.ne[1] % batch ||
        output.ne[1] / batch >= width || output.ne[2] != 1 || output.ne[3] != 1) {
        throw std::invalid_argument(
            "rmsnorm_pack_tail: output must have shape [H,g*B] with 1 <= g < W");
    }
    const common::operand::Range ranges[]{
        common::operand::range(input.data, input.bytes(), "rmsnorm_pack_tail"),
        common::operand::range(weight.data, weight.bytes(), "rmsnorm_pack_tail"),
        common::operand::range(output.data, output.bytes(), "rmsnorm_pack_tail"),
    };
    common::operand::disjoint(ranges, "rmsnorm_pack_tail");

    if (hidden == kPackTailHidden) {
        launch_rmsnorm_pack_tail<kPackTailHidden, kPackTailVectors>(input, weight, epsilon, output,
                                                                    stream);
    } else if (hidden == 4096) {
        if (output.ne[1] <= 32)
            launch_rmsnorm_pack_tail<4096, 128>(input, weight, epsilon, output, stream);
        else
            launch_rmsnorm_pack_tail<4096, 256>(input, weight, epsilon, output, stream);
    } else {
        launch_rmsnorm_pack_tail<5120, 512>(input, weight, epsilon, output, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
