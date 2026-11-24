#include "ninfer/ops/add_bias.h"
#include "core/device.h"
#include "ops/common/host/operands.h"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/numeric/math.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {
inline constexpr int kAddBiasPairsPerThread = 4;

__device__ __forceinline__ __nv_bfloat162 add_bias_pair(__nv_bfloat162 value, __nv_bfloat162 bias) {
    return __floats2bfloat162_rn(__low2float(value) + __low2float(bias),
                                 __high2float(value) + __high2float(bias));
}

template <int Block, int RowsPerBlock>
__launch_bounds__(Block) __global__
    void add_bias_bf16x8_kernel(const common::Bf16x8Pack* bias, common::Bf16x8Pack* x,
                                std::int32_t packs, std::int32_t rows) {
    const int pack = static_cast<int>(blockIdx.x) * Block + static_cast<int>(threadIdx.x);
    if (pack >= packs) { return; }
    const common::Bf16x8Pack bv = common::load_vec<common::Bf16x8Pack>(bias + pack);
    const int first_row         = static_cast<int>(blockIdx.y) * RowsPerBlock;
#pragma unroll
    for (int item = 0; item < RowsPerBlock; ++item) {
        const int row = first_row + item;
        if (row >= rows) { return; }
        const std::int64_t offset = static_cast<std::int64_t>(row) * packs + pack;
        common::Bf16x8Pack value  = common::load_vec<common::Bf16x8Pack>(x + offset);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            value.pair[pair] = add_bias_pair(value.pair[pair], bv.pair[pair]);
        }
        common::store_vec(x + offset, value);
    }
}

template <int Block>
__launch_bounds__(Block) __global__
    void add_bias_bf16x2_kernel(const __nv_bfloat162* bias, __nv_bfloat162* x, std::int32_t pairs,
                                std::int32_t rows) {
    const std::int32_t first =
        (static_cast<std::int32_t>(blockIdx.x) * Block + static_cast<std::int32_t>(threadIdx.x)) *
        kAddBiasPairsPerThread;
    for (std::int32_t row = static_cast<std::int32_t>(blockIdx.y); row < rows;
         row += static_cast<std::int32_t>(gridDim.y)) {
        const std::int64_t base = static_cast<std::int64_t>(row) * pairs;
#pragma unroll
        for (int item = 0; item < kAddBiasPairsPerThread; ++item) {
            const std::int32_t pair = first + item;
            if (pair < pairs) { x[base + pair] = add_bias_pair(x[base + pair], bias[pair]); }
        }
    }
}

__global__ void add_bias_kernel(const __nv_bfloat16* bias, __nv_bfloat16* x, std::int32_t d,
                                std::int64_t n) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const float value = __bfloat162float(x[i]) + __bfloat162float(bias[i % d]);
        x[i]              = __float2bfloat16_rn(value);
    }
}

using AddBiasLaunch = void (*)(const Tensor&, Tensor&, cudaStream_t);

template <int RowsPerBlock>
void launch_add_bias_vector(const Tensor& bias, Tensor& x, cudaStream_t stream) {
    constexpr int block = 256;
    const auto rows     = x.numel() / x.ne[0];
    const int packs     = x.ne[0] / 8;
    const dim3 grid(common::div_up(packs, block),
                    static_cast<unsigned>(common::div_up(rows, std::int64_t{RowsPerBlock})));
    add_bias_bf16x8_kernel<block, RowsPerBlock><<<grid, block, 0, stream>>>(
        static_cast<const common::Bf16x8Pack*>(bias.data), static_cast<common::Bf16x8Pack*>(x.data),
        packs, static_cast<std::int32_t>(rows));
}

void launch_add_bias_pair(const Tensor& bias, Tensor& x, cudaStream_t stream) {
    constexpr int block = 256;
    const auto rows     = x.numel() / x.ne[0];
    const int pairs     = x.ne[0] / 2;
    const dim3 grid(common::div_up(pairs, block * kAddBiasPairsPerThread),
                    static_cast<unsigned>(std::min<std::int64_t>(rows, 65535)));
    add_bias_bf16x2_kernel<block><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat162*>(bias.data), static_cast<__nv_bfloat162*>(x.data), pairs,
        static_cast<std::int32_t>(rows));
}

void launch_add_bias_scalar(const Tensor& bias, Tensor& x, cudaStream_t stream) {
    constexpr int block = 256;
    const auto count    = x.numel();
    const int grid      = static_cast<int>(std::min<std::int64_t>(
        common::div_up(count, std::int64_t{block}), std::numeric_limits<int>::max()));
    add_bias_kernel<<<grid, block, 0, stream>>>(static_cast<const __nv_bfloat16*>(bias.data),
                                                static_cast<__nv_bfloat16*>(x.data), x.ne[0],
                                                count);
}

AddBiasLaunch select_add_bias(const Tensor& bias, const Tensor& x) {
    constexpr std::int64_t kVectorMaxElements = 32LL * 1024 * 1024;
    constexpr std::int64_t kVectorMaxRows     = 4LL * 65535;
    const auto count                          = x.numel();
    const auto rows                           = count / x.ne[0];
    if (x.ne[0] % 8 == 0 && common::operand::aligned(bias.data, 16) &&
        common::operand::aligned(x.data, 16) && count <= kVectorMaxElements &&
        rows <= kVectorMaxRows) {
        return rows >= 1024 ? launch_add_bias_vector<4> : launch_add_bias_vector<1>;
    }
    if (x.ne[0] % 2 == 0 && common::operand::aligned(bias.data, 4) &&
        common::operand::aligned(x.data, 4) && rows <= std::numeric_limits<std::int32_t>::max())
        return launch_add_bias_pair;
    return launch_add_bias_scalar;
}

} // namespace

void add_bias(const Tensor& bias, Tensor& x, CudaExecutionView execution) {
    const auto stream = execution.stream();
    if (bias.dtype != DType::BF16 || x.dtype != DType::BF16) {
        throw std::invalid_argument("add_bias: bias/x must be BF16");
    }
    const std::int64_t n = common::operand::numel(x, "add_bias");
    (void)common::operand::numel(bias, "add_bias");
    if (bias.ne[0] != x.ne[0] || bias.ne[1] != 1 || bias.ne[2] != 1 || bias.ne[3] != 1) {
        throw std::invalid_argument("add_bias: bias must have shape [x.ne[0]]");
    }
    if (n == 0) { return; }
    if (bias.device != execution.device() || x.device != execution.device()) {
        throw std::invalid_argument("add_bias: operands must belong to the execution device");
    }
    if (!bias.is_contiguous() || !x.is_contiguous()) {
        throw std::invalid_argument("add_bias: bias/x must be contiguous");
    }
    if (bias.data == nullptr || x.data == nullptr) {
        throw std::invalid_argument("add_bias: bias/x data must be non-null");
    }
    select_add_bias(bias, x)(bias, x, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
