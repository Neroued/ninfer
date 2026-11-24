#include "ninfer/ops/gelu.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/math.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

inline constexpr int kGeluPairsPerThread = 4;
inline constexpr std::int64_t kVectorMaxElements = 32LL * 1024LL * 1024LL;

template <bool TanhApprox>
__device__ __forceinline__ __nv_bfloat162 gelu_pair(__nv_bfloat162 input) {
    return __floats2bfloat162_rn(common::gelu<TanhApprox>(__low2float(input)),
                                 common::gelu<TanhApprox>(__high2float(input)));
}

template <bool TanhApprox, int Block>
__launch_bounds__(Block) __global__
    void gelu_bf16x8_kernel(common::Bf16x8Pack* x, std::int64_t packs) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        common::Bf16x8Pack value = common::load_vec<common::Bf16x8Pack>(x + i);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            value.pair[pair] = gelu_pair<TanhApprox>(value.pair[pair]);
        }
        common::store_vec(x + i, value);
    }
}

template <bool TanhApprox, int Block>
__launch_bounds__(Block) __global__
    void gelu_bf16x2_kernel(__nv_bfloat162* x, std::int64_t pairs, __nv_bfloat16* scalar,
                            bool has_scalar) {
    const std::int64_t first =
        (static_cast<std::int64_t>(blockIdx.x) * Block + threadIdx.x) * kGeluPairsPerThread;
#pragma unroll
    for (int item = 0; item < kGeluPairsPerThread; ++item) {
        const std::int64_t pair = first + item;
        if (pair < pairs) { x[pair] = gelu_pair<TanhApprox>(x[pair]); }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0 && has_scalar) {
        *scalar = __float2bfloat16_rn(common::gelu<TanhApprox>(__bfloat162float(*scalar)));
    }
}

template <bool TanhApprox>
__global__ void gelu_kernel(__nv_bfloat16* x, std::int64_t n) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        x[i] = __float2bfloat16_rn(common::gelu<TanhApprox>(__bfloat162float(x[i])));
    }
}

void launch_gelu(Tensor& x, GeluMode mode, cudaStream_t stream) {
    constexpr int block   = 256;
    constexpr int maxGrid = 16384;
    const std::int64_t n  = x.numel();
    const auto address    = reinterpret_cast<std::uintptr_t>(x.data);
    if ((address & (alignof(common::Bf16x8Pack) - 1)) == 0 && (n % 8) == 0 &&
        n <= kVectorMaxElements) {
        const std::int64_t packs = n / 8;
        const int grid           = static_cast<int>(std::min<std::int64_t>(
            maxGrid,
            std::max<std::int64_t>(1, common::div_up(packs, static_cast<std::int64_t>(block)))));
        if (mode == GeluMode::Tanh) {
            gelu_bf16x8_kernel<true, block>
                <<<grid, block, 0, stream>>>(static_cast<common::Bf16x8Pack*>(x.data), packs);
        } else {
            gelu_bf16x8_kernel<false, block>
                <<<grid, block, 0, stream>>>(static_cast<common::Bf16x8Pack*>(x.data), packs);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const bool paired = (address & 0x3u) == 0;
    if (paired && n >= 2) {
        const std::int64_t pairs = n / 2;
        const int grid           = static_cast<int>(std::min<std::int64_t>(
            common::div_up(pairs, static_cast<std::int64_t>(block * kGeluPairsPerThread)),
            std::numeric_limits<int>::max()));
        auto* x2                 = static_cast<__nv_bfloat162*>(x.data);
        auto* tail               = static_cast<__nv_bfloat16*>(x.data) + pairs * 2;
        if (mode == GeluMode::Tanh) {
            gelu_bf16x2_kernel<true, block>
                <<<grid, block, 0, stream>>>(x2, pairs, tail, (n & 1) != 0);
        } else {
            gelu_bf16x2_kernel<false, block>
                <<<grid, block, 0, stream>>>(x2, pairs, tail, (n & 1) != 0);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int grid = static_cast<int>(std::min<std::int64_t>(
        common::div_up(n, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    if (mode == GeluMode::Tanh) {
        gelu_kernel<true><<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16*>(x.data), n);
    } else {
        gelu_kernel<false><<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16*>(x.data), n);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void gelu(Tensor& x, GeluMode mode, CudaExecutionView execution) {
    const auto stream = execution.stream();
    if (x.dtype != DType::BF16) { throw std::invalid_argument("gelu: x must be BF16"); }
    std::int64_t n = 1;
    for (int i = 0; i < 4; ++i) {
        if (x.ne[i] < 0) { throw std::invalid_argument("gelu: negative dimension"); }
        if (x.ne[i] == 0) { return; }
        if (n > std::numeric_limits<std::int64_t>::max() / x.ne[i]) {
            throw std::overflow_error("gelu: tensor size overflows int64");
        }
        n *= x.ne[i];
    }
    if (x.device != execution.device()) {
        throw std::invalid_argument("gelu: operand belongs to another device");
    }
    if (!x.is_contiguous()) { throw std::invalid_argument("gelu: x must be contiguous"); }
    if (x.data == nullptr) { throw std::invalid_argument("gelu: x data must be non-null"); }
    launch_gelu(x, mode, stream);
}

} // namespace ninfer::ops
