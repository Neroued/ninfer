#include "ninfer/ops/residual_add.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/numeric/math.h"

#include "core/device.h"
#include "core/platform.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

inline constexpr int kResidualAddPairsPerThread = 4;

__device__ __forceinline__ __nv_bfloat162 residual_add_pair(__nv_bfloat162 y, __nv_bfloat162 x) {
    const float r0 = __low2float(x) + __low2float(y);
    const float r1 = __high2float(x) + __high2float(y);
    return __floats2bfloat162_rn(r0, r1);
}

__global__ void residual_add_scalar_kernel(const __nv_bfloat16* y, __nv_bfloat16* x,
                                           std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) + __bfloat162float(y[i]));
    }
}

__launch_bounds__(256) __global__
    void residual_add_bf16x8_kernel(const common::Bf16x8Pack* y, common::Bf16x8Pack* x,
                                    std::int64_t packs) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        const common::Bf16x8Pack yv = common::load_vec<common::Bf16x8Pack>(y + i);
        common::Bf16x8Pack xv       = common::load_vec<common::Bf16x8Pack>(x + i);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            xv.pair[pair] = residual_add_pair(yv.pair[pair], xv.pair[pair]);
        }
        common::store_vec(x + i, xv);
    }
}

__launch_bounds__(256) __global__
    void residual_add_bf16x2_kernel(const __nv_bfloat16* y, __nv_bfloat16* x, std::int64_t n) {
    const std::int64_t tid = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride =
        static_cast<std::int64_t>(gridDim.x) * blockDim.x * kResidualAddPairsPerThread;
    const std::int64_t n2 = n / 2;

    const auto* y2 = reinterpret_cast<const __nv_bfloat162*>(y);
    auto* x2       = reinterpret_cast<__nv_bfloat162*>(x);
    for (std::int64_t j = tid * kResidualAddPairsPerThread; j < n2; j += stride) {
        const __nv_bfloat162 y0 = y2[j];
        const __nv_bfloat162 x0 = x2[j];
        if (j + 3 < n2) {
            const __nv_bfloat162 y1  = y2[j + 1];
            const __nv_bfloat162 x1  = x2[j + 1];
            const __nv_bfloat162 y2v = y2[j + 2];
            const __nv_bfloat162 x2v = x2[j + 2];
            const __nv_bfloat162 y3  = y2[j + 3];
            const __nv_bfloat162 x3  = x2[j + 3];
            x2[j]                    = residual_add_pair(y0, x0);
            x2[j + 1]                = residual_add_pair(y1, x1);
            x2[j + 2]                = residual_add_pair(y2v, x2v);
            x2[j + 3]                = residual_add_pair(y3, x3);
        } else {
            x2[j] = residual_add_pair(y0, x0);
            if (j + 1 < n2) { x2[j + 1] = residual_add_pair(y2[j + 1], x2[j + 1]); }
            if (j + 2 < n2) { x2[j + 2] = residual_add_pair(y2[j + 2], x2[j + 2]); }
        }
    }

    if (tid == 0 && (n & 1) != 0) {
        const std::int64_t i = n - 1;
        x[i]                 = __float2bfloat16_rn(__bfloat162float(x[i]) + __bfloat162float(y[i]));
    }
}

void launch_residual_add(const Tensor& y, Tensor& x, cudaStream_t stream) {
    const std::int64_t n   = x.numel();
    constexpr int kBlock   = 256;
    constexpr int kMaxGrid = 4096;
    const auto y_addr      = reinterpret_cast<std::uintptr_t>(y.data);
    const auto x_addr      = reinterpret_cast<std::uintptr_t>(x.data);
    if (((y_addr | x_addr) & (alignof(common::Bf16x8Pack) - 1)) == 0 && (n % 8) == 0) {
        constexpr int max_grid   = NINFER_TARGET_SPARK ? 16384 : kMaxGrid;
        const std::int64_t packs = n / 8;
        const int grid           = static_cast<int>(std::min<std::int64_t>(
            max_grid,
            std::max<std::int64_t>(1, common::div_up(packs, static_cast<std::int64_t>(kBlock)))));
        residual_add_bf16x8_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const common::Bf16x8Pack*>(y.data),
            static_cast<common::Bf16x8Pack*>(x.data), packs);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (((y_addr | x_addr) & (alignof(__nv_bfloat162) - 1)) != 0) {
        const int scalar_grid = static_cast<int>(
            std::min<std::int64_t>(kMaxGrid, common::div_up(n, static_cast<std::int64_t>(kBlock))));
        residual_add_scalar_kernel<<<scalar_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(y.data), static_cast<__nv_bfloat16*>(x.data), n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const std::int64_t n2                 = n / 2;
    constexpr std::int64_t kPairsPerBlock = kBlock * kResidualAddPairsPerThread;
    const int grid                        = static_cast<int>(std::min<std::int64_t>(
        kMaxGrid, std::max<std::int64_t>(1, common::div_up(n2, kPairsPerBlock))));

    residual_add_bf16x2_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(y.data), static_cast<__nv_bfloat16*>(x.data), n);
    CUDA_CHECK(cudaGetLastError());
}

std::int64_t numel_allow_zero(const Tensor& t) {
    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument("residual_add: y/x dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { return 0; }
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("residual_add: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

} // namespace

void residual_add(const Tensor& y, Tensor& x, CudaExecutionView execution) {
    const auto stream = execution.stream();
    if (y.dtype != DType::BF16 || x.dtype != DType::BF16) {
        throw std::invalid_argument("residual_add: y/x must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (y.ne[d] != x.ne[d]) {
            throw std::invalid_argument("residual_add: y/x shapes must match");
        }
    }
    if (numel_allow_zero(x) == 0) { return; }
    if (y.device != execution.device() || x.device != execution.device()) {
        throw std::invalid_argument("residual_add: operands must belong to the execution device");
    }
    if (!y.is_contiguous() || !x.is_contiguous()) {
        throw std::invalid_argument("residual_add: y/x must be contiguous");
    }
    if (y.data == nullptr || x.data == nullptr) {
        throw std::invalid_argument("residual_add: y/x data must be non-null");
    }

    launch_residual_add(y, x, stream);
}

} // namespace ninfer::ops
