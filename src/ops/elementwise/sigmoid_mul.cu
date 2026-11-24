#include "ninfer/ops/sigmoid_mul.h"
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

inline constexpr int kSigmoidGateMulPairsPerThread = 4;

__device__ __forceinline__ __nv_bfloat162 sigmoid_mul_pair(__nv_bfloat162 gate, __nv_bfloat162 x) {
    const float r0 = __low2float(x) * common::sigmoid(__low2float(gate));
    const float r1 = __high2float(x) * common::sigmoid(__high2float(gate));
    return __floats2bfloat162_rn(r0, r1);
}

__global__ void sigmoid_mul_scalar_kernel(const __nv_bfloat16* gate, __nv_bfloat16* x,
                                          std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        x[i] = __float2bfloat16_rn(__bfloat162float(x[i]) *
                                   common::sigmoid(__bfloat162float(gate[i])));
    }
}

__launch_bounds__(256) __global__
    void sigmoid_mul_bf16x8_kernel(const common::Bf16x8Pack* gate, common::Bf16x8Pack* x,
                                   std::int64_t packs) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < packs; i += stride) {
        const common::Bf16x8Pack gv = common::load_vec<common::Bf16x8Pack>(gate + i);
        common::Bf16x8Pack xv       = common::load_vec<common::Bf16x8Pack>(x + i);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            xv.pair[pair] = sigmoid_mul_pair(gv.pair[pair], xv.pair[pair]);
        }
        common::store_vec(x + i, xv);
    }
}

__launch_bounds__(256) __global__
    void sigmoid_mul_bf16x2_kernel(const __nv_bfloat16* gate, __nv_bfloat16* x, std::int64_t n) {
    const std::int64_t tid = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride =
        static_cast<std::int64_t>(gridDim.x) * blockDim.x * kSigmoidGateMulPairsPerThread;
    const std::int64_t n2 = n / 2;

    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate);
    auto* x2          = reinterpret_cast<__nv_bfloat162*>(x);
    for (std::int64_t j = tid * kSigmoidGateMulPairsPerThread; j < n2; j += stride) {
        const __nv_bfloat162 g0 = gate2[j];
        const __nv_bfloat162 x0 = x2[j];
        if (j + 3 < n2) {
            const __nv_bfloat162 g1  = gate2[j + 1];
            const __nv_bfloat162 x1  = x2[j + 1];
            const __nv_bfloat162 g2  = gate2[j + 2];
            const __nv_bfloat162 x2v = x2[j + 2];
            const __nv_bfloat162 g3  = gate2[j + 3];
            const __nv_bfloat162 x3  = x2[j + 3];
            x2[j]                    = sigmoid_mul_pair(g0, x0);
            x2[j + 1]                = sigmoid_mul_pair(g1, x1);
            x2[j + 2]                = sigmoid_mul_pair(g2, x2v);
            x2[j + 3]                = sigmoid_mul_pair(g3, x3);
        } else {
            x2[j] = sigmoid_mul_pair(g0, x0);
            if (j + 1 < n2) { x2[j + 1] = sigmoid_mul_pair(gate2[j + 1], x2[j + 1]); }
            if (j + 2 < n2) { x2[j + 2] = sigmoid_mul_pair(gate2[j + 2], x2[j + 2]); }
        }
    }

    if (tid == 0 && (n & 1) != 0) {
        const std::int64_t i = n - 1;
        x[i]                 = __float2bfloat16_rn(__bfloat162float(x[i]) *
                                                   common::sigmoid(__bfloat162float(gate[i])));
    }
}

void launch_sigmoid_mul(const Tensor& gate, Tensor& x, cudaStream_t stream) {
    const std::int64_t n   = x.numel();
    constexpr int kBlock   = 256;
    constexpr int kMaxGrid = 4096;
    const auto gate_addr   = reinterpret_cast<std::uintptr_t>(gate.data);
    const auto x_addr      = reinterpret_cast<std::uintptr_t>(x.data);
    if (((gate_addr | x_addr) & (alignof(common::Bf16x8Pack) - 1)) == 0 && (n % 8) == 0) {
        const std::int64_t packs = n / 8;
        const int grid           = static_cast<int>(std::min<std::int64_t>(
            kMaxGrid,
            std::max<std::int64_t>(1, common::div_up(packs, static_cast<std::int64_t>(kBlock)))));
        sigmoid_mul_bf16x8_kernel<<<grid, kBlock, 0, stream>>>(
            static_cast<const common::Bf16x8Pack*>(gate.data),
            static_cast<common::Bf16x8Pack*>(x.data), packs);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    if (((gate_addr | x_addr) & (alignof(__nv_bfloat162) - 1)) != 0) {
        const int scalar_grid = static_cast<int>(
            std::min<std::int64_t>(kMaxGrid, common::div_up(n, static_cast<std::int64_t>(kBlock))));
        sigmoid_mul_scalar_kernel<<<scalar_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(x.data), n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const std::int64_t n2                 = n / 2;
    constexpr std::int64_t kPairsPerBlock = kBlock * kSigmoidGateMulPairsPerThread;
    const int grid                        = static_cast<int>(std::min<std::int64_t>(
        kMaxGrid, std::max<std::int64_t>(1, common::div_up(n2, kPairsPerBlock))));

    sigmoid_mul_bf16x2_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(x.data), n);
    CUDA_CHECK(cudaGetLastError());
}

std::int64_t numel_allow_zero(const Tensor& t) {
    std::int64_t total = 1;
    for (int d = 0; d < 4; ++d) {
        if (t.ne[d] < 0) {
            throw std::invalid_argument("sigmoid_mul: gate/x dimensions must be nonnegative");
        }
        if (t.ne[d] == 0) { return 0; }
        if (total > std::numeric_limits<std::int64_t>::max() / t.ne[d]) {
            throw std::overflow_error("sigmoid_mul: tensor size overflows int64");
        }
        total *= t.ne[d];
    }
    return total;
}

} // namespace

void sigmoid_mul(const Tensor& gate, Tensor& x, CudaExecutionView execution) {
    const auto stream = execution.stream();
    if (gate.dtype != DType::BF16 || x.dtype != DType::BF16) {
        throw std::invalid_argument("sigmoid_mul: gate/x must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (gate.ne[d] != x.ne[d]) {
            throw std::invalid_argument("sigmoid_mul: gate/x shapes must match");
        }
    }
    if (numel_allow_zero(x) == 0) { return; }
    if (gate.device != execution.device() || x.device != execution.device()) {
        throw std::invalid_argument("sigmoid_mul: operands must belong to the execution device");
    }
    if (!gate.is_contiguous() || !x.is_contiguous()) {
        throw std::invalid_argument("sigmoid_mul: gate/x must be contiguous");
    }
    if (gate.data == nullptr || x.data == nullptr) {
        throw std::invalid_argument("sigmoid_mul: gate/x data must be non-null");
    }

    launch_sigmoid_mul(gate, x, stream);
}

} // namespace ninfer::ops
