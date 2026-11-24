#include "ninfer/ops/silu_mul.h"
#include "ops/common/host/launch.h"
#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/math.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

inline constexpr int kSiluAndMulPairsPerThread = 4;

__device__ __forceinline__ __nv_bfloat162 silu_mul_pair(__nv_bfloat162 g, __nv_bfloat162 u) {
    const float r0 = common::silu(__low2float(g)) * __low2float(u);
    const float r1 = common::silu(__high2float(g)) * __high2float(u);
    return __floats2bfloat162_rn(r0, r1);
}

__global__ void silu_mul_scalar_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                       __nv_bfloat16* out, std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        out[i] =
            __float2bfloat16(common::silu(__bfloat162float(gate[i])) * __bfloat162float(up[i]));
    }
}

__global__ void silu_mul_strided_input_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                              __nv_bfloat16* out, std::int64_t n, std::int32_t ne0,
                                              std::int32_t ne1, std::int32_t ne2, std::int64_t gnb0,
                                              std::int64_t gnb1, std::int64_t gnb2,
                                              std::int64_t gnb3, std::int64_t unb0,
                                              std::int64_t unb1, std::int64_t unb2,
                                              std::int64_t unb3) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const auto* gate_bytes    = reinterpret_cast<const unsigned char*>(gate);
    const auto* up_bytes      = reinterpret_cast<const unsigned char*>(up);
    for (std::int64_t i = start; i < n; i += stride) {
        std::int64_t rem = i;
        const auto d0    = static_cast<std::int32_t>(rem % ne0);
        rem /= ne0;
        const auto d1 = static_cast<std::int32_t>(rem % ne1);
        rem /= ne1;
        const auto d2 = static_cast<std::int32_t>(rem % ne2);
        const auto d3 = static_cast<std::int32_t>(rem / ne2);

        const std::int64_t goff =
            static_cast<std::int64_t>(d0) * gnb0 + static_cast<std::int64_t>(d1) * gnb1 +
            static_cast<std::int64_t>(d2) * gnb2 + static_cast<std::int64_t>(d3) * gnb3;
        const std::int64_t uoff =
            static_cast<std::int64_t>(d0) * unb0 + static_cast<std::int64_t>(d1) * unb1 +
            static_cast<std::int64_t>(d2) * unb2 + static_cast<std::int64_t>(d3) * unb3;
        const auto gv = *reinterpret_cast<const __nv_bfloat16*>(gate_bytes + goff);
        const auto uv = *reinterpret_cast<const __nv_bfloat16*>(up_bytes + uoff);
        out[i]        = __float2bfloat16(common::silu(__bfloat162float(gv)) * __bfloat162float(uv));
    }
}

__launch_bounds__(256) __global__
    void silu_mul_dim0_split_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                    __nv_bfloat16* out, std::int32_t rows,
                                    std::int64_t gate_col_stride, std::int64_t up_col_stride) {
    const int col            = blockIdx.y;
    const std::int64_t pair0 = (blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x) *
                               kSiluAndMulPairsPerThread;
    const std::int64_t row_pairs = rows / 2;

    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate + col * gate_col_stride);
    const auto* up2   = reinterpret_cast<const __nv_bfloat162*>(up + col * up_col_stride);
    auto* out2 = reinterpret_cast<__nv_bfloat162*>(out + col * static_cast<std::int64_t>(rows));

    for (int p = 0; p < kSiluAndMulPairsPerThread; ++p) {
        const std::int64_t j = pair0 + p;
        if (j < row_pairs) { out2[j] = silu_mul_pair(gate2[j], up2[j]); }
    }
}

__launch_bounds__(256) __global__
    void silu_mul_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out,
                         std::int64_t n) {
    const std::int64_t tid = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride =
        static_cast<std::int64_t>(gridDim.x) * blockDim.x * kSiluAndMulPairsPerThread;
    const std::int64_t n2 = n / 2;

    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate);
    const auto* up2   = reinterpret_cast<const __nv_bfloat162*>(up);
    auto* out2        = reinterpret_cast<__nv_bfloat162*>(out);
    for (std::int64_t j = tid * kSiluAndMulPairsPerThread; j < n2; j += stride) {
        const __nv_bfloat162 g0 = gate2[j];
        const __nv_bfloat162 u0 = up2[j];
        if (j + 3 < n2) {
            const __nv_bfloat162 g1 = gate2[j + 1];
            const __nv_bfloat162 u1 = up2[j + 1];
            const __nv_bfloat162 g2 = gate2[j + 2];
            const __nv_bfloat162 u2 = up2[j + 2];
            const __nv_bfloat162 g3 = gate2[j + 3];
            const __nv_bfloat162 u3 = up2[j + 3];
            out2[j]                 = silu_mul_pair(g0, u0);
            out2[j + 1]             = silu_mul_pair(g1, u1);
            out2[j + 2]             = silu_mul_pair(g2, u2);
            out2[j + 3]             = silu_mul_pair(g3, u3);
        } else {
            out2[j] = silu_mul_pair(g0, u0);
            if (j + 1 < n2) { out2[j + 1] = silu_mul_pair(gate2[j + 1], up2[j + 1]); }
            if (j + 2 < n2) { out2[j + 2] = silu_mul_pair(gate2[j + 2], up2[j + 2]); }
        }
    }

    if (tid == 0) {
        if ((n & 1) != 0) {
            const std::int64_t i = n - 1;
            out[i] =
                __float2bfloat16(common::silu(__bfloat162float(gate[i])) * __bfloat162float(up[i]));
        }
    }
}

bool can_use_dim0_split_fast_path(const Tensor& gate, const Tensor& up, const Tensor& out) {
    if (gate.ne[2] != 1 || gate.ne[3] != 1) { return false; }
    if ((gate.ne[0] & 1) != 0) { return false; }
    if (gate.nb[0] != static_cast<std::int64_t>(sizeof(__nv_bfloat16)) ||
        up.nb[0] != static_cast<std::int64_t>(sizeof(__nv_bfloat16))) {
        return false;
    }
    if ((gate.nb[1] % static_cast<std::int64_t>(sizeof(__nv_bfloat162))) != 0 ||
        (up.nb[1] % static_cast<std::int64_t>(sizeof(__nv_bfloat162))) != 0) {
        return false;
    }

    const auto gate_addr = reinterpret_cast<std::uintptr_t>(gate.data);
    const auto up_addr   = reinterpret_cast<std::uintptr_t>(up.data);
    const auto out_addr  = reinterpret_cast<std::uintptr_t>(out.data);
    return ((gate_addr | up_addr | out_addr) & (alignof(__nv_bfloat162) - 1)) == 0;
}

void launch_silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    const std::int64_t n = out.numel();
    constexpr int kBlock = 128;
    if (!gate.is_contiguous() || !up.is_contiguous()) {
        if (can_use_dim0_split_fast_path(gate, up, out)) {
            const std::int64_t row_pairs          = gate.ne[0] / 2;
            constexpr std::int64_t kPairsPerBlock = kBlock * kSiluAndMulPairsPerThread;
            const int grid_x                      = static_cast<int>(
                std::max<std::int64_t>(1, common::div_up(row_pairs, kPairsPerBlock)));
            common::for_each_token_slice(gate.ne[1], 1, [&](int token_offset, int token_count) {
                const Tensor gate_slice = gate.slice(1, token_offset, token_count);
                const Tensor up_slice   = up.slice(1, token_offset, token_count);
                Tensor out_slice        = out.slice(1, token_offset, token_count);
                const dim3 grid(static_cast<unsigned int>(grid_x),
                                static_cast<unsigned int>(token_count));
                silu_mul_dim0_split_kernel<<<grid, kBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(gate_slice.data),
                    static_cast<const __nv_bfloat16*>(up_slice.data),
                    static_cast<__nv_bfloat16*>(out_slice.data), gate.ne[0],
                    gate.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
                    up.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
                CUDA_CHECK(cudaGetLastError());
            });
            return;
        }
        const int scalar_grid =
            static_cast<int>(common::div_up(n, static_cast<std::int64_t>(kBlock)));
        silu_mul_strided_input_kernel<<<scalar_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate.data),
            static_cast<const __nv_bfloat16*>(up.data), static_cast<__nv_bfloat16*>(out.data), n,
            gate.ne[0], gate.ne[1], gate.ne[2], gate.nb[0], gate.nb[1], gate.nb[2], gate.nb[3],
            up.nb[0], up.nb[1], up.nb[2], up.nb[3]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const auto gate_addr = reinterpret_cast<std::uintptr_t>(gate.data);
    const auto up_addr   = reinterpret_cast<std::uintptr_t>(up.data);
    const auto out_addr  = reinterpret_cast<std::uintptr_t>(out.data);
    if (((gate_addr | up_addr | out_addr) & (alignof(__nv_bfloat162) - 1)) != 0) {
        const int scalar_grid =
            static_cast<int>(common::div_up(n, static_cast<std::int64_t>(kBlock)));
        silu_mul_scalar_kernel<<<scalar_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate.data),
            static_cast<const __nv_bfloat16*>(up.data), static_cast<__nv_bfloat16*>(out.data), n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const std::int64_t n2                 = n / 2;
    constexpr std::int64_t kPairsPerBlock = kBlock * kSiluAndMulPairsPerThread;
    const int grid =
        static_cast<int>(std::max<std::int64_t>(1, common::div_up(n2, kPairsPerBlock)));

    silu_mul_kernel<<<grid, kBlock, 0, stream>>>(static_cast<const __nv_bfloat16*>(gate.data),
                                                 static_cast<const __nv_bfloat16*>(up.data),
                                                 static_cast<__nv_bfloat16*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, CudaExecutionView execution) {
    const auto stream = execution.stream();
    if (gate.dtype != DType::BF16 || up.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("silu_mul: gate/up/out must be BF16");
    }
    for (int d = 0; d < 4; ++d) {
        if (gate.ne[d] != up.ne[d] || gate.ne[d] != out.ne[d]) {
            throw std::invalid_argument("silu_mul: gate/up/out shapes must match");
        }
    }
    if (!out.is_contiguous()) { throw std::invalid_argument("silu_mul: out must be contiguous"); }
    if (out.numel() == 0) { return; }
    if (gate.device != execution.device() || up.device != execution.device() ||
        out.device != execution.device()) {
        throw std::invalid_argument("silu_mul: operands must belong to the execution device");
    }
    if (gate.data == nullptr || up.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("silu_mul: gate/up/out data must be non-null");
    }

    launch_silu_mul(gate, up, out, stream);
}

} // namespace ninfer::ops
