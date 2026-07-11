// qus::kernels — silu_mul launcher: grid/block/stream configuration + kernel launch.
// The only translation unit that includes this op's kernel header.
// See docs/kernel-development.md §2.
#include "kernels/launcher/silu_and_mul.h"

#include "kernels/common/math.h"
#include "kernels/kernel/silu_and_mul.cuh"
#include "qus/core/device.h"  // CUDA_CHECK

#include <algorithm>
#include <cstdint>

namespace qus::kernels::detail {
namespace {

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

} // namespace

void silu_and_mul_launch(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream) {
    const std::int64_t n = out.numel();
    constexpr int kBlock = 128;
    if (!gate.is_contiguous() || !up.is_contiguous()) {
        if (can_use_dim0_split_fast_path(gate, up, out)) {
            const std::int64_t row_pairs = gate.ne[0] / 2;
            constexpr std::int64_t kPairsPerBlock = kBlock * kSiluAndMulPairsPerThread;
            const int grid_x = static_cast<int>(
                std::max<std::int64_t>(1, div_up(row_pairs, kPairsPerBlock)));
            const dim3 grid(grid_x, static_cast<unsigned int>(gate.ne[1]));
            silu_and_mul_dim0_split_kernel<<<grid, kBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(gate.data),
                static_cast<const __nv_bfloat16*>(up.data), static_cast<__nv_bfloat16*>(out.data),
                gate.ne[0], gate.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
                up.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        const int scalar_grid =
            static_cast<int>(div_up(n, static_cast<std::int64_t>(kBlock)));
        silu_and_mul_strided_input_kernel<<<scalar_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate.data), static_cast<const __nv_bfloat16*>(up.data),
            static_cast<__nv_bfloat16*>(out.data), n, gate.ne[0], gate.ne[1], gate.ne[2],
            gate.nb[0], gate.nb[1], gate.nb[2], gate.nb[3], up.nb[0], up.nb[1], up.nb[2],
            up.nb[3]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const auto gate_addr = reinterpret_cast<std::uintptr_t>(gate.data);
    const auto up_addr   = reinterpret_cast<std::uintptr_t>(up.data);
    const auto out_addr  = reinterpret_cast<std::uintptr_t>(out.data);
    if (((gate_addr | up_addr | out_addr) & (alignof(__nv_bfloat162) - 1)) != 0) {
        const int scalar_grid =
            static_cast<int>(div_up(n, static_cast<std::int64_t>(kBlock)));
        silu_and_mul_scalar_kernel<<<scalar_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate.data), static_cast<const __nv_bfloat16*>(up.data),
            static_cast<__nv_bfloat16*>(out.data), n);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const std::int64_t n2 = n / 2;
    constexpr std::int64_t kPairsPerBlock = kBlock * kSiluAndMulPairsPerThread;
    const int grid =
        static_cast<int>(std::max<std::int64_t>(1, div_up(n2, kPairsPerBlock)));

    silu_and_mul_kernel<<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate.data), static_cast<const __nv_bfloat16*>(up.data),
        static_cast<__nv_bfloat16*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
