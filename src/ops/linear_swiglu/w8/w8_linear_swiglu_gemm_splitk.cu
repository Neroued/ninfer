#include "ops/linear_swiglu/w8/w8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/linear/w8/w8_rowsplit_gemm_exact_t_splitk.cuh"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kIntermediate = 6144;
constexpr int kHidden       = 2048;
constexpr int kRowsPerCta   = 8;
constexpr int kFirstExactT  = 2;
constexpr int kLastExactT   = 32;
using ProjectionLauncher    = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

struct W8SwiGluExactTRows {
    static constexpr int kOutputRowsPerCta = kRowsPerCta;
    int intermediate;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & (kRowsPerCta - 1)) +
               (local_row >= kRowsPerCta ? intermediate : 0);
    }
};

struct W8SwiGluExactTEpilogue {
    __nv_bfloat16* out;
    int rows;

    template <int ActiveCols>
    __device__ __forceinline__ void store(int row, float (&projected)[ActiveCols]) const {
        constexpr unsigned kPairMask = 0x0000ffffu;
        const int lane               = static_cast<int>(threadIdx.x) & 31;
        const int output_row         = row - lane + (lane & (kRowsPerCta - 1));
#pragma unroll
        for (int token = 0; token < ActiveCols; ++token) {
            const float up = __shfl_sync(kPairMask, projected[token], (lane & 7) + 8);
            if (lane < kRowsPerCta) {
                out[static_cast<std::int64_t>(token) * rows + output_row] =
                    __float2bfloat16_rn(silu(projected[token]) * up);
            }
        }
    }
};

template <int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    const W8ContiguousOutput ignored_output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const W8SwiGluExactTEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const W8SwiGluExactTRows row_policy{kIntermediate};
    w8_rowsplit_exact_t_splitk_kernel<kHidden, TileCols, ActiveCols, W8ContiguousOutput,
                                      W8SwiGluExactTEpilogue, W8SwiGluExactTRows>
        <<<kIntermediate / kRowsPerCta, 8 * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), ignored_output, epilogue, row_policy);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<kFirstExactT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kLastExactT - kFirstExactT + 1>{});

} // namespace

void w8_linear_swiglu_splitk_exact_t_launch(W8KernelVariant variant, const Tensor& x,
                                            const Weight& w, Tensor& out, cudaStream_t stream) {
    if (variant != W8KernelVariant::None || x.ne[1] < kFirstExactT || x.ne[1] > kLastExactT) {
        throw std::invalid_argument("W8 LinearSwiGLU exact split-K requires T=2..32");
    }
    kLaunchers[x.ne[1] - kFirstExactT](x, w, out, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
