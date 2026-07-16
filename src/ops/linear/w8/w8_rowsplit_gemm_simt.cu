#include "ops/linear/w8/w8_rowsplit_gemm_simt.cuh"

#include "ops/common/math.h"
#include "ops/linear/w8/w8_rowsplit_kernels.h"
#include "core/device.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRowsPerBlockDefault = 8;
constexpr int kStages              = 2;

template <int ColsPerTile, W8KernelVariant Variant>
void launch_tt(const __nv_bfloat16* xp, const std::uint8_t* codes, const std::uint8_t* scales,
               __nv_bfloat16* outp, std::int32_t n, std::int32_t k, std::int32_t t,
               std::int32_t padded_k, std::int32_t full_slabs, cudaStream_t stream) {
    constexpr int kBlockThreads = kRowsPerBlockDefault * 32;
    const dim3 grid(static_cast<unsigned>(div_up(n, kRowsPerBlockDefault)),
                    static_cast<unsigned>(div_up(t, ColsPerTile)), 1u);
    w8_rowsplit_gemm_simt_kernel<W8RowSplitSimtSchedule, ColsPerTile, kRowsPerBlockDefault, kStages,
                                 Variant><<<grid, kBlockThreads, 0, stream>>>(
        xp, codes, scales, outp, n, k, t, padded_k, full_slabs);
}

template <int ColsPerTile>
void launch_variant(W8KernelVariant variant, const Tensor& x, const Weight& w, Tensor& out,
                    cudaStream_t stream) {
    const auto* xp       = static_cast<const __nv_bfloat16*>(x.data);
    const bool aligned_x = (x.ne[0] % 8) == 0 && (reinterpret_cast<std::uintptr_t>(xp) & 0xfu) == 0;
    const std::int32_t full_slabs = aligned_x ? x.ne[0] / 1024 : 0;

    switch (variant) {
    case W8KernelVariant::Full:
        launch_tt<ColsPerTile, W8KernelVariant::Full>(
            xp, static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            out.ne[0], x.ne[0], x.ne[1], w.padded_shape[1], full_slabs, stream);
        break;
    case W8KernelVariant::Predicated:
        launch_tt<ColsPerTile, W8KernelVariant::Predicated>(
            xp, static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
            out.ne[0], x.ne[0], x.ne[1], w.padded_shape[1], full_slabs, stream);
        break;
    case W8KernelVariant::None:
        throw std::invalid_argument("w8 SIMT launch requires Full or Predicated variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void w8_rowsplit_gemm_simt_r8_c4_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                        Tensor& out, cudaStream_t stream) {
    launch_variant<4>(variant, x, w, out, stream);
}

void w8_rowsplit_gemm_simt_r8_c8_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                        Tensor& out, cudaStream_t stream) {
    launch_variant<8>(variant, x, w, out, stream);
}

} // namespace ninfer::ops::detail
