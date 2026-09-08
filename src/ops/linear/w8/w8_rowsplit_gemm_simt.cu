#include "ops/linear/w8/w8_rowsplit_gemm_simt.cuh"

#include "ops/common/math.h"
#include "core/device.h"
#include "ops/common/token_slices.h"
#include "ops/linear/w8/w8_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kRowsPerBlockDefault = 8;
constexpr int kStages              = 2;

template <int ColsPerTile, bool Full>
void launch_tt(const __nv_bfloat16* xp, const std::uint8_t* codes, const std::uint8_t* scales,
               __nv_bfloat16* outp, std::int32_t n, std::int32_t out_ld, std::int32_t k,
               std::int32_t t, std::int32_t padded_k, std::int32_t full_slabs, cudaStream_t stream) {
    constexpr int kBlockThreads = kRowsPerBlockDefault * 32;
    const dim3 grid(static_cast<unsigned>(div_up(n, kRowsPerBlockDefault)),
                    static_cast<unsigned>(div_up(t, ColsPerTile)), 1u);
    const W8ContiguousOutput output{outp, out_ld};
    w8_rowsplit_gemm_simt_kernel<W8RowSplitSimtSchedule, ColsPerTile, kRowsPerBlockDefault, kStages,
                                 Full><<<grid, kBlockThreads, 0, stream>>>(
        xp, codes, scales, output, n, k, t, padded_k, full_slabs);
}

template <int ColsPerTile, bool Full>
void launch_slice(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const auto* xp       = static_cast<const __nv_bfloat16*>(x.data);
    const bool aligned_x = (x.ne[0] % 8) == 0 && (reinterpret_cast<std::uintptr_t>(xp) & 0xfu) == 0;
    const std::int32_t full_slabs = aligned_x ? x.ne[0] / 1024 : 0;

    launch_tt<ColsPerTile, Full>(xp, static_cast<const std::uint8_t*>(w.qdata),
                                 static_cast<const std::uint8_t*>(w.scales),
                                 static_cast<__nv_bfloat16*>(out.data), w.n,
                                 static_cast<std::int32_t>(out.nb[1] / out.nb[0]), x.ne[0],
                                 x.ne[1], w.padded_shape[1], full_slabs, stream);
    CUDA_CHECK(cudaGetLastError());
}

template <int ColsPerTile>
void launch_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const bool full = (out.ne[0] % kRowsPerBlockDefault) == 0 && (x.ne[1] % ColsPerTile) == 0;
    for_each_token_slice(x.ne[1], ColsPerTile, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        if (full) {
            launch_slice<ColsPerTile, true>(x_slice, w, out_slice, stream);
        } else {
            launch_slice<ColsPerTile, false>(x_slice, w, out_slice, stream);
        }
    });
}

} // namespace

void launch_w8_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<4>(x, w, out, stream);
}

void launch_w8_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_route<8>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
