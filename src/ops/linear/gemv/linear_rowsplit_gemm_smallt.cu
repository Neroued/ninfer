// Compatibility launcher for W8G32 until that format owns a local backend.
#include "ops/linear/gemv/linear_rowsplit_gemm_smallt.cuh"

#include "ops/common/math.h"
#include "ops/linear/reference/linear_generic.h" // launch declaration
#include "core/device.h"                         // CUDA_CHECK

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRowsPerBlockDefault = 8;
constexpr int kStages              = 2;

template <class SC, int kTt, int kRowsPerBlock>
void launch_tt(const __nv_bfloat16* xp, const std::uint8_t* codes, const std::uint8_t* high,
               const std::uint8_t* scales, __nv_bfloat16* outp, std::int32_t n, std::int32_t k,
               std::int32_t t, std::int32_t padded_k, std::int32_t full_slabs,
               cudaStream_t stream) {
    constexpr int kBlockThreads = kRowsPerBlock * 32;
    const dim3 grid(static_cast<unsigned>(div_up(n, kRowsPerBlock)),
                    static_cast<unsigned>(div_up(t, kTt)), 1u);
    linear_rowsplit_gemm_smallt_kernel<SC, kTt, kRowsPerBlock, kStages>
        <<<grid, kBlockThreads, 0, stream>>>(xp, codes, high, scales, outp, n, k, t, padded_k,
                                             full_slabs);
}

// Fallback kTt is the column-tile width: T <= kTt streams the weights exactly
// once. Only 4 and 8 exist for the generic fallback: wider tiles blow the
// register budget and end up latency-bound (see the kernel header). T > 8
// re-streams the weights per 8-column tile until the LargeT tensor-core GEMM
// takes over. Unused columns are skipped by uniform predicates, so kTt only has
// to cover T, not match it.
template <class SC>
void launch_codec(const __nv_bfloat16* xp, const std::uint8_t* codes, const std::uint8_t* high,
                  const std::uint8_t* scales, __nv_bfloat16* outp, std::int32_t n, std::int32_t k,
                  std::int32_t t, std::int32_t padded_k, std::int32_t full_slabs,
                  cudaStream_t stream) {
    if (t <= 4) {
        launch_tt<SC, 4, kRowsPerBlockDefault>(xp, codes, high, scales, outp, n, k, t, padded_k,
                                               full_slabs, stream);
    } else {
        launch_tt<SC, 8, kRowsPerBlockDefault>(xp, codes, high, scales, outp, n, k, t, padded_k,
                                               full_slabs, stream);
    }
}

} // namespace

void linear_rowsplit_gemm_smallt_launch(const Tensor& x, const Weight& w, Tensor& out,
                                        LinearFormat fmt, cudaStream_t stream) {
    const std::int32_t n        = out.ne[0];
    const std::int32_t k        = x.ne[0];
    const std::int32_t t        = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const auto* codes           = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales          = static_cast<const std::uint8_t*>(w.scales);
    const auto* xp              = static_cast<const __nv_bfloat16*>(x.data);
    auto* outp                  = static_cast<__nv_bfloat16*>(out.data);

    // The fast slab path streams x as 16-byte uint4 per column, which needs
    // k % 8 == 0 and a 16-byte aligned x base; otherwise the scalar tail path
    // covers the whole K range (test-only shapes; every model K is 1024-aligned).
    const bool aligned_x = (k % 8) == 0 && (reinterpret_cast<std::uintptr_t>(xp) & 0xfu) == 0;
    const std::int32_t full_slabs = aligned_x ? k / 1024 : 0;

    if (fmt != LinearFormat::W8G32_RowSplit) {
        throw std::invalid_argument("linear: legacy small-T GEMM requires W8G32 RowSplit");
    }
    launch_codec<W8Smallt>(xp, codes, nullptr, scales, outp, n, k, t, padded_k, full_slabs, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
