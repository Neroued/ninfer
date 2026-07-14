// Launcher for the warp-per-row small-T row-split low-bit GEMM. This is the
// universal low-bit path outside the tuned T==1 GEMVs and the LargeT tensor-core
// GEMM: SmallT for Q4/Q5/Q6, every T for W8G32 (no W8 MMA path exists), generic
// T==1 shapes, and the k%8!=0 LargeT fallback. The codec is selected by fmt.
#include "kernels/linear/gemv/linear_rowsplit_gemm_smallt.cuh"

#include "kernels/common/math.h"
#include "kernels/linear/reference/linear_generic.h" // launch declaration
#include "ninfer/core/device.h"                          // CUDA_CHECK

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::kernels::detail {
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

template <int kTt, int kFullSlabs, int kStride>
void launch_q5_direct_split2(const __nv_bfloat16* xp, const std::uint8_t* codes,
                             const std::uint8_t* high, const std::uint8_t* scales,
                             __nv_bfloat16* outp, std::int32_t n, std::int32_t k,
                             std::int32_t t, std::int32_t padded_k, std::int32_t full_slabs,
                             cudaStream_t stream) {
    constexpr int kBlockThreads = 2 * 32;
    const dim3    grid(static_cast<unsigned>(n), 1u, 1u);
    linear_rowsplit_gemm_smallt_kernel_direct_split2_q5<Q5Smallt, kTt, kFullSlabs, kStride>
        <<<grid, kBlockThreads, 0, stream>>>(xp, codes, high, scales, outp, n, k, t, padded_k,
                                             full_slabs);
}

template <int kTt, int kFullSlabs, int kStride>
void launch_q5_direct_split4(const __nv_bfloat16* xp, const std::uint8_t* codes,
                             const std::uint8_t* high, const std::uint8_t* scales,
                             __nv_bfloat16* outp, std::int32_t n, std::int32_t k,
                             std::int32_t t, std::int32_t padded_k, std::int32_t full_slabs,
                             cudaStream_t stream) {
    constexpr int kBlockThreads = 4 * 32;
    const dim3    grid(static_cast<unsigned>(n), 1u, 1u);
    linear_rowsplit_gemm_smallt_kernel_direct_split4_q5<Q5Smallt, kTt, kFullSlabs, kStride>
        <<<grid, kBlockThreads, 0, stream>>>(xp, codes, high, scales, outp, n, k, t, padded_k,
                                             full_slabs);
}

template <int kTt>
bool try_launch_q5_direct(const __nv_bfloat16* xp, const std::uint8_t* codes,
                          const std::uint8_t* high, const std::uint8_t* scales,
                          __nv_bfloat16* outp, std::int32_t n, std::int32_t k,
                          std::int32_t t, std::int32_t padded_k, std::int32_t full_slabs,
                          cudaStream_t stream) {
    if (full_slabs * 1024 != k || padded_k != k) { return false; }
    if (n == 7168 && k == 5120) {
        launch_q5_direct_split2<kTt, 5, 5120>(xp, codes, high, scales, outp, n, k, t, padded_k,
                                              full_slabs, stream);
        return true;
    }
    if (n == 6144 && k == 5120) {
        launch_q5_direct_split4<kTt, 5, 5120>(xp, codes, high, scales, outp, n, k, t, padded_k,
                                              full_slabs, stream);
        return true;
    }
    if (n == 5120 && k == 6144) {
        launch_q5_direct_split2<kTt, 6, 6144>(xp, codes, high, scales, outp, n, k, t, padded_k,
                                              full_slabs, stream);
        return true;
    }
    if (n == 5120 && k == 17408) {
        launch_q5_direct_split2<kTt, 17, 17408>(xp, codes, high, scales, outp, n, k, t, padded_k,
                                                full_slabs, stream);
        return true;
    }
    return false;
}

bool try_launch_q5_direct(const __nv_bfloat16* xp, const std::uint8_t* codes,
                          const std::uint8_t* high, const std::uint8_t* scales,
                          __nv_bfloat16* outp, std::int32_t n, std::int32_t k,
                          std::int32_t t, std::int32_t padded_k, std::int32_t full_slabs,
                          cudaStream_t stream) {
    switch (t) {
    case 2:
        return try_launch_q5_direct<2>(xp, codes, high, scales, outp, n, k, t, padded_k,
                                       full_slabs, stream);
    case 3:
        return try_launch_q5_direct<3>(xp, codes, high, scales, outp, n, k, t, padded_k,
                                       full_slabs, stream);
    case 4:
        return try_launch_q5_direct<4>(xp, codes, high, scales, outp, n, k, t, padded_k,
                                       full_slabs, stream);
    case 5:
        return try_launch_q5_direct<5>(xp, codes, high, scales, outp, n, k, t, padded_k,
                                       full_slabs, stream);
    case 6:
        return try_launch_q5_direct<6>(xp, codes, high, scales, outp, n, k, t, padded_k,
                                       full_slabs, stream);
    default:
        return false;
    }
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
    if constexpr (std::is_same_v<SC, Q5Smallt>) {
        if (try_launch_q5_direct(xp, codes, high, scales, outp, n, k, t, padded_k, full_slabs,
                                 stream)) {
            return;
        }
    }
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
    const auto*        codes    = static_cast<const std::uint8_t*>(w.qdata);
    const auto*        high     = static_cast<const std::uint8_t*>(w.qhigh);
    const auto*        scales   = static_cast<const std::uint8_t*>(w.scales);
    const auto*        xp       = static_cast<const __nv_bfloat16*>(x.data);
    auto*              outp     = static_cast<__nv_bfloat16*>(out.data);

    // The fast slab path streams x as 16-byte uint4 per column, which needs
    // k % 8 == 0 and a 16-byte aligned x base; otherwise the scalar tail path
    // covers the whole K range (test-only shapes; every model K is 1024-aligned).
    const bool aligned_x =
        (k % 8) == 0 && (reinterpret_cast<std::uintptr_t>(xp) & 0xfu) == 0;
    const std::int32_t full_slabs = aligned_x ? k / 1024 : 0;

    switch (fmt) {
    case LinearFormat::Q4G64_RowSplit:
        launch_codec<Q4Smallt>(xp, codes, high, scales, outp, n, k, t, padded_k, full_slabs,
                               stream);
        break;
    case LinearFormat::Q5G64_RowSplit:
        launch_codec<Q5Smallt>(xp, codes, high, scales, outp, n, k, t, padded_k, full_slabs,
                               stream);
        break;
    case LinearFormat::Q6G64_RowSplit:
        launch_codec<Q6Smallt>(xp, codes, high, scales, outp, n, k, t, padded_k, full_slabs,
                               stream);
        break;
    case LinearFormat::W8G32_RowSplit:
        launch_codec<W8Smallt>(xp, codes, nullptr, scales, outp, n, k, t, padded_k, full_slabs,
                               stream);
        break;
    default:
        throw std::invalid_argument(
            "linear: small-T GEMM requires a Q4/Q5/Q6/W8G32 row-split format");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::kernels::detail
