#include "ops/linear/w8/w8_dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

W8Launch select_w8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) { throw std::invalid_argument("w8 linear: unsupported shape or T"); }

    switch (k) {
    case 10240:
        if (n == 5120) {
            if (t <= 48) { return launch_w8_small_t; }
            return launch_w8_mma_r64_c128;
        }
        break;
    case 5120:
        switch (n) {
        case 1024:
            if (t <= 4) { return launch_w8_simt_r8_c4; }
            if (t <= 16) { return launch_w8_simt_r8_c8; }
            return launch_w8_mma_r32_c128;
        case 6144:
            if (t <= 4) { return launch_w8_simt_r8_c4; }
            if (t <= 16) { return launch_w8_simt_r8_c8; }
            return launch_w8_mma_r64_c128;
        case 14336:
            if (t <= 48) { return launch_w8_small_t; }
            return launch_w8_mma_r64_c128;
        case 34816:
            if (t <= 40) { return launch_w8_small_t; }
            if (t <= 48) { return launch_w8_mma_r64x16_c48_k128_a1; }
            return launch_w8_mma_r64_c128;
        case 248320:
            if (t <= 33) { return launch_w8_small_t; }
            if (t <= 48) { return launch_w8_mma_r64x16_c48_k128_a1; }
            if (t <= 64) { return launch_w8_mma_r32_c64; }
            return launch_w8_mma_r64_c128;
        default:
            break;
        }
        break;
    case 6144:
        if (n == 5120) {
            if (t <= 48) { return launch_w8_small_t; }
            return launch_w8_mma_r64_c128;
        }
        break;
    case 17408:
        if (n == 5120) {
            if (t <= 48) { return launch_w8_small_t; }
            return launch_w8_mma_r64_c128;
        }
        break;
    case 4096:
        if (n == 2048) {
            if (t <= 48) { return launch_w8_small_t; }
            if (t <= 56) { return launch_w8_simt_r8_c4; }
            if (t <= 895) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        }
        break;
    case 2048:
        switch (n) {
        case 1024:
            if (t <= 4) { return launch_w8_simt_r8_c4; }
            if (t <= 16) { return launch_w8_simt_r8_c8; }
            return launch_w8_mma_r32_c128;
        case 9216:
            if (t <= 13) { return launch_w8_simt_r8_c4; }
            if (t <= 128) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        case 12288:
            if (t <= 16) { return launch_w8_simt_r8_c4; }
            return launch_w8_mma_r64_c128;
        default:
            break;
        }
        break;
    case 4608:
        if (t > 32768) { break; }
        switch (n) {
        case 2048:
            if (t <= 14 || t == 16 || t == 20 || t == 24 || t == 28 || t == 32) {
                return launch_w8_simt_r8_c4;
            }
            if (t <= 871) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        case 4608:
            if (t <= 8 || t == 12) { return launch_w8_simt_r8_c4; }
            if (t <= 256) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        case 5120:
            if (t <= 4) { return launch_w8_simt_r8_c4; }
            if (t == 5) { return launch_w8_simt_r8_c8; }
            return launch_w8_mma_r64_c128;
        default:
            break;
        }
        break;
    case 16384:
        if (n != 2048) { break; }
        if (t == 1) { return launch_w8_decode_r4; }
        if (t <= 48) { return launch_w8_exact_t_splitk; }
        if (t <= 128) { return launch_w8_dflash_medium; }
        if (t <= 144) { return launch_w8_medium_splitk_c144; }
        if (t <= 255) { return launch_w8_mma_r32_c128; }
        if (t <= 384) { return launch_w8_mma_r32_c64; }
        if (t <= 480) { return launch_w8_mma_r32_c96; }
        if (t == 481) { return launch_w8_exact_mma_r32_c96; }
        if (t <= 640) { return launch_w8_mma_r32_c128; }
        if (t <= 668) { return launch_w8_exact_mma_r32_c128; }
        if (t <= 672) { return launch_w8_mma_r48_c96; }
        if (t == 673) { return launch_w8_exact_mma_r48_c96; }
        if (t <= 704) { return launch_w8_mma_r48_c64; }
        if (t <= 784) { return launch_w8_mma_r48_c112; }
        if (t <= 896) { return launch_w8_mma_r48_c128; }
        if (t <= 912) { return launch_w8_exact_mma_r48_c128; }
        if (t <= 960) { return launch_w8_mma_r64_c96; }
        if (t <= 1007) { return launch_w8_exact_mma_r64_c96; }
        if (t == 1008) { return launch_w8_mma_r64_c112; }
        if (t <= 1119) { return launch_w8_mma_r64_c128; }
        if (t == 1120) { return launch_w8_mma_r64_c112; }
        if (t <= 1280) { return launch_w8_mma_r64_c128; }
        if (t <= 1313) { return launch_w8_exact_mma_r64_c128; }
        if (t <= 1344) { return launch_w8_mma_r128_c64; }
        if (t <= 1440) { return launch_w8_mma_r96_c96; }
        if (t <= 1500) { return launch_w8_exact_mma_r96_c96; }
        if (t <= 1680) { return launch_w8_mma_r128_c80; }
        if (t <= 1745) { return launch_w8_exact_mma_r128_c80; }
        if (t <= 1791) { return launch_w8_mma_r48_c128; }
        if (t == 1792) { return launch_w8_mma_r64_c128; }
        if (t <= 1919) { return launch_w8_mma_r48_c128; }
        if (t == 1920) { return launch_w8_mma_r64_c128; }
        if (t <= 1953) { return launch_w8_exact_mma_r64_c128; }
        if (t <= 2016) { return launch_w8_mma_r64_c96; }
        if (t <= 2048) { return launch_w8_exact_mma_r64_c96; }
        if (t <= 2112) { return launch_w8_mma_r96_c96; }
        return launch_w8_mma_r64_c128;
    default:
        break;
    }

    throw std::invalid_argument("w8 linear: unsupported shape or T");
}

#ifdef NINFER_VOLTA_BUILD
// select_w8_a16_launch's huge shape/T threshold table (above) routes most non-trivial T to one
// of ~25 Ampere+ mma/ldmatrix schedules (the mma_*/exact_mma_* family) or the split-K composite
// path (w8_rowsplit_gemm_splitk.cu, itself built on the trap-stubbed
// w8_rowsplit_medium_t_splitk_kernel) -- all trap-stubbed on sm_70. Intercepting here rather
// than rewriting that table: launch_w8_small_t already redirects unconditionally to
// launch_w8_simt_r8_c8 under NINFER_VOLTA_BUILD (see w8_small_t.cu), which is fully general
// over T via for_each_token_slice, so it's a safe universal replacement for any of those
// schedules regardless of which (n,k,t) triggered them. The genuinely SIMT-native schedules
// (simt_r8_c4/c8, decode_r4, small_t itself) are left as originally selected. See
// the V100 implementation.
bool w8_launch_needs_volta_fallback(W8Launch launch) noexcept {
    return launch != launch_w8_decode_r4 && launch != launch_w8_small_t &&
           launch != launch_w8_simt_r8_c4 && launch != launch_w8_simt_r8_c8 &&
           launch != launch_w8_simt_r4_c16;
}

// Band for the fused tensor-core route. Whatever the table above selected, on Volta it resolves
// to sliced r8_c8, which re-reads the whole weight once per 8 output columns. So the choice is
// between a route whose cost is linear in T and one whose cost is a single weight read and
// therefore nearly flat, and the only question is where the two lines cross. Measured at k=5120
// on a V100-SXM2-32GB, microseconds, sliced r8_c8 against fused:
//
//   T:               2          4          6          9         16
//   n=  1024    33/198     45/199     75/198     92/201     94/202
//   n=  6144    84/214    109/228    171/239    258/231    401/240
//   n= 34816   398/803    693/810    898/819   1360/819   2175/839
//   n=248320  2232/5652  4457/5702  5785/5790  8932/5839  14323/6052
//
// There is no upper edge to find: the incumbent stays linear all the way up (at T=160 on the
// output head it is 142.2ms against the fused route's 34.2ms) because Volta has no tensor-core W8
// route to hand off to. So the band is open above.
//
// The lower edge is shape-dependent, because the fused route's flat cost is set by how much of the
// machine N alone can fill -- it has no split-K (see w8_volta_mma_gemm.cuh), so at 32 output rows
// per CTA it wants thousands of rows. n=248320 gives 7760 CTAs and reaches 225 GB/s; n=34816 gives
// 1088 and reaches 222; n=6144 gives 192 and manages 147; n=1024 gives 32 and collapses to 26,
// which is why that shape is excluded outright rather than given a wider band.
constexpr std::int32_t kVoltaMmaMinRows      = 4096;
constexpr std::int32_t kVoltaMmaWideRows     = 32768;
constexpr std::int32_t kVoltaMmaMinCols      = 9; // 4096 <= n < 32768
constexpr std::int32_t kVoltaMmaWideMinCols  = 7; // n >= 32768, where the flat cost is lowest

// Band for the quadpair-split-N route (w8_volta_qpn_gemm.cuh), measured on the 27B output head,
// W8 [248320,5120], against whichever route production would otherwise take. Microseconds:
//
//   T:            1     2     3     4     5     6     7     8     9    12    16
//   incumbent  1797  2119  3838  4461  5136  5789  5270  5300  5390  5434  5502
//   qpn        1797  2933  2936  2940  2987  2996  3001  2971  5115  5368  5655
//
// QPN is flat across the band, as the mapping intends -- 2.93-3.00ms at every width -- while the
// incumbent climbs from 2.1 to 5.8ms. Both edges are the incumbent's shape again. T=1 and T=2 are
// the GEMV and its near neighbour, and T=1 in particular is already *at* the machine (752 GB/s of
// this card's ~794 measured ceiling), so there is nothing to take there. From T=9 the route needs
// a second 8-row A tile, which doubles it, while the 32x8 fused route absorbs those rows free.
//
// In band this is 1.31x at T=3 rising to 1.93x at T=6 and 1.78x at T=8, on the op the phase timing
// shows inside the 90%-of-a-round target verification pass.
constexpr std::int32_t kVoltaQpnMinCols = 3;
constexpr std::int32_t kVoltaQpnMaxCols = 8;
constexpr std::int32_t kVoltaQpnMinRows = 4096;

bool w8_uses_volta_qpn(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    return t >= kVoltaQpnMinCols && t <= kVoltaQpnMaxCols && n >= kVoltaQpnMinRows &&
           w8_volta_qpn_supported(n, k, t);
}

bool w8_uses_volta_mma(std::int32_t n, std::int32_t k, std::int32_t t) noexcept {
    if (n < kVoltaMmaMinRows) { return false; }
    const std::int32_t first = n >= kVoltaMmaWideRows ? kVoltaMmaWideMinCols : kVoltaMmaMinCols;
    return t >= first && w8_volta_mma_supported(n, k, t);
}
#endif

W8Launch select_w8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8: {
        const W8Launch launch = select_w8_a16_launch(n, k, t);
#ifdef NINFER_VOLTA_BUILD
        // The r4_c16 interception that used to sit here is deliberately gone. It was introduced to
        // route around the T=8 collapse, which turned out to be the full-specialization register
        // spill since fixed in w8_rowsplit_gemm_simt.cu. With the spill gone, sliced r8_c8 (via
        // launch_w8_small_t, general over T) beats r4_c16 at every T the latter covered:
        // 2.0-2.3x at T=9, 1.26-1.34x at T=16, 1.31-1.43x at T=32, measured over all six W8
        // shapes. Note this is the opposite conclusion to Q5, whose r8_c8 is much weaker and which
        // therefore keeps its r4_c16 route -- see q5_dispatch.cpp.
        // The fused tensor-core route displaces both of those inside its band; see
        // w8_volta_mma_gemm.cuh and the band constants above.
        if (w8_uses_volta_qpn(n, k, t)) { return launch_w8_volta_qpn; }
        if (w8_uses_volta_mma(n, k, t)) { return launch_w8_volta_mma; }
        if (w8_launch_needs_volta_fallback(launch)) { return launch_w8_small_t; }
#endif
        return launch;
    }
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("w8 linear: unsupported policy");
}

void w8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    const W8Launch launch = select_w8_launch(w.n, w.k, x.ne[1], policy);
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail
