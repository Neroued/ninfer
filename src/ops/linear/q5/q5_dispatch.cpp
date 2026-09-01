#include "ops/linear/q5/q5_dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

Q5Launch select_q5_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) { throw std::invalid_argument("q5 linear: unsupported shape or T"); }

    switch (k) {
    case 5120:
        switch (n) {
        case 1024:
            if (t <= 4) { return launch_q5_simt_r8_c4; }
            if (t <= 16) { return launch_q5_simt_r8_c8; }
            return launch_q5_mma_r64_c128;
        case 6144:
            if (t == 1) { return launch_q5_gemv_r16_s2_x; }
            if (t <= 6) { return launch_q5_simt_split4_exact; }
            if (t <= 24) { return launch_q5_simt_r8_c8; }
            if (t <= 64) { return launch_q5_mma_r64_c64; }
            return launch_q5_mma_r64_c128;
        case 7168:
            if (t == 1) { return launch_q5_gemv_r16_s2_x; }
            if (t <= 6) { return launch_q5_simt_split4_exact; }
            if (t <= 16) { return launch_q5_simt_r8_c4; }
            return launch_q5_mma_r64_c128;
        default:
            break;
        }
        break;
    case 6144:
        if (n == 5120) {
            if (t == 1) { return launch_q5_simt_r8_c4; }
            if (t <= 6) { return launch_q5_simt_split2_exact; }
            if (t <= 24) { return launch_q5_simt_r8_c8; }
            return launch_q5_mma_r64_c128;
        }
        break;
    case 17408:
        if (n == 5120) {
            if (t == 1) { return launch_q5_simt_r8_c4; }
            if (t <= 6) { return launch_q5_simt_split2_exact; }
            if (t <= 24) { return launch_q5_simt_r8_c8; }
            return launch_q5_mma_r64_c128;
        }
        break;
    case 1152:
        if (n == 1152 && t >= 4 && t <= 131072 && (t % 4) == 0) {
            if (t <= 76) { return launch_q5_simt_r8_c4; }
            if (t <= 636) { return launch_q5_mma_r64_c64; }
            if (t <= 700) { return launch_q5_mma_r64_c128; }
            if (t == 704) { return launch_q5_mma_r64_c64; }
            if (t <= 828) { return launch_q5_mma_r64_c128; }
            if (t == 832) { return launch_q5_mma_r64_c64; }
            if (t <= 896) { return launch_q5_mma_r64_c128; }
            if (t <= 960) { return launch_q5_mma_r64_c64; }
            if (t <= 1024) { return launch_q5_mma_r64_c128; }
            if (t <= 1088) { return launch_q5_mma_r64_c64; }
            return launch_q5_mma_r64_c128;
        }
        break;
    case 4304:
        if (n == 1152 && t >= 4 && t <= 131072 && (t % 4) == 0) {
            if (t <= 120) { return launch_q5_simt_r8_c4; }
            if (t <= 1148) { return launch_q5_mma_r64_c64; }
            return launch_q5_mma_r64_c128;
        }
        break;
    default:
        break;
    }

    throw std::invalid_argument("q5 linear: unsupported shape or T");
}

#ifdef NINFER_VOLTA_BUILD
bool q5_launch_needs_volta_fallback(Q5Launch launch) noexcept {
    return launch != launch_q5_gemv_r16_s2_x && launch != launch_q5_simt_split4_exact &&
           launch != launch_q5_simt_split2_exact && launch != launch_q5_simt_r8_c4 &&
           launch != launch_q5_simt_r8_c8 && launch != launch_q5_simt_r4_c16;
}
#endif

Q5Launch select_q5_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8: {
        const Q5Launch launch = select_q5_a16_launch(n, k, t);
#ifdef NINFER_VOLTA_BUILD
        // The exact split2/split4 routes are the best thing Q5 has on Volta, but the shared table
        // caps them at T<=6 because dispatch_exact_cols only instantiated [2,6]. That cap was the
        // single worst Q5 cliff: mlp_down T=6 194us (split2) -> T=7 670us (r8_c8), a 3.4x step for
        // one extra column. dispatch_exact_cols now covers [2,16], so carry the routes further.
        //
        // The upper bounds are measured crossovers against what the table would otherwise pick,
        // and they differ per kernel: split2 (K=6144/17408) still wins at T=15 (mlp_down 744us ->
        // 594us) and loses at 16; split4 (K=5120) wins to T=10 (attn_gate_value 205us -> 153us)
        // and loses from 11.
        //
        // Shapes are named explicitly rather than probed via select_q5_a16_launch(n, k, 6): that
        // probe throws for the 1152 vision entries, whose table guard requires t % 4 == 0, which
        // crashed ninfer_linear_q5_a16_test.
        if (t >= 7) {
            const bool split2 = (n == 5120 && (k == 6144 || k == 17408));
            const bool split4 = (k == 5120 && (n == 6144 || n == 7168));
            if (split2 && t <= 15) { return launch_q5_simt_split2_exact; }
            if (split4 && t <= 10) { return launch_q5_simt_split4_exact; }
        }
        // r4_c16 is kept for Q5 (unlike W8, where it was removed): Q5's r8_c8 is the weak kernel
        // here, and dropping r4_c16 measured ~2x worse at T>=9 (mlp_down T=16 746us -> 1412us,
        // T=32 1412us -> 2808us). It only covers what the split routes above do not reach.
        if (t > 8 && t <= 32) { return launch_q5_simt_r4_c16; }
        if (q5_launch_needs_volta_fallback(launch)) { return launch_q5_simt_r8_c8; }
#endif
        return launch;
    }
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("q5 linear: unsupported policy");
}

void q5_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    const Q5Launch launch = select_q5_launch(w.n, w.k, x.ne[1], policy);
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail
