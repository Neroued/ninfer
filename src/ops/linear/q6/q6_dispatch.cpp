#include "ops/linear/q6/q6_dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

Q6Launch select_q6_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) { throw std::invalid_argument("q6 linear: unsupported shape or T"); }

    switch (k) {
    case 5120:
        if (n == 248320) {
            if (t <= 4) { return launch_q6_simt_r8_c4; }
            if (t == 5) { return launch_q6_simt_r8_c5; }
            if (t == 6) { return launch_q6_simt_r8_c6; }
            if (t == 7) { return launch_q6_simt_r8_c7; }
            if (t <= 16) { return launch_q6_mma_r64_c16_k128; }
            if (t <= 24) { return launch_q6_mma_r64_c24_k128; }
            if (t <= 32) { return launch_q6_mma_r64_c32_k128; }
            if (t <= 48) { return launch_q6_mma_r64_c48_k128; }
            return launch_q6_mma_r64_c128;
        }
        break;
    case 2048:
        if (n == 248320) {
            if (t <= 3) { return launch_q6_simt_r8_c4; }
            if (t <= 16) { return launch_q6_mma_r64_c16_k128; }
            if (t <= 24) { return launch_q6_mma_r64_c24_k128; }
            if (t <= 32) { return launch_q6_mma_r64_c32_k128; }
            if (t <= 40) { return launch_q6_mma_r64_c40_k128; }
            if (t <= 48) { return launch_q6_mma_r64_c48_k128; }
            if (t <= 56) { return launch_q6_mma_r64_c56_k128; }
            if (t <= 64) { return launch_q6_mma_r64_c64_k128; }
            if (t <= 72) { return launch_q6_mma_r64_c72_k128; }
            if (t <= 80) { return launch_q6_mma_r64_c80; }
            if (t <= 87) { return launch_q6_mma_r64_c96; }
            if (t == 88) { return launch_q6_mma_r64_c88_k128; }
            if (t <= 96) { return launch_q6_mma_r64_c96; }
            if (t <= 111) { return launch_q6_mma_r64_c112_partial; }
            if (t == 112) { return launch_q6_mma_r64_c112; }
            return launch_q6_mma_r64_c128;
        }
        break;
    case 1536:
        if (n == 1152) {
            if (t < 4 || t > 131072 || (t % 4) != 0) { break; }
            if (t <= 96) { return launch_q6_simt_r8_c4; }
            if (t <= 704) { return launch_q6_mma_r64_c64; }
            if (t <= 828) { return launch_q6_mma_r64_c128; }
            if (t == 832) { return launch_q6_mma_r64_c64; }
            if (t <= 896) { return launch_q6_mma_r64_c128; }
            if (t <= 960) { return launch_q6_mma_r64_c64; }
            if (t <= 1024) { return launch_q6_mma_r64_c128; }
            if (t <= 1088) { return launch_q6_mma_r64_c64; }
            return launch_q6_mma_r64_c128;
        }
        break;
    default:
        break;
    }

    throw std::invalid_argument("q6 linear: unsupported shape or T");
}

#ifdef NINFER_VOLTA_BUILD
bool q6_launch_needs_volta_fallback(Q6Launch launch) noexcept {
    return launch != launch_q6_simt_r8_c4 && launch != launch_q6_simt_r8_c5 &&
           launch != launch_q6_simt_r8_c6 && launch != launch_q6_simt_r8_c7 &&
           launch != launch_q6_simt_r8_c8;
}
#endif

Q6Launch select_q6_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8: {
        const Q6Launch launch = select_q6_a16_launch(n, k, t);
#ifdef NINFER_VOLTA_BUILD
        if (q6_launch_needs_volta_fallback(launch)) { return launch_q6_simt_r8_c8; }
#endif
        return launch;
    }
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("q6 linear: unsupported policy");
}

void q6_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    const Q6Launch launch = select_q6_launch(w.n, w.k, x.ne[1], policy);
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail
