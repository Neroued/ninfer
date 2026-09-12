#include "ops/linear/w8/w8_dispatch.h"
#include "ops/linear/w8/w8_feature.h"

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
            // Exact-T schedules own the DFlash2 decode interval. R32/C64 is the single bridge;
            // R64/C128 is the measured T=1024 prefill winner.
            if (t <= 53) { return launch_w8_small_t; }
            if (t <= 192) { return launch_w8_mma_r32_c64; }
            return launch_w8_mma_r64_c128;
        case 14336:
            if (t <= 48) { return launch_w8_small_t; }
            return launch_w8_mma_r64_c128;
        case 34816:
            if (t <= 40) { return launch_w8_small_t; }
            if (t <= 48) { return launch_w8_mma_r64x16_c48_k128_a1; }
            if (t <= 52) { return launch_w8_small_t; }
            if (t <= 64) { return launch_w8_mma_r128_c64; }
            return launch_w8_mma_r64_c128;
        case 248320:
            if (t <= 33) { return launch_w8_small_t; }
            if (t <= 48) { return launch_w8_mma_r64x16_c48_k128_a1; }
            if (t <= 64) { return launch_w8_mma_r64x32_c64_k128_a1; }
            if (t <= 96) { return launch_w8_mma_r64_c96; }
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
    case 25600:
        if (n == 5120) {
            if (t <= 56) { return launch_w8_feature_small_t; }
            if (t <= 64) { return launch_w8_feature_r16_c64; }
            if (t <= 128) { return launch_w8_feature_r32_c64; }
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

W8Launch select_w8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
        return select_w8_a16_launch(n, k, t);
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
