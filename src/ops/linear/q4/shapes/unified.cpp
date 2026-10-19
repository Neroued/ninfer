#include "ops/linear/q4/q4_shapes.h"

#include <stdexcept>

namespace ninfer::ops::detail {

// Unified-template routes, selected with complete-Op cold CUDA Graph measurements on an RTX 5090.

Q4Launch select_q4_n1024_k5120_unified(std::int32_t tokens) {
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_direct;
    if (tokens <= 8) return launch_q4_a16_simt_r4_t4_w2_g8_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 80) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 320) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 576) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 768) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    if (tokens <= 1280) return launch_q4_a16_mma_r32_t128_k64_s2_a2;
    if (tokens <= 1344) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    return launch_q4_a16_mma_r64_t128;
}

Q4Launch select_q4_n4096_k5120_unified(std::int32_t tokens) {
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_direct;
    if (tokens <= 8) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 96) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 128) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 320) return launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2;
    return launch_q4_a16_mma_r64_t128;
}

Q4Launch select_q4_n5120_k6144_unified(std::int32_t tokens) {
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_k6144;
    if (tokens <= 4) return launch_q4_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 96) return launch_q4_a16_mma_r32_t32_k128_s2_a2;
    if (tokens <= 128) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 192) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s3_a3_b2;
    return launch_q4_a16_mma_r64_t128;
}

Q4Launch select_q4_n6144_k5120_unified(std::int32_t tokens) {
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_direct;
    if (tokens <= 4) return launch_q4_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_k5120_t16;
    if (tokens <= 24) return launch_q4_a16_sliced_k5120_t24;
    if (tokens <= 48) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 64) return launch_q4_a16_mma_r32_t32_k128_s2_a2;
    if (tokens <= 96) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 192) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    if (tokens <= 384) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 640) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    return launch_q4_a16_mma_r64_t128;
}

Q4Launch select_q4_n7168_k5120_unified(std::int32_t tokens) {
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_direct;
    if (tokens <= 4) return launch_q4_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 24) return launch_q4_a16_sliced_k5120_t24;
    if (tokens <= 96) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 112) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2;
    if (tokens <= 128) return launch_q4_a16_mma_r64_t48;
    if (tokens <= 192) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    if (tokens <= 288) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 384) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 448) return launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2;
    if (tokens <= 576) return launch_q4_a16_mma_r64_t96;
    return launch_q4_a16_mma_r64_t128;
}

Q4Launch select_q4_n34816_k5120_unified(std::int32_t tokens) {
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_direct;
    if (tokens <= 4) return launch_q4_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q4_a16_sliced_r32_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 63) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 64) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 120) return launch_q4_a16_mma_r64_t120;
    if (tokens <= 128) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 192) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 224) return launch_q4_a16_mma_r64_t112;
    if (tokens <= 240) return launch_q4_a16_mma_r64_t120;
    if (tokens <= 256) return launch_q4_a16_mma_r64_t128;
    if (tokens <= 288) return launch_q4_a16_mma_r64_t96;
    return launch_q4_a16_mma_r64_t128;
}

Q4Launch select_q4_n131072_k5120_unified(std::int32_t tokens) {
    if (tokens <= 1) return launch_q4_a16_gemv_r4_w1_direct;
    if (tokens <= 4) return launch_q4_a16_sliced_k5120_t4;
    if (tokens <= 8) return launch_q4_a16_sliced_r32_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s1;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 64) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q4_a16_mma_r64_t112;
    return launch_q4_a16_mma_r64_t128;
}

Q4Launch select_q4_n131072_k2048_unified(std::int32_t tokens) {
    if (tokens <= 1) return launch_q4_a16_simt_r4_t1_w2_g8_s2;
    if (tokens <= 4) return launch_q4_a16_sliced_k2048_t4;
    if (tokens <= 8) return launch_q4_a16_sliced_r32_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 72) return launch_q4_a16_mma_r64_t72;
    if (tokens <= 80) return launch_q4_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q4_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q4_a16_mma_r64_t112;
    if (tokens <= 120) return launch_q4_a16_mma_r64_t120;
    return launch_q4_a16_mma_r64_t128;
}

Q4Launch select_q4_n3456_k1152_unified(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("q4 linear: T must be a multiple of 4 in [4,131072]");
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r16_t16_w2_s2;
    if (tokens <= 96) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 128) return launch_q4_a16_mma_r32_t32_k128_s2_a2;
    if (tokens <= 192) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 256) return launch_q4_a16_mma_r32_t32_k128_s2_a2;
    if (tokens <= 320) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 384) return launch_q4_a16_mma_r32_t128_k64_s2_a2;
    return launch_q4_a16_mma_r64_t128;
}

Q4Launch select_q4_n4304_k1152_unified(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("q4 linear: T must be a multiple of 4 in [4,131072]");
    if (tokens <= 16) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 96) return launch_q4_a16_mma_r32_t32_k128_s2_a2;
    if (tokens <= 384) return launch_q4_a16_sliced_r32_t32_w2_s2;
    return launch_q4_a16_mma_r64_t128;
}

Q4Launch select_q4_n4352_k1152_unified(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("q4 linear: T must be a multiple of 4 in [4,131072]");
    if (tokens <= 16) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 96) return launch_q4_a16_mma_r32_t32_k128_s2_a2;
    if (tokens <= 384) return launch_q4_a16_sliced_r32_t32_w2_s2;
    return launch_q4_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
