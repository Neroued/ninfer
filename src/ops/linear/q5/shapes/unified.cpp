#include "ops/linear/q5/q5_shapes.h"

#include <stdexcept>

namespace ninfer::ops::detail {

// Unified-template routes, selected with complete-Op cold CUDA Graph measurements on an RTX 5090.

Q5Launch select_q5_n1024_k5120_unified(std::int32_t tokens) {
    if (tokens <= 1) return launch_q5_a16_direct_r1_t1_w4_k5120;
    if (tokens <= 8) return launch_q5_a16_direct_r2_t4_w2_g8_b4;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 32) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 64) return launch_q5_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 80) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 112) return launch_q5_a16_sliced_r32_t24_w4_s2_pairwise;
    if (tokens <= 160) return launch_q5_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 480) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 640) return launch_q5_a16_sliced_r32_t64_w2_s2;
    if (tokens <= 768) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 1280) return launch_q5_a16_mma_r32_t128;
    if (tokens <= 1344) return launch_q5_a16_sliced_r32_t64_w2_s1;
    return launch_q5_a16_mma_r64_t128;
}

Q5Launch select_q5_n1152_k1152_unified(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("q5 linear: T must be a multiple of 4 in [4,131072]");
    if (tokens <= 8) return launch_q5_a16_direct_r2_t4_w2_g8_b4;
    if (tokens <= 64) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 128) return launch_q5_a16_sliced_r16_t16_w2_s2;
    if (tokens <= 576) return launch_q5_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 1152) return launch_q5_a16_mma_r32_t128;
    return launch_q5_a16_mma_r64_t128;
}

Q5Launch select_q5_n1152_k4304_unified(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("q5 linear: T must be a multiple of 4 in [4,131072]");
    if (tokens <= 4) return launch_q5_a16_direct_r2_t4_w4_g4_b4;
    if (tokens <= 8) return launch_q5_a16_direct_r2_t4_w2_g8_b4;
    if (tokens <= 32) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 48) return launch_q5_a16_sliced_r16_t24_w4_s2;
    if (tokens <= 64) return launch_q5_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 96) return launch_q5_a16_sliced_r32_t24_w4_s2_pairwise;
    if (tokens <= 288) return launch_q5_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 448) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 576) return launch_q5_a16_sliced_r32_t64_w2_s2;
    if (tokens <= 672) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 1152) return launch_q5_a16_mma_r32_t128;
    if (tokens <= 1536) return launch_q5_a16_mma_r64_t96_k128_s1_a1;
    return launch_q5_a16_mma_r64_t128;
}

Q5Launch select_q5_n5120_k17408_unified(std::int32_t tokens) {
    if (tokens <= 1) return launch_q5_a16_direct_r1_t1_w4_k17408;
    if (tokens <= 2) return launch_q5_a16_direct_r1_t2_w2_k17408;
    if (tokens <= 3) return launch_q5_a16_direct_r1_t3_w2_k17408;
    if (tokens <= 4) return launch_q5_a16_direct_r1_t4_w2_k17408;
    if (tokens <= 8) return launch_q5_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 24) return launch_q5_a16_sliced_r32_t24_w4_s2_pairwise;
    if (tokens <= 32) return launch_q5_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 48) return launch_q5_a16_sliced_r32_t24_w4_s2_pairwise;
    if (tokens <= 128) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 160) return launch_q5_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 256) return launch_q5_a16_mma_r32_t128;
    return launch_q5_a16_mma_r64_t128;
}

Q5Launch select_q5_n5120_k6144_unified(std::int32_t tokens) {
    if (tokens <= 1) return launch_q5_a16_direct_r1_t1_w4_k6144;
    if (tokens <= 2) return launch_q5_a16_direct_r1_t2_w2_k6144;
    if (tokens <= 3) return launch_q5_a16_direct_r1_t3_w2_k6144;
    if (tokens <= 4) return launch_q5_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q5_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 32) return launch_q5_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 48) return launch_q5_a16_sliced_r32_t24_w4_s2_pairwise;
    if (tokens <= 64) return launch_q5_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 128) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 160) return launch_q5_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 256) return launch_q5_a16_mma_r32_t128;
    return launch_q5_a16_mma_r64_t128;
}

Q5Launch select_q5_n6144_k5120_unified(std::int32_t tokens) {
    if (tokens <= 1) return launch_q5_a16_direct_r1_t1_w4_k5120;
    if (tokens <= 2) return launch_q5_a16_direct_r1_t2_w4_k5120;
    if (tokens <= 3) return launch_q5_a16_direct_r1_t3_w4_k5120;
    if (tokens <= 4) return launch_q5_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q5_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 24) return launch_q5_a16_sliced_r16_t24_w4_s2;
    if (tokens <= 32) return launch_q5_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 160) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 192) return launch_q5_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 384) return launch_q5_a16_mma_r64_t128;
    if (tokens <= 448) return launch_q5_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 672) return launch_q5_a16_mma_r64_t96_k128_s1_a1;
    return launch_q5_a16_mma_r64_t128;
}

Q5Launch select_q5_n7168_k5120_unified(std::int32_t tokens) {
    if (tokens <= 1) return launch_q5_a16_direct_r1_t1_w4_k5120;
    if (tokens <= 2) return launch_q5_a16_direct_r1_t2_w4_k5120;
    if (tokens <= 4) return launch_q5_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q5_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 24) return launch_q5_a16_sliced_r16_t24_w4_s2;
    if (tokens <= 32) return launch_q5_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 96) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 128) return launch_q5_a16_mma_r32_t128;
    if (tokens <= 192) return launch_q5_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 384) return launch_q5_a16_mma_r64_t128;
    if (tokens <= 576) return launch_q5_a16_mma_r64_t96_k128_s1_a1;
    return launch_q5_a16_mma_r64_t128;
}

Q5Launch select_q5_n1152_k4352_unified(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("q5 linear: T must be a multiple of 4 in [4,131072]");
    if (tokens <= 4) return launch_q5_a16_direct_r2_t4_w4_g4_b4;
    if (tokens <= 8) return launch_q5_a16_direct_r2_t4_w2_g8_b4;
    if (tokens <= 32) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 48) return launch_q5_a16_sliced_r16_t24_w4_s2;
    if (tokens <= 64) return launch_q5_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 96) return launch_q5_a16_sliced_r32_t24_w4_s2_pairwise;
    if (tokens <= 288) return launch_q5_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 448) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 576) return launch_q5_a16_sliced_r32_t64_w2_s2;
    if (tokens <= 672) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 1152) return launch_q5_a16_mma_r32_t128;
    if (tokens <= 1536) return launch_q5_a16_mma_r64_t96_k128_s1_a1;
    return launch_q5_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
