#include "ops/linear/swiglu/q4/launch.h"
#include "ops/linear/swiglu/dispatch.h"
#include "ops/linear/kernels/q4/q4_launch.h"

namespace ninfer::ops::detail {
namespace {
LinearSwiGluLaunch select_projection(int tokens) {
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
} // namespace

void launch_q4_swiglu_materialized_projection(const Tensor& x, const Weight& w, Tensor& out,
                                              cudaStream_t stream) {
    select_projection(x.ne[1])(x, w, out, stream);
}
} // namespace ninfer::ops::detail
