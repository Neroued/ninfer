#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_gemv_launch.cuh"
#include "ops/linear/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Gemv    = Q8A16GemvSchedule<4, 1, 2, 6144, 4>;
using Grouped = Q8A16GroupedSlicedKMmaSchedule<32, 4, 2, 1, 1, 6144, Cache::cg, Cache::ca, true>;
using Mma     = Q8A16MmaSchedule<32, 32, 128, 16, 16, 2, 3>;

void gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_gemv<Gemv>(q8_linear_operands(x, weight),
                             LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                             LinearIdentityEpilogue{}, stream);
}

void grouped(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_grouped_sliced_k_mma<Grouped>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

void mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_mma<Mma>(q8_linear_operands(x, weight),
                           LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                           LinearIdentityEpilogue{}, stream);
}
} // namespace

Q8Launch select_q8_n2560_k6144(std::int32_t tokens) {
    if (tokens == 1) return gemv;
    if (tokens <= 32) return launch_q8_a16_sliced_r16_t16_w8_s2;
    if (tokens <= 50) return launch_q8_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 64) return grouped;
    if (tokens <= 128) return mma;
    return launch_q8_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
