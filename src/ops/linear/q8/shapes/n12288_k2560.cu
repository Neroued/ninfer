#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_gemv_launch.cuh"
#include "ops/linear/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Gemv      = Q8A16GemvSchedule<8, 1, 2, 2560, 1>;
using Small     = Q8A16SlicedKMmaSchedule<16, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Medium    = Q8A16MmaSchedule<32, 32, 128, 16, 16, 2, 3>;
using Group32   = Q8A16GroupedSlicedKMmaSchedule<32, 2, 2, 1, 1, 2560, Cache::cg, Cache::ca, true>;
using Mid1      = Q8A16SlicedKMmaSchedule<40, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Mid2      = Q8A16SlicedKMmaSchedule<40, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Mma80     = Q8A16MmaSchedule<32, 80, 64, 32, 16, 2, 3>;
using Mma80Full = Q8A16MmaSchedule<48, 80, 64, 48, 16, 2, 3>;
using Mma96     = Q8A16MmaSchedule<48, 96, 64, 48, 16, 2, 2>;
using Mma112    = Q8A16MmaSchedule<48, 112, 64, 48, 16, 2, 2>;
using Mma112Serial = Q8A16MmaSchedule<48, 112, 64, 16, 16, 2, 2, Q8MmaFragmentPipeline::Serial>;
using Mma128       = Q8A16MmaSchedule<48, 128, 64, 16, 16, 2, 2, Q8MmaFragmentPipeline::Serial>;
using Bulk = Q8A16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Q8MmaFragmentPipeline::Serial, Cache::cg,
                              Cache::cg, Cache::ca, false, 2560, 20>;

void gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_gemv<Gemv>(q8_linear_operands(x, weight),
                             LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                             LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<Schedule>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

void grouped(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_grouped_sliced_k_mma<Group32>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_mma<Schedule>(q8_linear_operands(x, weight),
                                LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                LinearIdentityEpilogue{}, stream);
}
} // namespace

Q8Launch select_q8_n12288_k2560(std::int32_t tokens) {
    // Full token tiles select different kernels from tails; retain their measured winners.
    // The 34-column single-stage sliced schedule also has a repeatable advantage.
    if (tokens == 1) return gemv;
    if (tokens <= 16) return sliced<Small>;
    if (tokens == 32) return grouped;
    if (tokens < 32) return mma<Medium>;
    if (tokens == 34) return sliced<Mid1>;
    if (tokens <= 40) return sliced<Mid2>;
    if (tokens == 64) return launch_q8_a16_mma_r48_t64;
    if (tokens < 64) return launch_q8_a16_mma_r32_t64;
    if (tokens == 80) return mma<Mma80Full>;
    if (tokens < 80) return mma<Mma80>;
    if (tokens <= 96) return mma<Mma96>;
    if (tokens >= 104 && tokens <= 110) return mma<Mma112Serial>;
    if (tokens <= 112) return mma<Mma112>;
    if (tokens <= 128) return mma<Mma128>;
    return mma<Bulk>;
}
} // namespace ninfer::ops::detail
