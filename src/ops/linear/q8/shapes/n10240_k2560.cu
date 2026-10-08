#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_gemv_launch.cuh"
#include "ops/linear/q8/q8_mma_launch.cuh"
#include "ops/linear/q8/q8_sliced_k_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using S0  = Q8A16GemvSchedule<4, 1, 2, 2560, 4>;
using S1  = Q8A16SlicedKMmaSchedule<16, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                    Q8ActivationStage::PaddedZero, 2560>;
using S2  = Q8A16SlicedKMmaSchedule<24, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                    Q8ActivationStage::PaddedZero, 2560>;
using S3  = Q8A16SlicedKMmaSchedule<32, 2, 2, 2, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                    Q8ActivationStage::PaddedZero, 2560>;
using S4  = Q8A16SlicedKMmaSchedule<40, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                    Q8ActivationStage::PaddedZero, 2560>;
using S5  = Q8A16SlicedKMmaSchedule<32, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                    Q8ActivationStage::PaddedZero, 2560>;
using S6  = Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>;
using S7  = Q8A16MmaSchedule<32, 80, 64, 32, 16, 2, 3, Q8MmaFragmentPipeline::PingPong>;
using S8  = Q8A16MmaSchedule<64, 48, 128, 16, 24, 1, 2>;
using S9  = Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2, Q8MmaFragmentPipeline::PingPong, Cache::cg,
                             Cache::cg, Cache::cg>;
using S10 = Q8A16MmaSchedule<64, 128, 64, 32, 16, 2, 2, Q8MmaFragmentPipeline::Serial>;
using S11 = Q8A16MmaSchedule<64, 128, 64, 32, 32, 2, 2, Q8MmaFragmentPipeline::Serial, Cache::cg,
                             Cache::cg, Cache::ca, false, 2560, 20>;

template <class Schedule>
void gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_gemv<Schedule>(q8_linear_operands(x, weight),
                                 LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                 LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_mma<Schedule>(q8_linear_operands(x, weight),
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
} // namespace

Q8Launch select_q8_n10240_k2560(std::int32_t tokens) {
    if (tokens <= 1) return gemv<S0>;
    if (tokens <= 16) return sliced<S1>;
    if (tokens <= 24) return sliced<S2>;
    if (tokens <= 32) return sliced<S3>;
    if (tokens <= 40) return sliced<S4>;
    if (tokens <= 48) return sliced<S5>;
    if (tokens <= 64) return mma<S6>;
    if (tokens <= 67) return mma<S7>;
    if (tokens <= 96) return mma<S8>;
    if (tokens <= 127) return mma<S9>;
    if (tokens <= 128) return mma<S10>;
    return mma<S11>;
}
} // namespace ninfer::ops::detail
