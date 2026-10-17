#include "ops/linear/projection/q8/shapes.h"
#include "ops/linear/projection/q8/dgx_spark/launch.h"
#include "ops/linear/kernels/q8/q8_gemv_launch.cuh"
#include "ops/linear/kernels/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_mma_launch.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_a16_tma.cuh"

#include "ops/linear/kernels/q8/q8_instances.cuh"

namespace ninfer::ops::detail {
namespace {

using Gemv    = Q8A16GemvSchedule<8, 1, 2, 2560, 1>;
using Small   = Q8A16SlicedKMmaSchedule<16, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 2560>;
using Medium  = Q8A16MmaSchedule<32, 32, 128, 16, 16, 2, 3>;
using Group32 = Q8A16GroupedSlicedKMmaSchedule<32, 2, 2, 1, 1, 2560, Cache::cg, Cache::ca, true>;
using Mid1    = Q8A16SlicedKMmaSchedule<40, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 2560>;
using Mid2    = Q8A16SlicedKMmaSchedule<40, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 2560>;

template <class Base>
struct Q8N12288K2560Schedule : Base {};

using Hot96   = Q8A16TmaMmaSchedule<64, 96, 64, 16, 2, 8, 2560>;
using Bulk128 = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 8, 2560>;

template <class Schedule>
void tma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_tma_mma<Q8N12288K2560Schedule<Schedule>>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_gemv<Q8N12288K2560Schedule<Schedule>>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<Q8N12288K2560Schedule<Schedule>>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void grouped(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_grouped_sliced_k_mma<Q8N12288K2560Schedule<Schedule>>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_mma<Q8N12288K2560Schedule<Schedule>>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

} // namespace

Q8Launch select_q8_n12288_k2560(std::int32_t tokens) {
    if (tokens == 1) return gemv<Gemv>;
    if (tokens <= 16) return sliced<Small>;
    if (tokens == 32) return grouped<Group32>;
    if (tokens < 32) return mma<Medium>;
    if (tokens == 34) return sliced<Mid1>;
    if (tokens <= 40) return sliced<Mid2>;
    if (tokens == 64) return mma<q8_instances::MmaR48T64>;
    if (tokens < 64) return mma<q8_instances::MmaR32T64>;
    if (tokens >= 1024 && tokens % 128 == 0) return launch_q8_n12288_k2560_bulk_full;
    if (tokens <= 96 || (tokens > 128 && tokens <= 192)) return tma<Hot96>;
    return tma<Bulk128>;
}

} // namespace ninfer::ops::detail
