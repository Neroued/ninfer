#include "ops/linear/projection/q8/shapes.h"
#include "ops/linear/projection/q8/dgx_spark/launch.h"
#include "ops/linear/kernels/q8/q8_gemv_launch.cuh"
#include "ops/linear/kernels/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_a16_tma.cuh"

namespace ninfer::ops::detail {
namespace {
using Gemv      = Q8A16GemvSchedule<4, 2, 2, 3072, 4>;
using Small8    = Q8A16SlicedKMmaSchedule<8, 8, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 3072>;
using Small16   = Q8A16SlicedKMmaSchedule<16, 8, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 3072>;
using Small24   = Q8A16SlicedKMmaSchedule<24, 8, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 3072>;
using Group16   = Q8A16GroupedSlicedKMmaSchedule<16, 2, 2, 2, 1, 3072, Cache::cg, Cache::ca, true>;
using Tma64K128 = Q8A16TmaMmaSchedule<32, 64, 16, 16, 2, 0, 3072, 128>;
using Hot64     = Q8A16TmaMmaSchedule<64, 64, 64, 16, 3, 8, 3072>;
using Hot96     = Q8A16TmaMmaSchedule<64, 96, 64, 16, 2, 8, 3072>;
using Bulk128   = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 8, 3072>;

template <class Schedule>
void gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_gemv<Schedule>(q8_linear_operands(x, weight),
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

template <class Schedule>
void tma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_tma_mma<Schedule>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void grouped(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_grouped_sliced_k_mma<Schedule>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}
} // namespace

Q8Launch select_q8_n2560_k3072(std::int32_t tokens) {
    if (tokens == 1) return gemv<Gemv>;
    if (tokens <= 8) return sliced<Small8>;
    if (tokens <= 16) return sliced<Small16>;
    if (tokens <= 24) return sliced<Small24>;
    if (tokens <= 32) return grouped<Group16>;
    if (tokens <= 64) return tma<Tma64K128>;
    if (tokens <= 96) return tma<Hot64>;
    if (tokens <= 128) return tma<Bulk128>;
    if (tokens <= 192) return tma<Hot96>;
    if (tokens > 512 && tokens <= 576) return tma<Hot96>;
    if (tokens == 512 || (tokens >= 2048 && tokens % 128 == 0))
        return launch_q8_n2560_k3072_bulk_full;
    return tma<Bulk128>;
}
} // namespace ninfer::ops::detail
