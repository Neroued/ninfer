#include "ops/linear/projection/q8/shapes.h"
#include "ops/linear/projection/q8/dgx_spark/launch.h"
#include "ops/linear/kernels/q8/q8_gemv_launch.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_a16_tma.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Base>
struct Q8N8192K2560Schedule : Base {};

using Gemv    = Q8A16GemvSchedule<4, 2, 2, 2560, 4>;
using Small8  = Q8A16SlicedKMmaSchedule<8, 8, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 2560>;
using Small16 = Q8A16SlicedKMmaSchedule<16, 8, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 2560>;
using Small24 = Q8A16SlicedKMmaSchedule<24, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 2560>;
using Small40 = Q8A16SlicedKMmaSchedule<40, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 2560>;
using Hot32   = Q8A16TmaMmaSchedule<64, 32, 32, 16, 3, 8, 2560>;
using Hot64   = Q8A16TmaMmaSchedule<64, 64, 64, 16, 3, 8, 2560>;
using Hot96   = Q8A16TmaMmaSchedule<64, 96, 64, 16, 2, 8, 2560>;
using Bulk128 = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 8, 2560>;

template <class Schedule>
void gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_gemv<Q8N8192K2560Schedule<Schedule>>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<Q8N8192K2560Schedule<Schedule>>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void tma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_tma_mma<Q8N8192K2560Schedule<Schedule>>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}
} // namespace

Q8Launch select_q8_n8192_k2560(std::int32_t tokens) {
    if (tokens == 1) return gemv<Gemv>;
    if (tokens <= 8) return sliced<Small8>;
    if (tokens <= 16) return sliced<Small16>;
    if (tokens <= 24) return sliced<Small24>;
    if (tokens <= 40) return sliced<Small40>;
    if (tokens <= 96) return tma<Hot32>;
    if (tokens <= 128) return tma<Hot64>;
    if (tokens <= 192) return tma<Hot96>;
    if (tokens >= 1024 && tokens % 128 == 0) return launch_q8_n8192_k2560_bulk_full;
    return tma<Bulk128>;
}
} // namespace ninfer::ops::detail
