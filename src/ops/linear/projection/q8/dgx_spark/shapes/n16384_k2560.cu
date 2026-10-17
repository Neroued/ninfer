#include "ops/linear/projection/q8/shapes.h"
#include "ops/linear/projection/q8/dgx_spark/launch.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/linear/kernels/q8/q8_gemv_launch.cuh"
#include "ops/linear/kernels/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_mma_launch.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_a16_tma.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Base>
struct Q8N16384K2560Schedule : Base {};

using S0      = Q8A16GemvSchedule<8, 1, 2, 2560, 8>;
using S1      = Q8A16SlicedKMmaSchedule<8, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 2560>;
using S2      = Q8A16GroupedSlicedKMmaSchedule<16, 2, 2, 2, 1, 2560, Cache::cg, Cache::ca, true>;
using S3      = Q8A16SlicedKMmaSchedule<24, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 2560>;
using S4      = Q8A16SlicedKMmaSchedule<32, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 2560>;
using S5      = Q8A16SlicedKMmaSchedule<40, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 2560>;
using S6      = Q8A16MmaSchedule<64, 48, 128, 16, 24, 1, 2>;
using S7      = Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>;
using S8      = Q8A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>;
using Hot96   = Q8A16TmaMmaSchedule<64, 96, 64, 16, 2, 8, 2560>;
using Bulk128 = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 8, 2560>;

template <class Schedule>
void gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_gemv<Q8N16384K2560Schedule<Schedule>>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void grouped(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_grouped_sliced_k_mma<Q8N16384K2560Schedule<Schedule>>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_mma<Q8N16384K2560Schedule<Schedule>>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<Q8N16384K2560Schedule<Schedule>>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void tma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_tma_mma<Q8N16384K2560Schedule<Schedule>>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}
} // namespace

Q8Launch select_q8_n16384_k2560(std::int32_t tokens) {
    if (tokens <= 1) return gemv<S0>;
    if (tokens <= 8) return sliced<S1>;
    if (tokens <= 16) return grouped<S2>;
    if (tokens <= 24) return sliced<S3>;
    if (tokens <= 32) return sliced<S4>;
    if (tokens <= 40) return sliced<S5>;
    if (tokens <= 48) return mma<S6>;
    if (tokens <= 63) return mma<S7>;
    if (tokens <= 64) return mma<S8>;
    if (tokens <= 96) return tma<Hot96>;
    if (tokens <= 128) return tma<Bulk128>;
    // Avoid doubling the 128-token tile just after the first full tile.
    if (tokens <= 192) return tma<Hot96>;
    if (tokens >= 1024 && tokens % 128 == 0) return launch_q8_n16384_k2560_bulk_full;
    return tma<Bulk128>;
}
} // namespace ninfer::ops::detail
