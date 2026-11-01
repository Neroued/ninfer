#include "ops/linear/projection/q8/shapes.h"
#include "ops/linear/projection/q8/dgx_spark/launch.h"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_gemv_launch.cuh"
#include "ops/linear/kernels/q8/q8_a16_tma.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
using Small8     = Q8A16SlicedKMmaSchedule<8, 16, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                           Q8ActivationStage::PaddedZero, 4096>;
using Small16    = Q8A16SlicedKMmaSchedule<16, 8, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                           Q8ActivationStage::PaddedZero, 4096>;
using Small40    = Q8A16SlicedKMmaSchedule<40, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                           Q8ActivationStage::PaddedZero, 4096>;
using Small48    = Q8A16SlicedKMmaSchedule<48, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                           Q8ActivationStage::PaddedZero, 4096>;
using Hot64K128  = Q8A16TmaMmaSchedule<64, 64, 32, 32, 1, 8, 4096, 128, true, true, true>;
using Hot128K128 = Q8A16TmaMmaSchedule<64, 128, 32, 32, 1, 8, 4096, 128, true, true, true>;
using Hot64      = Q8A16TmaMmaSchedule<64, 64, 64, 16, 3, 8, 4096, 64, false, false, true>;
using Medium96   = Q8A16TmaMmaSchedule<64, 96, 64, 16, 2, 8, 4096, 64, false, false, true>;
using Bulk128    = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 8, 4096, 64, true, true, true>;

template <class Schedule>
void sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<Schedule>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

void gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_gemv<Q8A16GemvSchedule<4, 4, 2, 4096, 4>>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule, class Epilogue = LinearIdentityEpilogue>
void tma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_tma_mma<Schedule>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]}, Epilogue{}, stream);
}
} // namespace

Q8Launch select_q8_n2048_k4096(std::int32_t tokens) {
    if (tokens == 1) return gemv;
    if (tokens <= 8) return sliced<Small8>;
    if (tokens <= 16) return sliced<Small16>;
    if (tokens <= 40) return sliced<Small40>;
    if (tokens <= 48) return sliced<Small48>;
    if (tokens <= 64) return tma<Hot64K128, Bf16MmaCtaWarpStoreEpilogue<>>;
    if (tokens <= 128) return tma<Hot128K128, Bf16MmaCtaWarpStoreEpilogue<>>;
    if (tokens <= 192) return tma<Hot64>;
    if (tokens <= 576) return tma<Medium96>;
    if (tokens >= 2048) return launch_q8_n2048_k4096_bulk;
    return tma<Bulk128, Bf16MmaCtaWarpStoreEpilogue<>>;
}
} // namespace ninfer::ops::detail
