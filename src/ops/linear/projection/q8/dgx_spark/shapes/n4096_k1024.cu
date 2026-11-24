#include "ops/linear/projection/q8/shapes.h"
#include "ops/linear/projection/q8/dgx_spark/launch.h"
#include "ops/linear/kernels/q8/q8_gemv_launch.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_mma_launch.cuh"
#include "ops/linear/kernels/q8/q8_a16_tma.cuh"
#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
using Small8   = Q8A16SlicedKMmaSchedule<8, 8, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                         Q8ActivationStage::PaddedZero, 1024>;
using Small16  = Q8A16SlicedKMmaSchedule<16, 8, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                         Q8ActivationStage::PaddedZero, 1024>;
using Small40  = Q8A16SlicedKMmaSchedule<40, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                         Q8ActivationStage::PaddedZero, 1024>;
using Small32  = Q8A16MmaSchedule<32, 32, 128, 16, 16, 2, 2, Q8MmaFragmentPipeline::PingPong,
                                  Cache::cg, Cache::cg, Cache::cg, false, 1024, 1, true>;
using Hot32    = Q8A16TmaMmaSchedule<64, 32, 32, 16, 3, 8, 1024, 64, true, true, true, true>;
using Medium64 = Q8A16TmaMmaSchedule<128, 64, 64, 16, 2, 8, 1024, 64, true, true, true>;
using Medium96 = Q8A16TmaMmaSchedule<64, 96, 64, 16, 2, 8, 1024, 64, true, true, true>;
using Bulk128  = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 8, 1024, 64, true, true, true>;

void gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_gemv<Q8A16GemvSchedule<4, 1, 4, 1024, 4>>(
        q8_linear_operands(x, weight),
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

void small32(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_mma<Small32>(q8_linear_operands(x, weight),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                               Bf16MmaCtaWarpStoreEpilogue<>{}, stream);
}

template <class Schedule>
void tma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_tma_mma<Schedule>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        Bf16MmaCtaWarpStoreEpilogue<>{}, stream);
}
} // namespace

Q8Launch select_q8_n4096_k1024(std::int32_t tokens) {
    if (tokens == 1) return gemv;
    if (tokens <= 8) return sliced<Small8>;
    if (tokens <= 16) return sliced<Small16>;
    if (tokens <= 32) return small32;
    if (tokens <= 40) return sliced<Small40>;
    if (tokens <= 64) return tma<Hot32>;
    if (tokens <= 96) return tma<Medium96>;
    if (tokens <= 128) return tma<Bulk128>;
    if (tokens <= 192) return tma<Medium64>;
    if (tokens <= 288) return tma<Medium96>;
    if (tokens <= 512) return tma<Bulk128>;
    if (tokens <= 576) return tma<Medium96>;
    return launch_q8_n4096_k1024_bulk;
}
} // namespace ninfer::ops::detail
