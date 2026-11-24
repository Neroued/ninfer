#include "ops/moe/swiglu/dispatch.h"
#include "ops/moe/swiglu/kernel/output.cuh"
#include "ops/linear/swiglu/bf16/paired.cuh"

namespace ninfer::ops::detail::swiglu_ffn {
namespace {
template <class Schedule>
void tma(const Bf16A16Operands& p, SwiGluCompensatedOutput out, float limit, cudaStream_t stream) {
    launch_swiglu_bf16_tma<Bf16ScheduleInstance<Schedule, 4096>, 1024>(
        p, out, ClampedSwiGluProduct{limit}, stream);
}
} // namespace

void launch_bf16_gu_float(const Tensor& x, const Weight& w, float* values, float limit,
                          cudaStream_t stream) {
    const auto p = bf16_a16_operands(x, w);
    const SwiGluFloatOutput out{values, 1024};
    const ClampedSwiGluProduct product{limit};
    if (p.tokens == 1) {
        swiglu_bf16_paired_gemv<4096, 1024><<<256, 128, 0, stream>>>(p.x, p.weight, out, product);
        CUDA_CHECK(cudaGetLastError());
    } else if (p.tokens <= 8) {
        using S = SwiGluBf16SlicedSchedule<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 2>, 4096>, 1024>;
        launch_bf16_a16_sliced_k_mma<S>(
            p, out, SwiGluBf16FragmentEpilogue<ClampedSwiGluProduct>{product}, stream);
    } else {
        using S = SwiGluBf16SlicedSchedule<
            Bf16ScheduleInstance<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 2>, 4096>, 1024>;
        launch_bf16_a16_sliced_k_mma<S>(
            p, out, SwiGluBf16FragmentEpilogue<ClampedSwiGluProduct>{product}, stream);
    }
}

void launch_bf16_gu_compensated(const Tensor& x, const Weight& w, __nv_bfloat16* planes,
                                float limit, cudaStream_t stream) {
    const auto p = bf16_a16_operands(x, w);
    const SwiGluCompensatedOutput out{planes, 1024, p.tokens};
    if (p.tokens <= 32)
        tma<Bf16A16TmaMmaSchedule<32, 16, 128, 16, 8, 2, 2>>(p, out, limit, stream);
    else if (p.tokens <= 64)
        tma<Bf16A16TmaMmaSchedule<32, 32, 128, 16, 16, 2, 2>>(p, out, limit, stream);
    else if (p.tokens <= 96)
        tma<Bf16A16TmaMmaSchedule<64, 32, 128, 32, 8, 2, 2>>(p, out, limit, stream);
    else if (p.tokens <= 128)
        tma<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 2, 2>>(p, out, limit, stream);
    else if (p.tokens <= 512)
        tma<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 16, 2, 1>>(p, out, limit, stream);
    else if (p.tokens <= 2048)
        tma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 2, 1>>(p, out, limit, stream);
    else
        tma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 2, 1, Bf16MmaRaster::RowFast>>(
            p, out, limit, stream);
}
} // namespace ninfer::ops::detail::swiglu_ffn
