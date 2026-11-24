#include "ops/linear/projection/q8/shapes.h"
#include "ops/linear/projection/q8/dgx_spark/launch.h"
#include "ops/linear/kernels/q8/q8_simt_launch.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_grouped_sliced_k_launch.cuh"
#include "ops/linear/kernels/q8/q8_a16_tma.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Base>
struct Q8N12288K4096Schedule : Base {};

using Simt4   = Q8A16SimtSchedule<4, 4, 2, 16, 2, Cache::cg, 2>;
using Simt8   = Q8A16SimtSchedule<8, 8, 1, 32, 2, Cache::cg, 1>;
using Small32 = Q8A16SlicedKMmaSchedule<32, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 4096>;
using Group16 = Q8A16GroupedSlicedKMmaSchedule<16, 2, 2, 2, 1, 4096, Cache::cg, Cache::ca, true>;
using Group32 = Q8A16GroupedSlicedKMmaSchedule<32, 2, 2, 1, 1, 4096, Cache::cg, Cache::ca, true>;
using Small40 = Q8A16SlicedKMmaSchedule<40, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 4096>;
using Hot64   = Q8A16TmaMmaSchedule<64, 64, 64, 16, 3, 8, 4096>;
using Hot96   = Q8A16TmaMmaSchedule<64, 96, 64, 16, 2, 8, 4096>;
using Bulk128 = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 8, 4096>;

template <class Schedule>
void simt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_simt<Schedule>(q8_linear_operands(x, weight),
                                 LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
                                 LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void sliced(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<Q8N12288K4096Schedule<Schedule>>(
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

template <class Schedule>
void tma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_q8_a16_tma_mma<Q8N12288K4096Schedule<Schedule>>(
        q8_linear_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), out.ne[0]},
        LinearIdentityEpilogue{}, stream);
}
} // namespace

Q8Launch select_q8_n12288_k4096(std::int32_t tokens) {
    if (tokens <= 4) return simt<Simt4>;
    if (tokens <= 8) return simt<Simt8>;
    if (tokens <= 32) return sliced<Small32>;
    if (tokens <= 40) return sliced<Small40>;
    if (tokens <= 48) return grouped<Group16>;
    if (tokens <= 64) return grouped<Group32>;
    if (tokens <= 96) return tma<Hot64>;
    if (tokens <= 192) return tma<Hot96>;
    if (tokens >= 1024 && tokens % 128 == 0) return launch_q8_n12288_k4096_bulk_full;
    return tma<Bulk128>;
}
} // namespace ninfer::ops::detail
