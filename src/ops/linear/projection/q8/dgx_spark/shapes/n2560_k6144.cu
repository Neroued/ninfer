#include "ops/linear/kernels/q8/q8_a16_tma.cuh"
#include "ops/linear/kernels/q8/q8_gemv_launch.cuh"
#include "ops/linear/kernels/q8/q8_launch.h"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/projection/q8/shapes.h"
#include "ops/linear/projection/q8/dgx_spark/launch.h"

namespace ninfer::ops::detail {
namespace {
using Decode  = Q8A16GemvSchedule<4, 1, 2, 6144, 4>;
using Small16 = Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::RuntimeActive, 6144>;

// Give this shape its own CUDA entry.
struct Small32 : Q8A16SlicedKMmaSchedule<32, 8, 1, 2, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                         Q8ActivationStage::RuntimeActive, 6144> {};

struct Hot64 : Q8A16TmaMmaSchedule<32, 64, 16, 16, 1, 0, 6144, 128> {};

struct Hot128 : Q8A16TmaMmaSchedule<64, 128, 32, 32, 1, 0, 6144, 128> {};

using Bulk  = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 8, 6144>;
using Large = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 0, 6144>;

void decode(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_q8_a16_gemv<Decode>(q8_linear_operands(x, w),
                               LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), y.ne[0]},
                               LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void sliced(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_q8_a16_sliced_k_mma<Schedule>(
        q8_linear_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), y.ne[0]},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void tma(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    launch_q8_a16_tma_mma<Schedule>(q8_linear_operands(x, w),
                                    LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), y.ne[0]},
                                    LinearIdentityEpilogue{}, stream);
}
} // namespace

Q8Launch select_q8_n2560_k6144(std::int32_t tokens) {
    if (tokens == 1) return decode;
    if (tokens <= 8) return launch_q8_a16_sliced_r16_t16_w8_s2;
    if (tokens <= 16) return sliced<Small16>;
    if (tokens <= 32) return sliced<Small32>;
    if (tokens <= 64) return tma<Hot64>;
    if (tokens <= 128) return tma<Hot128>;
    if (tokens == 512) return launch_q8_n2560_k6144_prefill512;
    if (tokens <= 2048) return tma<Bulk>;
    return tma<Large>;
}
} // namespace ninfer::ops::detail
