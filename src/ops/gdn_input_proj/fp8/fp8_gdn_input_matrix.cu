#include "ops/linear/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_schedule.cuh"
#include "ops/linear/fp8/fp8_a16_simt.cuh"
#include "ops/linear/fp8/fp8_a16_sliced_k_mma.cuh"
#include "ops/linear/fp8/fp8_a16_mma.cuh"

namespace ninfer::ops::detail {
namespace {

using Geometry = Fp8N16384K5120;

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                  cudaStream_t stream) {
    using Schedule =
        Fp8A16SimtSchedule<8, 2, (ActiveTokens >= 5 && ActiveTokens <= 6) ? 8 : 16, ActiveTokens, 1,
                           ActiveTokens <= 4 ? Fp8SimtActivationAccess::SharedPhase
                                             : Fp8SimtActivationAccess::TokenPacked,
                           Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
    const Fp8GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                   static_cast<__nv_bfloat16*>(z.data)};
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, Geometry::kInputRows, ActiveTokens, true>>(
        fp8_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

template <int Capacity>
void launch_small_mma(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                      cudaStream_t stream) {
    constexpr int warps = Capacity <= 8 ? 16 : Capacity <= 24 ? 8 : 4;
    using Schedule      = Fp8A16SlicedKMmaSchedule<warps, Capacity, warps == 16 ? 1 : 2>;
    const Fp8GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                   static_cast<__nv_bfloat16*>(z.data)};
    launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows, Capacity>>(
        fp8_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_gemm(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                 cudaStream_t stream) {
    static_assert(10240 % Schedule::kBlockRows == 0);
    static_assert(6144 % Schedule::kBlockRows == 0);
    static_assert(Schedule::kSharedBytes <= 48 * 1024);
    const Fp8GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                   static_cast<__nv_bfloat16*>(z.data)};
    launch_fp8_a16_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

} // namespace

void fp8_gdn_input_matrix_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream) {
    // SIMT for the latency regime, bounded MMA column capacities, then amortized weight decode.
    const int columns = x.ne[1];
    if (columns == 2) return launch_exact<2>(x, weight, qkv, z, stream);
    if (columns == 3) return launch_exact<3>(x, weight, qkv, z, stream);
    if (columns == 4) return launch_exact<4>(x, weight, qkv, z, stream);
    if (columns <= 8) return launch_small_mma<8>(x, weight, qkv, z, stream);
    if (columns <= 16) return launch_small_mma<16>(x, weight, qkv, z, stream);
    if (columns <= 24) return launch_small_mma<24>(x, weight, qkv, z, stream);
    if (columns <= 32) return launch_small_mma<32>(x, weight, qkv, z, stream);
    if (columns <= 64)
        return launch_gemm<Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(x, weight, qkv, z, stream);
    if (columns <= 96)
        return launch_gemm<Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(x, weight, qkv, z, stream);
    return launch_gemm<Fp8A16MmaSchedule<64, 128, 64, 32, 16, 2, 2>>(x, weight, qkv, z, stream);
}

} // namespace ninfer::ops::detail
