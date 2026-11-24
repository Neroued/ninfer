#include "ops/linear/kernels/fp8/fp8_template_launch.cuh"
#include "core/weight.h"
#include "ops/linear_attention/gated_delta_net/input_conv/fp8/launch.h"
#include "core/device.h"
#include "ops/linear/common/output.cuh"
#include "ops/linear/kernels/fp8/fp8_schedule.cuh"
#include "ops/linear/kernels/fp8/fp8_a16_simt.cuh"
#include "ops/linear/kernels/fp8/fp8_a16_sliced_k_mma.cuh"
#include "ops/linear/kernels/fp8/fp8_a16_mma.cuh"

namespace ninfer::ops::detail {
namespace {

using Geometry = Fp8N16384K5120;

template <int Capacity>
void launch_small_mma(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                      cudaStream_t stream) {
    constexpr int warps = Capacity <= 8 ? 16 : Capacity <= 24 ? 8 : 4;
    using Schedule      = Fp8A16SlicedKMmaSchedule<warps, Capacity, warps == 16 ? 1 : 2>;
    const LinearBf16SegmentedOutput<10240, 6144> output{static_cast<__nv_bfloat16*>(qkv.data),
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
    const LinearBf16SegmentedOutput<10240, 6144> output{static_cast<__nv_bfloat16*>(qkv.data),
                                                        static_cast<__nv_bfloat16*>(z.data)};
    launch_fp8_a16_mma<Fp8ScheduleInstance<Schedule, Geometry::kInputRows>>(
        fp8_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

void launch_simt4(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                  cudaStream_t stream) {
    using Schedule =
        Fp8A16SimtSchedule<8, 2, 16, 4, 1, Fp8SimtActivationAccess::SharedPhase,
                           Fp8CodeCache::Default, 1, Fp8SimtBlockOrder::RowsContiguous, 1>;
    const LinearBf16SegmentedOutput<10240, 6144> output{static_cast<__nv_bfloat16*>(qkv.data),
                                                        static_cast<__nv_bfloat16*>(z.data)};
    launch_fp8_a16_simt<Fp8ScheduleInstance<Schedule, 5120, 4>>(fp8_a16_operands(x, weight), output,
                                                                LinearIdentityEpilogue{}, stream);
}

void launch_sliced16(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                     cudaStream_t stream) {
    using Schedule =
        Fp8A16SlicedKMmaSchedule<4, 16, 7, Cache::ca, Cache::cg, Fp8ActivationStage::PaddedZero, 1>;
    const LinearBf16SegmentedOutput<10240, 6144> output{static_cast<__nv_bfloat16*>(qkv.data),
                                                        static_cast<__nv_bfloat16*>(z.data)};
    launch_fp8_a16_sliced_k_mma<Fp8ScheduleInstance<Schedule, 5120>>(
        fp8_a16_operands(x, weight), output, LinearIdentityEpilogue{}, stream);
}

using Launcher = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);

Launcher select_projection(int columns) {
    if (columns <= 4) return launch_simt4;
    if (columns <= 16) return launch_sliced16;
    if (columns <= 24) return launch_small_mma<24>;
    if (columns <= 32) return launch_small_mma<32>;
    if (columns <= 64) return launch_gemm<Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>;
    if (columns <= 96) return launch_gemm<Fp8A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>;
    return launch_gemm<Fp8A16MmaSchedule<64, 128, 64, 32, 16, 2, 2>>;
}
} // namespace

void launch_fp8_gdn_input_matrix(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream) {
    select_projection(x.ne[1])(x, weight, qkv, z, stream);
}
} // namespace ninfer::ops::detail
