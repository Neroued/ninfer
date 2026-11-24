#include "ops/linear_attention/gated_delta_net/input_conv/nvfp4/launch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
void launch_sliced(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                   cudaStream_t stream) {
    launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Schedule, 5120>>(
        nvfp4_a16_operands(x, weight),
        LinearBf16SegmentedOutput<10240, 6144>{static_cast<__nv_bfloat16*>(qkv.data),
                                               static_cast<__nv_bfloat16*>(z.data)},
        LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
void launch_matrix(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                   cudaStream_t stream) {
    launch_nvfp4_a16_mma<Nvfp4ScheduleInstance<Schedule, 5120>>(
        nvfp4_a16_operands(x, weight),
        LinearBf16SegmentedOutput<10240, 6144>{static_cast<__nv_bfloat16*>(qkv.data),
                                               static_cast<__nv_bfloat16*>(z.data)},
        LinearIdentityEpilogue{}, stream);
}

using Launcher = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);

Launcher select_projection(int tokens) {
    if (tokens == 1) return launch_nvfp4_gdn_input_decode;
    if (tokens <= 2) return launch_nvfp4_gdn_input_small_t;
    if (tokens <= 8) return launch_sliced<Nvfp4SlicedInstance<8, 8, 2>>;
    if (tokens <= 16) return launch_sliced<Nvfp4SlicedInstance<16, 8, 2>>;
    if (tokens <= 24) return launch_sliced<Nvfp4SlicedInstance<32, 8, 1>>;
    if (tokens <= 32) return launch_sliced<Nvfp4SlicedInstance<32, 8, 2>>;
    if (tokens <= 64) return launch_matrix<Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>>;
    if (tokens <= 96) return launch_matrix<Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>;
    return launch_matrix<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>;
}
} // namespace

void launch_nvfp4_gdn_input_a16(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream) {
    select_projection(x.ne[1])(x, weight, qkv, z, stream);
}
} // namespace ninfer::ops::detail
