#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"

namespace ninfer::ops::detail {
void nvfp4_gdn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) {
        nvfp4_gdn_input_decode_launch(x, weight, qkv, z, stream);
        return;
    }
    if (tokens <= 2) {
        nvfp4_gdn_input_small_t_launch(x, weight, qkv, z, stream);
        return;
    }
    const auto p        = nvfp4_a16_operands(x, weight);
    const auto output   = Nvfp4GdnInputOutput{static_cast<__nv_bfloat16*>(qkv.data),
                                            static_cast<__nv_bfloat16*>(z.data)};
    const auto epilogue = LinearIdentityEpilogue{};
    if (tokens <= 8) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<8, 8, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 16) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<16, 8, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 24) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 8, 1>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 32) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 8, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 64) {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 96) {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
}
} // namespace ninfer::ops::detail
