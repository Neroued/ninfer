#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/nvfp4/nvfp4_instances.cuh"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"

namespace ninfer::ops::detail {
void nvfp4_attn_input_a16_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                 Tensor& k, Tensor& v, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) {
        nvfp4_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (tokens <= 2) {
        nvfp4_attn_input_small_t_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    const auto p      = nvfp4_a16_operands(x, weight);
    const auto output = Nvfp4AttentionInputOutput{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
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
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 1>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 32) {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 32, 128, 16, 16, 2, 2>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 64) {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>, 5120>>(
            p, output, epilogue, stream);
        return;
    }
    if (tokens <= 128) {
        launch_nvfp4_a16_mma<
            Nvfp4ScheduleInstance<Nvfp4A16MmaSchedule<64, 64, 64, 32, 16, 2, 2>, 5120>>(
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
