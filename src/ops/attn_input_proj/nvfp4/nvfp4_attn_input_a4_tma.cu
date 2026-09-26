#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"

namespace ninfer::ops::detail {
void launch_nvfp4_a4_tma_attention(const Nvfp4A4Operands& p, __nv_bfloat16* query,
                                   __nv_bfloat16* gate, __nv_bfloat16* key, __nv_bfloat16* value,
                                   cudaStream_t stream) {
    const Nvfp4AttentionInputOutput output{query, key, gate, value};
    if (p.scale_layout == Nvfp4ScaleLayout::Tiled128)
        launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<128, 4, 1>, 5120>>(
            p, output, LinearIdentityEpilogue{}, stream);
    else
        launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1>, 5120>>(
            p, output, LinearIdentityEpilogue{}, stream);
}
} // namespace ninfer::ops::detail
