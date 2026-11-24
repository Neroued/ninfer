#include "ops/linear/split/nvfp4/shapes/n16384_k5120/launch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/linear/common/output.cuh"

namespace ninfer::ops::detail::linear_split::nvfp4_n16384_k5120 {

void launch_a4_tma(const Nvfp4A4Operands& p, std::span<const Tensor> outputs, cudaStream_t stream) {
    launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1>, 5120>>(
        p,
        LinearBf16SegmentedOutput<10240, 6144>{static_cast<__nv_bfloat16*>(outputs[0].data),
                                               static_cast<__nv_bfloat16*>(outputs[1].data)},
        LinearIdentityEpilogue{}, stream);
}

} // namespace ninfer::ops::detail::linear_split::nvfp4_n16384_k5120
