#include "ops/linear/projection/nvfp4/dgx_spark/launch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule, Nvfp4ScaleLayout Layout>
void launch(const Weight& weight, Tensor& out, Nvfp4A4Workspace scratch, int tokens,
            cudaStream_t stream) {
    launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<Schedule, 6144>>(
        nvfp4_a4_operands(weight, scratch, tokens, Layout),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, LinearIdentityEpilogue{},
        stream);
}
} // namespace

void launch_nvfp4_n4096_k6144_tma128(const Weight& weight, Tensor& out, Nvfp4A4Workspace scratch,
                                     int tokens, cudaStream_t stream) {
    launch<Nvfp4A4TmaMmaSchedule<128, 4, 1>, Nvfp4ScaleLayout::Tiled128>(weight, out, scratch,
                                                                         tokens, stream);
}

void launch_nvfp4_n4096_k6144_tma256(const Weight& weight, Tensor& out, Nvfp4A4Workspace scratch,
                                     int tokens, cudaStream_t stream) {
    launch<Nvfp4A4TmaMmaSchedule<256, 3, 1>, Nvfp4ScaleLayout::Tiled256>(weight, out, scratch,
                                                                         tokens, stream);
}
} // namespace ninfer::ops::detail
