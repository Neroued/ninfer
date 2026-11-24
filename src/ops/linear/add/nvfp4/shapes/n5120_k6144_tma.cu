#include "ops/linear/add/nvfp4/dispatch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh"

namespace ninfer::ops::detail {
void launch_nvfp4_add_n5120_k6144_tma(const Tensor& x, const Weight& w, Tensor& out,
                                      Nvfp4A4Workspace scratch, cudaStream_t stream) {
    using S    = Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1>, 6144>;
    auto* data = static_cast<__nv_bfloat16*>(out.data);
    launch_nvfp4_a4_tma_mma<S>(nvfp4_a4_operands(w, scratch, x.ne[1], Nvfp4ScaleLayout::Tiled256),
                               LinearBf16Output{data, w.n}, LinearResidualAddEpilogue{{data, w.n}},
                               stream);
}
} // namespace ninfer::ops::detail
