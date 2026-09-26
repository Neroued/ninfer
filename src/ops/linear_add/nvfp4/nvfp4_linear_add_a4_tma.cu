#include "ops/linear_add/nvfp4/nvfp4_linear_add_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"

namespace ninfer::ops::detail {
namespace {
template <int K>
void launch(const Nvfp4A4Operands& p, __nv_bfloat16* y, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1>, K>;
    launch_nvfp4_a4_tma_mma<S>(p, LinearBf16Output{y, p.rows},
                               LinearResidualAddEpilogue{{y, p.rows}}, stream);
}
} // namespace

void launch_nvfp4_a4_tma_linear_add(const Nvfp4A4Operands& p, __nv_bfloat16* residual,
                                    cudaStream_t stream) {
    if (p.k == 6144)
        launch<6144>(p, residual, stream);
    else if (p.k == 17408)
        launch<17408>(p, residual, stream);
    else
        throw std::invalid_argument("NVFP4 TMA linear_add: unsupported K");
}
} // namespace ninfer::ops::detail
