#include "ops/linear/swiglu/nvfp4/tma_launch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/linear/swiglu/token_major_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule, int Rows = 28672>
struct FusedTmaSchedule : Nvfp4ScheduleInstance<Schedule, 5120> {};

template <class Schedule>
void launch(const Nvfp4A4Operands& p, __nv_bfloat16* y, cudaStream_t stream) {
    using S = FusedTmaSchedule<Schedule>;
    launch_nvfp4_a4_tma_mma<S>(p, LinearBf16Output{y, 14336}, SwiGluTokenMajorMmaEpilogue{}, stream,
                               SwiGluTokenMajorMmaRows<S>{});
}
} // namespace

void launch_nvfp4_swiglu_n28672_k5120_tma(const Nvfp4A4Operands& p, __nv_bfloat16* y,
                                          cudaStream_t stream) {
    if (p.scale_layout == Nvfp4ScaleLayout::Tiled128)
        launch<Nvfp4A4TmaMmaSchedule<128, 4, 1>>(p, y, stream);
    else
        launch<Nvfp4A4TmaMmaSchedule<256, 3, 1, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>>(p, y, stream);
}
} // namespace ninfer::ops::detail
