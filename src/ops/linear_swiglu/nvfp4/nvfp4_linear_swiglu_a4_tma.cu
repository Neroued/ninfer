#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/linear_swiglu/token_major_mma_epilogue.cuh"

namespace ninfer::ops::detail {
void launch_nvfp4_linear_swiglu_a4_tma(const Nvfp4A4Operands& p, __nv_bfloat16* output,
                                       cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1>, 5120>;
    launch_nvfp4_a4_tma_mma<S>(p, LinearBf16Output{output, p.rows / 2},
                               SwiGluTokenMajorMmaEpilogue{}, stream, SwiGluTokenMajorMmaRows<S>{});
}
} // namespace ninfer::ops::detail
