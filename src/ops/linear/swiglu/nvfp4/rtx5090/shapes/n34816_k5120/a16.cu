#include "ops/linear/swiglu/nvfp4/dispatch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_instances.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_template_launch.cuh"
#include "ops/linear/swiglu/paired_output.cuh"
#include "ops/linear/swiglu/row_major_mma_epilogue.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule>
void launch_mma(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    using S = Nvfp4ScheduleInstance<Schedule, 5120>;
    launch_nvfp4_a16_mma<S>(nvfp4_a16_operands(x, w),
                            LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), 17408},
                            SwiGluRowMajorMmaEpilogue{}, stream, SwiGluRowMajorMmaRows<S>{});
}
} // namespace

void launch_nvfp4_swiglu_large_t(const Tensor& x, const Weight& w, Tensor& y, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens <= 32) {
        launch_nvfp4_a16_sliced_k_mma<Nvfp4ScheduleInstance<Nvfp4SlicedInstance<32, 4, 2>, 5120>>(
            nvfp4_a16_operands(x, w), LinearBf16Output{static_cast<__nv_bfloat16*>(y.data), 17408},
            SwiGluPairEpilogue{}, stream, SwiGluBranchRows<8>{});
    } else if (tokens <= 64 || (tokens > 144 && tokens <= 192)) {
        launch_mma<Nvfp4A16MmaSchedule<128, 64, 64, 64, 16, 2, 2>>(x, w, y, stream);
    } else if (tokens <= 96) {
        launch_mma<Nvfp4A16MmaSchedule<64, 96, 128, 64, 16, 1, 2>>(x, w, y, stream);
    } else if (tokens > 128 && tokens <= 144) {
        launch_mma<Nvfp4A16MmaSchedule<64, 48, 64, 32, 16, 2, 3>>(x, w, y, stream);
    } else {
        // Group adjacent row tiles to reuse the activation panel across row CTAs.
        using S = Nvfp4A16MmaSchedule<64, 128, 64, 64, 16, 2, 2, Cache::cg, Cache::cg,
                                      Nvfp4MmaFragmentPipeline::PingPong, false, 4>;
        launch_mma<S>(x, w, y, stream);
    }
}
} // namespace ninfer::ops::detail
