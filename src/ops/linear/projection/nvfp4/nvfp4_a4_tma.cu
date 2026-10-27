#include "ops/linear/projection/nvfp4/nvfp4_a4_tma_launch.h"
#include "core/platform.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_tma_split.cuh"

namespace ninfer::ops::detail {
namespace {
template <int K, CUtensorMapL2promotion Promotion = CU_TENSOR_MAP_L2_PROMOTION_NONE>
void launch(const Nvfp4A4Operands& p, __nv_bfloat16* output, cudaStream_t stream) {
    const LinearBf16Output y{output, p.rows};
    if (p.scale_layout == Nvfp4ScaleLayout::Tiled128)
        launch_nvfp4_a4_tma_mma<
            Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<128, 4, 1, Promotion>, K>>(
            p, y, LinearIdentityEpilogue{}, stream);
    else
        launch_nvfp4_a4_tma_mma<
            Nvfp4ScheduleInstance<Nvfp4A4TmaMmaSchedule<256, 3, 1, Promotion>, K>>(
            p, y, LinearIdentityEpilogue{}, stream);
}
} // namespace

void launch_nvfp4_a4_tma_linear(Nvfp4GeometryId problem, const Nvfp4A4Operands& p,
                                __nv_bfloat16* output, cudaStream_t stream) {
    switch (problem) {
    case Nvfp4GeometryId::N14336K5120:
        launch<5120>(p, output, stream);
        return;
    case Nvfp4GeometryId::N16384K5120:
        launch<5120>(p, output, stream);
        return;
    case Nvfp4GeometryId::N28672K5120:
        if (p.scale_layout == Nvfp4ScaleLayout::Tiled128)
            launch<5120>(p, output, stream);
        else
            launch<5120, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>(p, output, stream);
        return;
    case Nvfp4GeometryId::N34816K5120:
        launch<5120, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>(p, output, stream);
        return;
    case Nvfp4GeometryId::N5120K6144:
        launch<6144>(p, output, stream);
        return;
    case Nvfp4GeometryId::N5120K17408:
        launch<17408>(p, output, stream);
        return;
#if NINFER_TARGET_RTX5090
    case Nvfp4GeometryId::N5120K14336:
        launch_nvfp4_a4_tma_mma<Nvfp4ScheduleInstance<
            Nvfp4A4TmaMmaSchedule<256, 3, 1, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>, 14336>>(
            p, LinearBf16Output{output, p.rows}, LinearIdentityEpilogue{}, stream);
        return;
#endif
    }
    throw std::invalid_argument("NVFP4 TMA linear: unsupported geometry");
}

void launch_nvfp4_a4_tma_linear_split(Nvfp4GeometryId problem, const Nvfp4A4Operands& p,
                                      Nvfp4A4Workspace workspace, __nv_bfloat16* output,
                                      cudaStream_t stream) {
#if NINFER_TARGET_RTX5090
    if (problem == Nvfp4GeometryId::N5120K14336) {
        if (p.scale_layout == Nvfp4ScaleLayout::Tiled128) {
            using S = Nvfp4A4TmaSplitKSchedule<
                Nvfp4A4TmaMmaSchedule<128, 4, 1>, kNvfp4LinearTmaKPartitions>;
            launch_nvfp4_a4_tma_split_linear<Nvfp4ScheduleInstance<S, 14336>>(
                p, workspace, output, stream);
        } else {
            using S = Nvfp4A4TmaSplitKSchedule<
                Nvfp4A4TmaMmaSchedule<256, 3, 1, CU_TENSOR_MAP_L2_PROMOTION_L2_128B>,
                kNvfp4LinearTmaKPartitions>;
            launch_nvfp4_a4_tma_split_linear<Nvfp4ScheduleInstance<S, 14336>>(
                p, workspace, output, stream);
        }
        return;
    }
#endif
    throw std::invalid_argument("NVFP4 TMA split-K linear: unsupported geometry");
}
} // namespace ninfer::ops::detail
