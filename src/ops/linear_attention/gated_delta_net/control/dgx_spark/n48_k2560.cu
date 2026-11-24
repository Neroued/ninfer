#include "ops/linear_attention/gated_delta_net/control/bf16/launch.h"
#include "ops/common/numeric/math.cuh"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"

#include <array>

namespace ninfer::ops::detail::gdn_control {
namespace {
template <int Heads>
struct Bf16ControlOutput {
    float* g;
    float* beta;

    __device__ __forceinline__ void store(int row, int column, float value) const {
        const bool is_b                                        = row >= Heads;
        const int head                                         = is_b ? row - Heads : row;
        (is_b ? beta : g)[std::int64_t(column) * Heads + head] = value;
    }
};

template <int Heads>
struct Bf16ControlEpilogue {
    const float* a_log;
    const float* dt_bias;

    __device__ __forceinline__ float apply(int row, int, float value) const {
        if (row >= Heads) return common::sigmoid(value);
        return -expf(a_log[row]) * common::softplus(value + dt_bias[row]);
    }
};

constexpr std::array kRoutes{
    Route{1, 1, {Kernel::Gemv}},
    Route{2, 16, {Kernel::TiledGemv, 2}},
    Route{17, 128, {Kernel::Mma, 8, 1, RowMode::Merged}},
    Route{129, 512, {Kernel::Mma, 16, 1, RowMode::Merged}},
    Route{513, std::numeric_limits<std::int32_t>::max(), {Kernel::Mma, 32, 1, RowMode::Merged}},
};
static_assert(closed(kRoutes));

using Gemv =
    Bf16ScheduleInstance<Bf16KTailSchedule<Bf16A16GemvSchedule<
                             16, 16, 1, 8, 4, Bf16ActivationAccess::Direct,
                             Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 4, 2>>,
                         2560>;
using Small = Bf16ScheduleInstance<
    Bf16A16SimtSchedule<4, 2, 1, 8, 4, 2, Bf16SimtActivationAccess::WarpPacked,
                        Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 4, 1, 2, 2>,
    2560>;
using Short =
    Bf16ScheduleInstance<Bf16KTailSchedule<Bf16A16SlicedKMmaSchedule<16, 8, 16, 64, 2>>, 2560>;
using Medium =
    Bf16ScheduleInstance<Bf16KTailSchedule<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 2>>, 2560>;
using Large = Bf16ScheduleInstance<
    Bf16A16MmaSchedule<48, 32, 256, 16, 8, 2, 1, Cache::cg, Cache::cg,
                       Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>,
    2560>;
} // namespace

std::span<const Route> routes_n48_k2560() { return kRoutes; }

void launch_n48_k2560(const Inputs& in, GdnControlOutputs out, Plan plan, Scratch,
                      CudaExecutionView execution) {
    const Bf16A16Operands operands{static_cast<const __nv_bfloat16*>(in.x.data),
                                   static_cast<const __nv_bfloat16*>(in.a), 48, 2560, in.x.ne[1]};
    const Bf16ControlOutput<24> output{static_cast<float*>(out.g.data),
                                       static_cast<float*>(out.beta.data)};
    const Bf16ControlEpilogue<24> epilogue{static_cast<const float*>(in.a_log.data),
                                           static_cast<const float*>(in.dt_bias.data)};
    if (plan.kernel == Kernel::Gemv) {
        launch_bf16_a16_gemv<Gemv>(operands, output, epilogue, execution.stream());
        return;
    }
    if (plan.kernel == Kernel::TiledGemv) {
        launch_bf16_a16_simt<Small>(operands, output, epilogue, execution.stream());
        return;
    }
    if (plan.kernel == Kernel::Mma) {
        if (plan.tile_cols == 8)
            launch_bf16_a16_sliced_k_mma<Short>(operands, output, epilogue, execution.stream());
        else if (plan.tile_cols == 16)
            launch_bf16_a16_sliced_k_mma<Medium>(operands, output, epilogue, execution.stream());
        else if (plan.tile_cols == 32)
            launch_bf16_a16_mma<Large>(operands, output, epilogue, execution.stream());
        else
            throw std::logic_error("gdn_control: invalid 48x2560 MMA tile");
        return;
    }
    throw std::logic_error("gdn_control: invalid 48x2560 plan");
}
} // namespace ninfer::ops::detail::gdn_control
