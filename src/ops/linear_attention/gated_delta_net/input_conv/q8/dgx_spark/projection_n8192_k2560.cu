#include "ops/common/cuda/memory.cuh"
#include "ops/linear_attention/gated_delta_net/input_conv/q8/dgx_spark/launch.cuh"
#include "ops/linear/kernels/q8/q8_output_launch.cuh"
#include "ops/linear/kernels/q8/q8_a16_tma.cuh"

namespace ninfer::ops::detail {
namespace {

using Gemv      = Q8A16GemvSchedule<4, 2, 2, 2560, 4>;
using Small8    = Q8A16SlicedKMmaSchedule<8, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Grouped32 = Q8A16GroupedSlicedKMmaSchedule<32, 2, 2, 1, 1, 2560, Cache::cg, Cache::ca, true>;
using Small40   = Q8A16SlicedKMmaSchedule<40, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Hot32     = Q8A16TmaMmaSchedule<64, 32, 32, 16, 3, 8, 2560>;
using Hot64     = Q8A16TmaMmaSchedule<64, 64, 64, 16, 3, 8, 2560>;
using Hot96     = Q8A16TmaMmaSchedule<64, 96, 64, 16, 2, 8, 2560>;
using Bulk128   = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 8, 2560>;

using Output = LinearBf16SegmentedOutput<5120, 3072>;

struct Routes : Q8A16OutputLaunch<Output> {
    void full() const { launch_q8_gdn_n8192_k2560_full(operands, output, stream); }

    template <class Schedule>
    void tma() const {
        launch_q8_a16_tma_mma<Schedule>(operands, output, LinearIdentityEpilogue{}, stream);
    }
};

void select_route(int tokens, const Routes& routes) {
    if (tokens == 1) return routes.gemv<Gemv>();
    if (tokens <= 8) return routes.sliced<Small8>();
    if (tokens <= 32) return routes.grouped<Grouped32>();
    if (tokens <= 40) return routes.sliced<Small40>();
    if (tokens <= 96) return routes.tma<Hot32>();
    if (tokens <= 128) return routes.tma<Hot64>();
    if (tokens <= 192) return routes.tma<Hot96>();
    if (tokens >= 1024 && tokens % 128 == 0) return routes.full();
    return routes.tma<Bulk128>();
}

} // namespace

void q8_gdn_projection_n8192_k2560(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream) {
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    select_route(x.ne[1], Routes{{q8_linear_operands(x, weight), output, stream}});
}
} // namespace ninfer::ops::detail
