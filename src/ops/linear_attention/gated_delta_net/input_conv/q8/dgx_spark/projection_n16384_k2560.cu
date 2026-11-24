#include "ops/linear_attention/gated_delta_net/input_conv/q8/dgx_spark/launch.cuh"
#include "ops/linear/kernels/q8/q8_output_launch.cuh"
#include "ops/linear/kernels/q8/q8_a16_tma.cuh"

namespace ninfer::ops::detail {
namespace {
using Gemv      = Q8A16GemvSchedule<8, 1, 2, 2560, 8>;
using Sliced8   = Q8A16SlicedKMmaSchedule<8, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Grouped16 = Q8A16GroupedSlicedKMmaSchedule<16, 2, 2, 2, 1, 2560, Cache::cg, Cache::ca, true>;
using Sliced24  = Q8A16SlicedKMmaSchedule<24, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Sliced32  = Q8A16SlicedKMmaSchedule<32, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Sliced40  = Q8A16SlicedKMmaSchedule<40, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                          Q8ActivationStage::PaddedZero, 2560>;
using Mma48     = Q8A16MmaSchedule<64, 48, 128, 16, 24, 1, 2>;
using Mma64     = Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>;
using Mma64K128 = Q8A16MmaSchedule<64, 64, 128, 32, 16, 1, 2>;
using Hot96     = Q8A16TmaMmaSchedule<64, 96, 64, 16, 2, 8, 2560>;
using Bulk128   = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 8, 2560>;

struct Output : LinearBf16SegmentedOutput<10240, 6144> {};

struct Routes : Q8A16OutputLaunch<Output> {
    void full() const { launch_q8_gdn_n16384_k2560_full(operands, output, stream); }

    template <class Schedule>
    void tma() const {
        launch_q8_a16_tma_mma<Schedule>(operands, output, LinearIdentityEpilogue{}, stream);
    }
};

template <class Routes>
auto select_route(int tokens, const Routes& routes) {
    if (tokens <= 1) return routes.template gemv<Gemv>();
    if (tokens <= 8) return routes.template sliced<Sliced8>();
    if (tokens <= 16) return routes.template grouped<Grouped16>();
    if (tokens <= 24) return routes.template sliced<Sliced24>();
    if (tokens <= 32) return routes.template sliced<Sliced32>();
    if (tokens <= 40) return routes.template sliced<Sliced40>();
    if (tokens <= 48) return routes.template mma<Mma48>();
    if (tokens <= 63) return routes.template mma<Mma64>();
    if (tokens <= 64) return routes.template mma<Mma64K128>();
    if (tokens >= 1024 && tokens % 128 == 0) return routes.full();
    if (tokens <= 96 || (tokens > 128 && tokens <= 192)) return routes.template tma<Hot96>();
    return routes.template tma<Bulk128>();
}

} // namespace

void q8_gdn_projection_n16384_k2560(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream) {
    const Output output{
        {static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)}};
    select_route(x.ne[1], Routes{{q8_linear_operands(x, weight), output, stream}});
}
} // namespace ninfer::ops::detail
