#include "ops/linear/split/dispatch.h"
#include "ops/linear/split/q8/dgx_spark/launch.cuh"
#include "ops/linear/kernels/q8/q8_simt_launch.cuh"

namespace ninfer::ops::detail::linear_split {
namespace {
template <class Base>
struct SplitN12288K4096 : Base {};

using Simt4   = Q8A16SimtSchedule<4, 4, 2, 16, 2, Cache::cg, 2>;
using Simt8   = Q8A16SimtSchedule<8, 8, 1, 32, 2, Cache::cg, 1>;
using Small32 = Q8A16SlicedKMmaSchedule<32, 4, 2, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 4096>;
using Group16 = Q8A16GroupedSlicedKMmaSchedule<16, 2, 2, 2, 1, 4096, Cache::cg, Cache::ca, true>;
using Group32 = Q8A16GroupedSlicedKMmaSchedule<32, 2, 2, 1, 1, 4096, Cache::cg, Cache::ca, true>;
using Small40 = Q8A16SlicedKMmaSchedule<40, 4, 1, 1, Q8ScaleAccess::Shared, Cache::ca, Cache::cg,
                                        Q8ActivationStage::PaddedZero, 4096>;
using Hot64   = Q8A16TmaMmaSchedule<64, 64, 64, 16, 3, 8, 4096>;
using Hot96   = Q8A16TmaMmaSchedule<64, 96, 64, 16, 2, 8, 4096>;
using Bulk128 = Q8A16TmaMmaSchedule<64, 128, 64, 16, 2, 8, 4096>;


using Output = LinearBf16SegmentedOutput<4096, 4096, 4096>;

struct Routes : Q8SplitRoutes<Output> {
    template <class S>
    void simt() const {
        launch_q8_a16_simt<SplitN12288K4096<S>>(operands, output, LinearIdentityEpilogue{}, stream);
    }

    void full() const { launch_q8_n12288_k4096_full(operands, output, stream); }
};

void launch_split(const Tensor& x, Weights weights, std::span<const Tensor> out, MemoryArena*,
                  cudaStream_t stream) {
    const Output output{{static_cast<__nv_bfloat16*>(out[0].data),
                         static_cast<__nv_bfloat16*>(out[1].data),
                         static_cast<__nv_bfloat16*>(out[2].data)}};
    const Routes routes{{{q8_linear_operands(x, weights.first), output, stream}}};
    const int tokens = x.ne[1];
    if (tokens <= 4) return routes.simt<Simt4>();
    if (tokens <= 8) return routes.simt<Simt8>();
    if (tokens <= 32) return routes.sliced<SplitN12288K4096<Small32>>();
    if (tokens <= 40) return routes.sliced<SplitN12288K4096<Small40>>();
    if (tokens <= 48) return routes.grouped<SplitN12288K4096<Group16>>();
    if (tokens <= 64) return routes.grouped<SplitN12288K4096<Group32>>();
    if (tokens <= 96) return routes.tma<SplitN12288K4096<Hot64>>();
    if (tokens <= 192) return routes.tma<SplitN12288K4096<Hot96>>();
    if (tokens >= 1024 && tokens % 128 == 0) return routes.full();
    return routes.tma<SplitN12288K4096<Bulk128>>();
}
} // namespace

const SplitShape kQ8SplitN12288K4096{12288, 4096, kSplitRows<4096, 4096, 4096>, launch_split};
} // namespace ninfer::ops::detail::linear_split
