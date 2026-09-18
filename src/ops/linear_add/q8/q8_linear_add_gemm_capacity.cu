#include "ops/linear_add/q8/q8_linear_add_kernels.h"
#include "ops/linear/q8/q8_ksplit_launch.cuh"

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);
using Access = Q8KSplitScaleAccess;
using Stage  = Q8KSplitActivationStage;

// Use the corresponding pure Linear schedules with a residual epilogue. Capacities cover
// intervals of live T; there are no exact-T instances.
template <class Geometry>
Launch select_small(std::int32_t tokens) {
    constexpr bool wide_k = Geometry::kInputRows == 17408;
    using C4 =
        Q8KSplitSchedule<8, 8, 2, Access::Direct, Cache::ca, Cache::cg, Stage::RuntimeActive>;
    using C8 =
        Q8KSplitSchedule<8, 8, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
    using C16 = Q8KSplitSchedule<8, 16, 2, Access::Shared, Cache::ca, Cache::cg,
                                 wide_k ? Stage::ActiveOnly : Stage::PaddedZero>;
    using C24 = Q8KSplitSchedule<8, 24, 2, Access::Shared, Cache::ca, Cache::cg,
                                 wide_k ? Stage::RuntimeActive : Stage::PaddedZero>;
    using C32 =
        Q8KSplitSchedule<8, 32, 2, Access::Shared, Cache::ca, Cache::cg, Stage::RuntimeActive>;
    using C40 = Q8KSplitSchedule<4, 40, wide_k ? 3 : 2, Access::Shared, Cache::ca, Cache::cg,
                                 wide_k ? Stage::RuntimeActive : Stage::PaddedZero>;
    using C48 = Q8KSplitSchedule<4, 48, wide_k ? 3 : 2, Access::Shared, Cache::ca, Cache::cg,
                                 wide_k ? Stage::ActiveOnly : Stage::PaddedZero>;
    using C56 = Q8KSplitSchedule<4, 56, wide_k ? 3 : 2, Access::Shared, Cache::ca, Cache::cg,
                                 wide_k ? Stage::ActiveOnly : Stage::PaddedZero>;
    using C64 = Q8KSplitSchedule<4, 64, wide_k ? 3 : 2, Access::Shared, Cache::ca, Cache::cg,
                                 wide_k ? Stage::ActiveOnly : Stage::PaddedZero>;
    if (tokens <= 4) return launch_q8_ksplit<Geometry, 4, C4, Q8KSplitResidualEpilogue>;
    if (tokens <= 8) return launch_q8_ksplit<Geometry, 8, C8, Q8KSplitResidualEpilogue>;
    if (tokens <= 16) return launch_q8_ksplit<Geometry, 16, C16, Q8KSplitResidualEpilogue>;
    if (tokens <= 24) return launch_q8_ksplit<Geometry, 24, C24, Q8KSplitResidualEpilogue>;
    if (tokens <= 32) return launch_q8_ksplit<Geometry, 32, C32, Q8KSplitResidualEpilogue>;
    if (tokens <= 40) return launch_q8_ksplit<Geometry, 40, C40, Q8KSplitResidualEpilogue>;
    if (tokens <= 48) return launch_q8_ksplit<Geometry, 48, C48, Q8KSplitResidualEpilogue>;
    if (tokens <= 56) return launch_q8_ksplit<Geometry, 56, C56, Q8KSplitResidualEpilogue>;
    return launch_q8_ksplit<Geometry, 64, C64, Q8KSplitResidualEpilogue>;
}

} // namespace

void q8_linear_add_splitk_capacity_launch(const Tensor& x, const Weight& w, Tensor& residual,
                                          cudaStream_t stream) {
    if (w.k == 6144) {
        select_small<Q8LinearGeometry<5120, 6144>>(x.ne[1])(x, w, residual, stream);
    } else {
        select_small<Q8LinearGeometry<5120, 17408>>(x.ne[1])(x, w, residual, stream);
    }
}

} // namespace ninfer::ops::detail
