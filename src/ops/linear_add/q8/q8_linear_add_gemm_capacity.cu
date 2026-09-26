#include "ops/linear/q8/q8_sliced_k_launch.cuh"
#include "ops/linear/q8/q8_geometry.h"
#include "ops/linear_add/q8/q8_linear_add_kernels.h"
#include "ops/linear/q8/q8_instances.cuh"

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);
using Access = Q8ScaleAccess;
using Stage  = Q8ActivationStage;

template <class Geometry, int Capacity, class Schedule>
void launch_add_sliced_k(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Instance = typename Schedule::template with_problem<Geometry::kInputRows, Capacity>;
    const LinearBf16Output output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    launch_q8_a16_sliced_k_mma<Instance>(q8_linear_operands(x, weight), output,
                                         LinearResidualAddEpilogue{{output.data, output.rows, 0}},
                                         stream);
}

// Use the corresponding pure Linear schedules with a residual epilogue. Capacities cover
// intervals of live T; there are no exact-T instances.
template <class Geometry>
Launch select_small(std::int32_t tokens) {
    constexpr bool wide_k = Geometry::kInputRows == 17408;
    using C4  = Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Direct, Cache::ca, Cache::cg,
                                        Stage::RuntimeActive>;
    using C8  = Q8A16SlicedKMmaSchedule<8, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                        Stage::RuntimeActive>;
    using C16 = Q8A16SlicedKMmaSchedule<16, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                        wide_k ? Stage::ActiveOnly : Stage::PaddedZero>;
    using C24 = Q8A16SlicedKMmaSchedule<24, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                        wide_k ? Stage::RuntimeActive : Stage::PaddedZero>;
    using C32 = Q8A16SlicedKMmaSchedule<32, 8, 1, 2, Access::Shared, Cache::ca, Cache::cg,
                                        Stage::RuntimeActive>;
    using C40 =
        Q8A16SlicedKMmaSchedule<40, 4, 1, wide_k ? 3 : 2, Access::Shared, Cache::ca, Cache::cg,
                                wide_k ? Stage::RuntimeActive : Stage::PaddedZero>;
    using C48 = Q8A16SlicedKMmaSchedule<48, 4, 1, wide_k ? 3 : 2, Access::Shared, Cache::ca,
                                        Cache::cg, wide_k ? Stage::ActiveOnly : Stage::PaddedZero>;
    using C56 = Q8A16SlicedKMmaSchedule<56, 4, 1, wide_k ? 3 : 2, Access::Shared, Cache::ca,
                                        Cache::cg, wide_k ? Stage::ActiveOnly : Stage::PaddedZero>;
    using C64 = Q8A16SlicedKMmaSchedule<64, 4, 1, wide_k ? 3 : 2, Access::Shared, Cache::ca,
                                        Cache::cg, wide_k ? Stage::ActiveOnly : Stage::PaddedZero>;
    if (tokens <= 4) return launch_add_sliced_k<Geometry, 4, C4>;
    if (tokens <= 8) return launch_add_sliced_k<Geometry, 8, C8>;
    if (tokens <= 16) return launch_add_sliced_k<Geometry, 16, C16>;
    if (tokens <= 24) return launch_add_sliced_k<Geometry, 24, C24>;
    if (tokens <= 32) return launch_add_sliced_k<Geometry, 32, C32>;
    if (tokens <= 40) return launch_add_sliced_k<Geometry, 40, C40>;
    if (tokens <= 48) return launch_add_sliced_k<Geometry, 48, C48>;
    if (tokens <= 56) return launch_add_sliced_k<Geometry, 56, C56>;
    return launch_add_sliced_k<Geometry, 64, C64>;
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
