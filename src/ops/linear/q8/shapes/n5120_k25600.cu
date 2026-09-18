#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_ksplit_launch.cuh"
#include "ops/linear/q8/q8_rowsplit_gemm_mma.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N5120K25600;
using Access   = Q8KSplitScaleAccess;
using Stage    = Q8KSplitActivationStage;
using C8       = Q8KSplitSchedule<8, 8, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C16 = Q8KSplitSchedule<8, 16, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C24 = Q8KSplitSchedule<8, 24, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C32 = Q8KSplitSchedule<8, 32, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C40 = Q8KSplitSchedule<4, 40, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C48 = Q8KSplitSchedule<4, 48, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;
using C56 = Q8KSplitSchedule<4, 56, 2, Access::Shared, Cache::ca, Cache::cg, Stage::ActiveOnly>;

template <int Rows>
void launch_tiled(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    // Retain the predicated variant on complete tiles: the Full variant regresses T=64.
    using Schedule = Q8RowSplitMmaGemmSchedule<Rows, 64, 16, 16, 1, 2, 128, 1>;
    const dim3 grid(weight.n / Rows, (x.ne[1] + 63) / 64);
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), weight.n};
    q8_rowsplit_gemm_mma_kernel<Schedule, false><<<grid, Schedule::THREADS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), output, weight.n, weight.k, x.ne[1],
        weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

Q8Launch select_q8_n5120_k25600(std::int32_t tokens) {
    if (tokens <= 8) return launch_q8_ksplit<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_ksplit<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_ksplit<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_ksplit<Geometry, 32, C32>;
    if (tokens <= 40) return launch_q8_ksplit<Geometry, 40, C40>;
    if (tokens <= 48) return launch_q8_ksplit<Geometry, 48, C48>;
    if (tokens <= 56) return launch_q8_ksplit<Geometry, 56, C56>;
    if (tokens <= 64) return launch_tiled<16>;
    if (tokens <= 128) return launch_tiled<32>;
    return launch_q8_mma_r64_c128;
}

} // namespace ninfer::ops::detail
