#include "core/weight.h"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_kernels.h"

#include "core/device.h"
#include "ops/linear/q8/q8_ksplit_config.h"
#include "ops/linear/q8/q8_rowsplit_output.cuh"
#include "ops/linear/q8/q8_ksplit_mma.cuh"
#include "ops/linear_swiglu/q8/q8_linear_swiglu_output.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Geometry                       = Q8LinearGeometry<34816, 5120>;
constexpr std::int32_t kIntermediate = Geometry::kOutputRows / 2;
constexpr std::int32_t kFirstSmallT  = 1;
constexpr std::int32_t kLastSmallT   = 40;
using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int Capacity>
void launch_tile(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule =
        Q8KSplitSchedule<Capacity == 24 ? 8 : 4, Capacity, 2, Q8KSplitScaleAccess::Shared>;
    using RowPolicy = Q8SwiGluPairedRows<kIntermediate>;
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    static_assert((kIntermediate % RowPolicy::kOutputRowsPerCta) == 0);

    const Q8ContiguousOutput ignored_output{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const Q8SwiGluDirectEpilogue epilogue{static_cast<__nv_bfloat16*>(out.data), kIntermediate};
    const RowPolicy row_policy{};
    constexpr int kBlocks = kIntermediate / RowPolicy::kOutputRowsPerCta;
    q8_ksplit_mma_kernel<Geometry, Capacity, Schedule, Q8ContiguousOutput, Q8SwiGluDirectEpilogue,
                         RowPolicy, true, true><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), ignored_output, epilogue, row_policy,
        x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_tile<8 * (1 + static_cast<int>(Offsets))>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<kLastSmallT / 8>{});

} // namespace

void q8_dflash2_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                             cudaStream_t stream) {
    if (x.ne[1] < kFirstSmallT || x.ne[1] > kLastSmallT) {
        throw std::invalid_argument("Q8 DFlash2 LinearSwiGLU small-T: unsupported T");
    }
    const std::size_t index = static_cast<std::size_t>((x.ne[1] - 1) / 8);
    kLaunchers[index](x, weight, out, stream);
}

} // namespace ninfer::ops::detail
