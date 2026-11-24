#include "ops/linear/kernels/q8/q8_geometry.h"
#include "core/weight.h"
#include "ops/linear/split/dispatch.h"

#include "core/device.h"
#include "ops/linear/kernels/q8/q8_schedule.cuh"
#include "ops/linear/kernels/q8/q8_mma_launch.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/kernels/q8/q8_sliced_k_launch.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail::linear_split {
namespace {

using Geometry                          = Q8LinearGeometry<6144, 5120>;
constexpr std::int32_t kFirstRows       = 4096;
constexpr std::int32_t kTailRows        = 1024;
constexpr std::int32_t kLastSmallTokens = 48;
using Output = LinearBf16SegmentedOutput<kFirstRows, kTailRows, kTailRows>;
using Launch = void (*)(const Tensor&, const Weight&, std::span<const Tensor>, cudaStream_t);

// Exact input-load unrolling matters in the first two MMA column tiles. Wider blocks use
// runtime live columns with one specialization per accumulator capacity.
template <int Columns, bool Exact>
void launch_small(const Tensor& x, const Weight& weight, std::span<const Tensor> out,
                  cudaStream_t stream) {
    constexpr int Capacity = (Columns + 7) / 8 * 8;
    constexpr int Warps    = Columns <= 4 ? 16 : Columns <= 16 ? 8 : 4;
    constexpr auto Scales  = Columns <= 4 || (Columns > 8 && Columns <= 16) ? Q8ScaleAccess::Direct
                                                                            : Q8ScaleAccess::Shared;
    using Schedule         = Q8A16SlicedKMmaSchedule<Capacity, Warps, 1, 2, Scales>;
    static_assert((kFirstRows % Schedule::kBlockRows) == 0);
    static_assert((kTailRows % Schedule::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);

    const Output output{static_cast<__nv_bfloat16*>(out[0].data),
                        static_cast<__nv_bfloat16*>(out[1].data),
                        static_cast<__nv_bfloat16*>(out[2].data)};
    launch_q8_a16_sliced_k_mma<
        typename Schedule::template with_problem<Geometry::kInputRows, Columns, Exact>,
        Q8SlicedKIdentityRows>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                               stream);
    CUDA_CHECK(cudaGetLastError());
}

template <bool Exact, std::size_t... I>
constexpr auto make_small_launchers(std::index_sequence<I...>) {
    return std::array<Launch, sizeof...(I)>{
        &launch_small<(Exact ? 1 + static_cast<int>(I) : 24 + 8 * static_cast<int>(I)), Exact>...};
}

constexpr auto kExactLaunchers = make_small_launchers<true>(std::make_index_sequence<16>{});
constexpr auto kTileLaunchers  = make_small_launchers<false>(std::make_index_sequence<4>{});

template <class Schedule>
void launch_mma(const Tensor& x, const Weight& weight, std::span<const Tensor> out,
                cudaStream_t stream) {
    const Output output{static_cast<__nv_bfloat16*>(out[0].data),
                        static_cast<__nv_bfloat16*>(out[1].data),
                        static_cast<__nv_bfloat16*>(out[2].data)};
    launch_q8_a16_mma<Schedule>(q8_linear_operands(x, weight), output, LinearIdentityEpilogue{},
                                stream);
}

void launch_small_band(const Tensor& x, const Weight& weight, std::span<const Tensor> out,
                       cudaStream_t stream) {
    if (x.ne[1] < 1 || x.ne[1] > kLastSmallTokens) {
        throw std::invalid_argument("Q8 split n6144_k5120 small-T: unsupported T");
    }
    if (x.ne[1] <= 16)
        kExactLaunchers[x.ne[1] - 1](x, weight, out, stream);
    else
        kTileLaunchers[(x.ne[1] - 17) / 8](x, weight, out, stream);
}

void launch_split(const Tensor& x, Weights weights, std::span<const Tensor> out, MemoryArena*,
                  cudaStream_t stream) {
    const auto& w = weights.first;
    if (x.ne[1] <= 48) return launch_small_band(x, w, out, stream);
    if (x.ne[1] <= 63)
        return launch_mma<Q8A16MmaSchedule<16, 64, 128, 16, 16, 1, 1>>(x, w, out, stream);
    if (x.ne[1] <= 96)
        return launch_mma<Q8A16MmaSchedule<32, 32, 128, 16, 16, 1, 1>>(x, w, out, stream);
    if (x.ne[1] <= 128)
        return launch_mma<Q8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(x, w, out, stream);
    if (x.ne[1] <= 192)
        return launch_mma<Q8A16MmaSchedule<32, 64, 64, 32, 16, 2, 3>>(x, w, out, stream);
    launch_mma<Q8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, w, out, stream);
}
} // namespace

const SplitShape kQ8SplitN6144K5120{6144, 5120, kSplitRows<4096, 1024, 1024>, launch_split};
} // namespace ninfer::ops::detail::linear_split
