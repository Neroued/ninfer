#include "ops/linear/fp8/fp8_launch.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_a16_small_t_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry = Fp8VocabularyGeometry;
    using Schedule = typename Fp8VocabularyA16SmallTMmaProductionSchedule<ActiveTokens>::Type;
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    fp8_a16_small_t_mma_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<kFp8VocabularyFirstA16SmallTMmaT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kFp8VocabularyLastA16SmallTMmaT -
                                            kFp8VocabularyFirstA16SmallTMmaT + 1>{});

} // namespace

void launch_fp8_vocabulary_a16_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream) {
    if (weight.n != Fp8VocabularyGeometry::kOutputRows ||
        weight.k != Fp8VocabularyGeometry::kInputRows ||
        x.ne[1] < kFp8VocabularyFirstA16SmallTMmaT || x.ne[1] > kFp8VocabularyLastA16SmallTMmaT) {
        throw std::invalid_argument("fp8 vocabulary A16 small-T MMA: invalid exact problem");
    }
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kFp8VocabularyFirstA16SmallTMmaT);
    kLaunchers[index](x, weight, out, stream);
}

} // namespace ninfer::ops::detail
