#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"
#include "ops/attn_input_proj/fp8/fp8_attn_input_output.cuh"

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

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                        cudaStream_t);

template <int Capacity>
void launch_tile(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                  Tensor& v, cudaStream_t stream) {
    using Geometry      = Fp8AttnInputGeometry;
    constexpr int tile  = Capacity;
    constexpr int warps = Capacity <= 8 ? 16 : Capacity <= 24 ? 8 : 4;
    using Schedule      = Fp8A16SmallTMmaSchedule<warps, tile, warps == 16 ? 1 : 2>;
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const Fp8AttentionInputOutput output{
        static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)};
    fp8_a16_small_t_mma_kernel<Geometry, Capacity, Schedule, Fp8AttentionInputOutput, true>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_tile<8 * (1 + static_cast<int>(Offsets))>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<(kFp8AttnInputLastSmallMmaT + 7) / 8>{});

} // namespace

void fp8_attn_input_a16_small_mma_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream) {
    kLaunchers.at((x.ne[1] - 1) / 8)(x, weight, q, gate, k, v, stream);
}
} // namespace ninfer::ops::detail
