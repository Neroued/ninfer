#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_simt.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                        cudaStream_t);

struct Nvfp4AttentionInputSmallTOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float result) const {
        constexpr std::int32_t kQueryRows  = 6144;
        constexpr std::int32_t kKeyRows    = 1024;
        constexpr std::int32_t kGateRows   = 6144;
        constexpr std::int32_t kKeyBegin   = kQueryRows;
        constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
        constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;
        const __nv_bfloat16 result_bf16    = __float2bfloat16_rn(result);

        if (parent_row < kKeyBegin) {
            query[static_cast<std::int64_t>(token) * kQueryRows + parent_row] = result_bf16;
        } else if (parent_row < kGateBegin) {
            key[static_cast<std::int64_t>(token) * kKeyRows + parent_row - kKeyBegin] = result_bf16;
        } else if (parent_row < kValueBegin) {
            gate[static_cast<std::int64_t>(token) * kGateRows + parent_row - kGateBegin] =
                result_bf16;
        } else {
            value[static_cast<std::int64_t>(token) * kKeyRows + parent_row - kValueBegin] =
                result_bf16;
        }
    }
};

// The four-output epilogue shifts the measured low-T warp crossover relative to contiguous Linear,
// so Attention owns this production mapping even though both routes share the compute body.
template <int ActiveTokens>
struct Nvfp4AttentionSmallTProductionSchedule {
    static_assert(ActiveTokens >= 2);
    static_assert(ActiveTokens <= 32);
    static constexpr int kWarpsPerCta       = ActiveTokens >= 17 ? 4 : (ActiveTokens >= 8 ? 16 : 8);
    static constexpr int kValuesPerLane     = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Nvfp4SimtActivationAccess::SharedPhase
                                                  : Nvfp4SimtActivationAccess::TokenPacked;
    using Type =
        Nvfp4SimtSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                          Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                          Nvfp4SimtBlockOrder::RowsContiguous, 1>;
};

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                  Tensor& v, cudaStream_t stream) {
    using Geometry            = Nvfp4N14336K5120;
    using Schedule            = typename Nvfp4AttentionSmallTProductionSchedule<ActiveTokens>::Type;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;

    const Nvfp4AttentionInputSmallTOutput output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    nvfp4_simt_kernel<Geometry, ActiveTokens, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor,
        Nvfp4IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{&launch_exact<2 + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<32 - 2 + 1>{});

} // namespace

void nvfp4_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream) {
    kLaunchers[x.ne[1] - 2](x, weight, q, gate, k, v, stream);
}

} // namespace ninfer::ops::detail
