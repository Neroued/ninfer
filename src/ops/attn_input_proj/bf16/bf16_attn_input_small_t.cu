#include "core/weight.h"
#include "ops/attn_input_proj/bf16/bf16_attn_input_plan.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_simt.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, Tensor&, Tensor&,
                        cudaStream_t);

struct Bf16AttentionInputSmallTOutput {
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
        const __nv_bfloat16 value_bf16     = __float2bfloat16_rn(result);

        if (parent_row < kKeyBegin) {
            query[static_cast<std::int64_t>(token) * kQueryRows + parent_row] = value_bf16;
        } else if (parent_row < kGateBegin) {
            key[static_cast<std::int64_t>(token) * kKeyRows + parent_row - kKeyBegin] = value_bf16;
        } else if (parent_row < kValueBegin) {
            gate[static_cast<std::int64_t>(token) * kGateRows + parent_row - kGateBegin] =
                value_bf16;
        } else {
            value[static_cast<std::int64_t>(token) * kKeyRows + parent_row - kValueBegin] =
                value_bf16;
        }
    }
};

template <int ActiveTokens>
struct Bf16AttentionSmallTProductionSchedule {
    static_assert(ActiveTokens >= 2 && ActiveTokens <= 32);
    // The Attention epilogue is measured separately from Linear, so it owns its exact-T winner
    // mapping while sharing the Linear computation body.
    static constexpr int kRowsPerWarp = ActiveTokens <= 4 ? 8 : (ActiveTokens <= 8 ? 4 : 2);
    static constexpr Bf16SimtActivationAccess kActivationAccess =
        ActiveTokens <= 8 ? Bf16SimtActivationAccess::WarpPacked
                          : Bf16SimtActivationAccess::DirectStream;
    static constexpr bool kSequential = ActiveTokens <= 9 || ActiveTokens >= 17;
    static constexpr bool kUnroll2 = ActiveTokens == 4 || ActiveTokens == 5 || ActiveTokens == 8 ||
                                     (ActiveTokens >= 10 && ActiveTokens <= 18) ||
                                     ActiveTokens >= 23;
    static constexpr Bf16WeightCache kWeightCache =
        ActiveTokens == 7 ? Bf16WeightCache::Streaming : Bf16WeightCache::Default;
    static constexpr Bf16PhaseOrder kPhaseOrder =
        kSequential ? Bf16PhaseOrder::Sequential : Bf16PhaseOrder::RowSwizzled;
    using Type = Bf16SimtSchedule<4, 1, kRowsPerWarp, 8, 1, 4, kActivationAccess, kWeightCache,
                                  kPhaseOrder, 1, kUnroll2 ? 2 : 1, 1, 2>;
};

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                  Tensor& v, cudaStream_t stream) {
    using Geometry = Bf16Geometry<14336, 5120>;
    using Schedule = typename Bf16AttentionSmallTProductionSchedule<ActiveTokens>::Type;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    static_assert((6144 % Schedule::kRowsPerCta) == 0);
    static_assert((1024 % Schedule::kRowsPerCta) == 0);

    const Bf16AttentionInputSmallTOutput output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    bf16_simt_kernel<Geometry, ActiveTokens, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        output);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<kBf16AttnInputSmallTMinTokens + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers = make_launchers(
    std::make_index_sequence<kBf16AttnInputSmallTMaxTokens - kBf16AttnInputSmallTMinTokens + 1>{});

} // namespace

void bf16_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream) {
    if (x.ne[1] < kBf16AttnInputSmallTMinTokens || x.ne[1] > kBf16AttnInputSmallTMaxTokens) {
        throw std::invalid_argument("bf16 attn_input_proj small-T requires T in [2,32]");
    }
    kLaunchers[x.ne[1] - kBf16AttnInputSmallTMinTokens](x, weight, q, gate, k, v, stream);
}

} // namespace ninfer::ops::detail
