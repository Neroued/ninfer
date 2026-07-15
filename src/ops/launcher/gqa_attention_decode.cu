// ninfer::ops - split-KV GQA small-T launcher and unified route dispatcher.
#include "ops/launcher/gqa_attention.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_decode_bf16.cuh"
#include "ops/kernel/gqa_attention_decode_i8.cuh"
#include "core/device.h" // CUDA_CHECK
#include "ninfer/ops/gqa_attention.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Supplies an upper bound for the device-side active-split policy over one explicit execution
// envelope. Eager calls normally pass an exact window; graph calls pass their target-private
// replay interval. The dtype-aware wrapper below adds the measured INT8 specializations.
template <typename Geometry>
std::int32_t gqa_small_t_split_upper_bound(std::int32_t window) {
    if (window <= 0) { return Geometry::DecodeSplits; }

    constexpr std::int32_t kMinSplits = 4 * Geometry::DecodeSplitScale;
    std::int32_t splits               = kMinSplits;

    const auto include_tier = [&](std::int32_t window_limit, std::int32_t target_keys_per_split) {
        const std::int32_t tier_window = (window < window_limit) ? window : window_limit;
        if (tier_window > 0) {
            const std::int32_t tier_splits = div_up(tier_window, target_keys_per_split);
            splits                         = (splits > tier_splits) ? splits : tier_splits;
        }
    };

    include_tier(4096, 64 / Geometry::DecodeSplitScale);
    if (window > 4096) { include_tier(8198, 128 / Geometry::DecodeSplitScale); }
    if (window > 8198) { include_tier(16390, 256 / Geometry::DecodeSplitScale); }
    if (window > 16390) { include_tier(window, 480 / Geometry::DecodeSplitScale); }

    return (splits < Geometry::DecodeSplits) ? splits : Geometry::DecodeSplits;
}

template <typename Geometry>
std::int32_t gqa_small_t_split_count(std::int32_t window, std::int32_t tokens, DType kv_dtype) {
    // A 64-key default split just above a 32-key boundary makes the partial
    // kernel execute a nearly empty second tile. These short ranges instead
    // launch one 32-key tile per split; the larger CTAs keep the small grid busy.
    if (kv_dtype == DType::I8 && tokens == 5 && window > 128 && window <= 512) {
        return div_up(window, 32 / Geometry::DecodeSplitScale);
    }
    if (kv_dtype == DType::I8 && tokens == 6 && window > 128 && window <= 160) {
        return div_up(window, 24 / Geometry::DecodeSplitScale);
    }
    // Bc=64 is one CTA/SM on these model shapes. Keep the 8K grid at or below
    // one 170-SM wave after accounting for the geometry's KV-head count.
    if (kv_dtype == DType::I8 && tokens == 6 && window > 5000 && window <= 8198) {
        const std::int32_t splits   = div_up(window, 192 / Geometry::DecodeSplitScale);
        constexpr std::int32_t kMin = 4 * Geometry::DecodeSplitScale;
        constexpr std::int32_t kMax = 42 * Geometry::DecodeSplitScale;
        const std::int32_t clamped  = (splits > kMin) ? splits : kMin;
        return (clamped < kMax) ? clamped : kMax;
    }
    return gqa_small_t_split_upper_bound<Geometry>(window);
}

template <typename Geometry>
std::int32_t gqa_small_t_launch_capacity(GqaExecutionEnvelope envelope, std::int32_t tokens,
                                         DType dtype) {
    std::int32_t capacity = 0;
    const auto include    = [&](std::uint32_t window) {
        if (window < envelope.min_visible_keys || window > envelope.max_visible_keys) { return; }
        const auto splits =
            gqa_small_t_split_count<Geometry>(static_cast<std::int32_t>(window), tokens, dtype);
        capacity = capacity > splits ? capacity : splits;
    };
    include(envelope.min_visible_keys);
    include(envelope.max_visible_keys);
    // The policy is monotonic inside these finite segments and may drop when crossing a boundary.
    // Evaluating every segment end plus both interval ends gives the exact interval maximum.
    constexpr std::uint32_t ends[] = {128, 160, 512, 4096, 5000, 8198, 16390};
    for (const std::uint32_t end : ends) { include(end); }
    return capacity;
}

template <typename Geometry, int TokenTile, int WarpsPerCta, typename CacheInput>
void launch_tc_partial_bf16(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                            KVCacheLayerView cache, std::int32_t padded_context,
                            std::int32_t max_context, std::int32_t splits, Tensor& partial_acc,
                            Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const int tokens     = q.ne[2];
    const dim3 grid(Geometry::KVHeads, splits, 1);
    Tensor& cache_k = cache.k;
    Tensor& cache_v = cache.v;
    // bf16 kernel uses only static smem (no dynamic staging).
    gqa_attention_small_t_tc_partial_bf16_kernel<Geometry, TokenTile, WarpsPerCta, CacheInput>
        <<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), input,
            static_cast<const std::int32_t*>(pos.data), static_cast<__nv_bfloat16*>(cache_k.data),
            static_cast<__nv_bfloat16*>(cache_v.data), tokens, padded_context, max_context, scale,
            static_cast<__nv_bfloat16*>(partial_acc.data), static_cast<float*>(partial_m.data),
            static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, int TokenTile, typename CacheInput>
void launch_tc_partial_i8(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                          KVCacheLayerView cache, std::int32_t padded_context,
                          std::int32_t max_context, std::int32_t implementation_window,
                          std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                          Tensor& partial_l, cudaStream_t stream) {
    Tensor& cache_k       = cache.k;
    Tensor& cache_v       = cache.v;
    Tensor& cache_k_scale = cache.k_scale;
    Tensor& cache_v_scale = cache.v_scale;
    auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        const dim3 grid(Geometry::KVHeads, splits, 1);
        constexpr std::size_t kDynamicBytes =
            DynamicArena ? static_cast<std::size_t>(4 * KeyBlock * kGqaHeadDim) : 0u;
        if constexpr (DynamicArena) {
            static const cudaError_t attr = cudaFuncSetAttribute(
                gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                     MinBlocksPerSm, KeyBlock, DynamicArena,
                                                     CacheInput>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
            CUDA_CHECK(attr);
        }
        gqa_attention_decode_i8_tiled_kernel<Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm,
                                             KeyBlock, DynamicArena, CacheInput>
            <<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data), static_cast<std::int8_t*>(cache_k.data),
                static_cast<std::int8_t*>(cache_v.data), static_cast<__half*>(cache_k_scale.data),
                static_cast<__half*>(cache_v_scale.data), padded_context, max_context, scale,
                static_cast<__nv_bfloat16*>(partial_acc.data), static_cast<float*>(partial_m.data),
                static_cast<float*>(partial_l.data));
    };
    if constexpr (TokenTile == 6) {
        // Small grids need more warps per CTA. From 2K to 8K, Bc=64 halves key
        // loop iterations; dynamic smem avoids penalizing the long-context path.
        if (implementation_window > 128 && implementation_window <= 160) {
            launch.template operator()<24, 1, 32, false>();
        } else if (implementation_window <= 2054) {
            launch.template operator()<12, 1, 32, false>();
        } else if (implementation_window <= 8198) {
            launch.template operator()<12, 1, 64, true>();
        } else {
            launch.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            // Two Q row tiles for the 27B group of six.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<32, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<16, 1, 32, false>();
            } else {
                launch.template operator()<8, 2, 32, false>();
            }
        } else {
            // Three Q row tiles for the 35B group of eight. The 24/12-warp
            // routes retain eight/four consumer warps per tile; the 6-warp
            // route is reserved for long windows where CTA residency wins.
            if (implementation_window > 128 && implementation_window <= 512) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 4096) {
                launch.template operator()<12, 1, 32, false>();
            } else {
                launch.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (implementation_window <= 1029) {
            launch.template operator()<16, 1, 32, false>();
        } else {
            launch.template operator()<8, 2, 32, false>();
        }
    } else {
        launch.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool gqa_attention_uses_small_t(std::int32_t tokens) { return tokens >= 1 && tokens <= 6; }

std::int32_t gqa_attention_decode_splits(std::int32_t q_heads, std::int32_t kv_heads) {
    if (q_heads == Gqa27Geometry::QHeads && kv_heads == Gqa27Geometry::KVHeads) {
        return Gqa27Geometry::DecodeSplits;
    }
    if (q_heads == Gqa35Geometry::QHeads && kv_heads == Gqa35Geometry::KVHeads) {
        return Gqa35Geometry::DecodeSplits;
    }
    throw std::invalid_argument("gqa_attention_decode_splits: unsupported head geometry");
}

template <typename Geometry, typename CacheInput>
void gqa_attention_small_t_launch_for(const Tensor& q, CacheInput input, const Tensor& pos,
                                      float scale, KVCacheLayerView cache,
                                      GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                      Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                      cudaStream_t stream) {
    const auto padded_context        = static_cast<std::int32_t>(cache.padded_context);
    const auto max_context           = static_cast<std::int32_t>(cache.max_context);
    const auto implementation_window = static_cast<std::int32_t>(envelope.max_visible_keys);
    const auto splits = gqa_small_t_launch_capacity<Geometry>(envelope, q.ne[2], cache.dtype);

    // BF16 keeps its row-tile warp count; INT8 selects its producer/consumer
    // geometry inside launch_tc_partial_i8.
#define NINFER_GQA_SMALL_T_DISPATCH(TOKENS, WARPS)                                                 \
    do {                                                                                           \
        if (cache.dtype == DType::I8) {                                                            \
            launch_tc_partial_i8<Geometry, (TOKENS)>(q, input, pos, scale, cache, padded_context,  \
                                                     max_context, implementation_window, splits,   \
                                                     partial_acc, partial_m, partial_l, stream);   \
        } else {                                                                                   \
            launch_tc_partial_bf16<Geometry, (TOKENS), (WARPS)>(                                   \
                q, input, pos, scale, cache, padded_context, max_context, splits, partial_acc,     \
                partial_m, partial_l, stream);                                                     \
        }                                                                                          \
    } while (0)

    switch (q.ne[2]) {
    case 1:
        NINFER_GQA_SMALL_T_DISPATCH(1, 2);
        break;
    case 2:
        NINFER_GQA_SMALL_T_DISPATCH(2, 4);
        break;
    case 3:
        NINFER_GQA_SMALL_T_DISPATCH(3, 4);
        break;
    case 4:
        NINFER_GQA_SMALL_T_DISPATCH(4, 4);
        break;
    case 5:
        NINFER_GQA_SMALL_T_DISPATCH(5, 4);
        break;
    case 6:
        NINFER_GQA_SMALL_T_DISPATCH(6, 4);
        break;
    default:
        throw std::invalid_argument("gqa_attention_small_t_launch: unsupported T");
    }
#undef NINFER_GQA_SMALL_T_DISPATCH

    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = 64;
    const dim3 reduce_grid(Geometry::QHeads, div_up(kGqaHeadDim, kDChunk), q.ne[2]);
    if (cache.dtype == DType::I8) {
        gqa_attention_small_t_reduce_output_kernel<Geometry, kDChunk, true>
            <<<reduce_grid, kReduceBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(pos.data), q.ne[2], splits,
                static_cast<__nv_bfloat16*>(out.data));
    } else {
        gqa_attention_small_t_reduce_output_kernel<Geometry, kDChunk, false>
            <<<reduce_grid, kReduceBlock, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(pos.data), q.ne[2], splits,
                static_cast<__nv_bfloat16*>(out.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& pos, float scale, KVCacheLayerView cache,
                                  GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                  Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                  cudaStream_t stream) {
    const GqaAppendInput input{static_cast<const __nv_bfloat16*>(k.data),
                               static_cast<const __nv_bfloat16*>(v.data)};
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_attention_small_t_launch_for<Gqa27Geometry>(
            q, input, pos, scale, cache, envelope, partial_acc, partial_m, partial_l, out, stream);
        return;
    }
    gqa_attention_small_t_launch_for<Gqa35Geometry>(q, input, pos, scale, cache, envelope,
                                                    partial_acc, partial_m, partial_l, out, stream);
}

void gqa_attention_cached_small_t_launch(const Tensor& q, const Tensor& pos, float scale,
                                         const KVCacheLayerView& cache,
                                         GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                         Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                         cudaStream_t stream) {
    const GqaCachedInput input{};
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_attention_small_t_launch_for<Gqa27Geometry>(
            q, input, pos, scale, cache, envelope, partial_acc, partial_m, partial_l, out, stream);
        return;
    }
    gqa_attention_small_t_launch_for<Gqa35Geometry>(q, input, pos, scale, cache, envelope,
                                                    partial_acc, partial_m, partial_l, out, stream);
}

void gqa_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                          const Tensor& positions, float scale, KVCacheLayerView cache,
                          GqaExecutionEnvelope envelope, Tensor* partial_acc, Tensor* partial_m,
                          Tensor* partial_l, Tensor& out, cudaStream_t stream) {
    if (gqa_attention_uses_small_t(q.ne[2])) {
        if (partial_acc == nullptr || partial_m == nullptr || partial_l == nullptr) {
            throw std::invalid_argument("gqa_attention: small-T route requires workspace");
        }
        gqa_attention_small_t_launch(q, k, v, positions, scale, cache, envelope, *partial_acc,
                                     *partial_m, *partial_l, out, stream);
        return;
    }
    gqa_attention_prompt_launch(q, k, v, positions, scale, cache, out, stream);
}

} // namespace ninfer::ops::detail
