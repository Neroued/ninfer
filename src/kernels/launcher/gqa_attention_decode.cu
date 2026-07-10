// qus::kernels - split-KV GQA small-T launcher and unified route dispatcher.
#include "kernels/launcher/gqa_attention.h"

#include "kernels/kernel/gqa_attention_decode.cuh"
#include "kernels/kernel/gqa_attention_decode_bf16.cuh"
#include "kernels/kernel/gqa_attention_decode_i8.cuh"
#include "qus/core/device.h" // CUDA_CHECK
#include "qus/kernels/gqa_attention.h"

#include <cstdint>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

std::int32_t ceil_div_i32(std::int32_t x, std::int32_t y) { return (x + y - 1) / y; }

// Split-KV grid sizing keyed on the actual attention window (kv.pos + tokens),
// not the allocation ceiling. Over-splitting inflates the partial scratch that
// the reduce kernel must stream back, so the count must track the live context.
// Mirrors the device-side gqa_small_t_active_splits tiers so host launch and
// in-kernel early-exit agree. Same schedule for every T, including decode T=1.
std::int32_t gqa_small_t_split_upper_bound(std::int32_t window) {
    if (window <= 0) { return kGqaDecodeSplits; }

    constexpr std::int32_t kMinSplits = 4;
    std::int32_t splits               = kMinSplits;

    const auto include_tier = [&](std::int32_t window_limit, std::int32_t target_keys_per_split) {
        const std::int32_t tier_window = (window < window_limit) ? window : window_limit;
        if (tier_window > 0) {
            const std::int32_t tier_splits = ceil_div_i32(tier_window, target_keys_per_split);
            splits                         = (splits > tier_splits) ? splits : tier_splits;
        }
    };

    include_tier(4096, 64);
    if (window > 4096) { include_tier(8198, 128); }
    if (window > 8198) { include_tier(16390, 256); }
    if (window > 16390) { include_tier(window, 480); }

    return (splits < kGqaDecodeSplits) ? splits : kGqaDecodeSplits;
}

template <int TokenTile, int WarpsPerCta>
void launch_tc_partial_bf16(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                            float scale, KVCache& kv, int layer, std::int32_t padded_context,
                            std::int32_t max_context, std::int32_t window, Tensor& partial_acc,
                            Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const int tokens     = q.ne[2];
    const int splits     = gqa_small_t_split_upper_bound(window);
    const dim3 grid(kGqaKVHeads, splits, 1);
    Tensor& cache_k = kv.k[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v = kv.v[static_cast<std::uint32_t>(layer)];
    // bf16 kernel uses only static smem (no dynamic staging).
    gqa_attention_small_t_tc_partial_bf16_kernel<TokenTile, WarpsPerCta>
        <<<grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat16*>(v.data), static_cast<const std::int32_t*>(pos.data),
            static_cast<__nv_bfloat16*>(cache_k.data), static_cast<__nv_bfloat16*>(cache_v.data),
            tokens, padded_context, max_context, scale,
            static_cast<__nv_bfloat16*>(partial_acc.data), static_cast<float*>(partial_m.data),
            static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

template <int TokenTile>
void launch_tc_partial_i8(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                          float scale, KVCache& kv, int layer, std::int32_t padded_context,
                          std::int32_t max_context, std::int32_t window, Tensor& partial_acc,
                          Tensor& partial_m, Tensor& partial_l, cudaStream_t stream) {
    const int splits = gqa_small_t_split_upper_bound(window);
    const dim3 grid(kGqaKVHeads, splits, 1);
    Tensor& cache_k            = kv.k[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v            = kv.v[static_cast<std::uint32_t>(layer)];
    Tensor& cache_k_scale      = kv.k_scale[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v_scale      = kv.v_scale[static_cast<std::uint32_t>(layer)];
    constexpr int kTiledWarps  = (TokenTile == 6) ? 6 : 8;
    constexpr int kMinBlocksSm = 2;
    gqa_attention_decode_i8_tiled_kernel<TokenTile, kTiledWarps, kMinBlocksSm>
        <<<grid, kTiledWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
            static_cast<const __nv_bfloat16*>(v.data), static_cast<const std::int32_t*>(pos.data),
            static_cast<std::int8_t*>(cache_k.data), static_cast<std::int8_t*>(cache_v.data),
            static_cast<__half*>(cache_k_scale.data), static_cast<__half*>(cache_v_scale.data),
            padded_context, max_context, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
            static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool gqa_attention_uses_small_t(std::int32_t tokens) { return tokens >= 1 && tokens <= 6; }

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& pos, float scale, KVCache& kv, int layer,
                                  Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                  Tensor& out, cudaStream_t stream) {
    const auto padded_context = static_cast<std::int32_t>(kv.padded_context);
    const auto max_context    = static_cast<std::int32_t>(kv.max_context);
    // Split count tracks the live attention window (kv.pos is the pre-round base
    // position; positions run [base, base+T)), so decode/verify at a short context
    // inside a large allocation is not over-split. The kernel still receives the
    // real max_context for cache-bounds checks.
    const auto window = static_cast<std::int32_t>(kv.pos) + q.ne[2];

    // BF16 keeps its row-tile warp count; INT8 selects its producer/consumer
    // geometry inside launch_tc_partial_i8.
#define QUS_GQA_SMALL_T_DISPATCH(TOKENS, WARPS)                                                    \
    do {                                                                                           \
        if (kv.dtype == DType::I8) {                                                               \
            launch_tc_partial_i8<(TOKENS)>(q, k, v, pos, scale, kv, layer, padded_context,         \
                                           max_context, window, partial_acc, partial_m, partial_l, \
                                           stream);                                                \
        } else {                                                                                   \
            launch_tc_partial_bf16<(TOKENS), (WARPS)>(q, k, v, pos, scale, kv, layer,              \
                                                      padded_context, max_context, window,         \
                                                      partial_acc, partial_m, partial_l, stream);  \
        }                                                                                          \
    } while (0)

    switch (q.ne[2]) {
    case 1:
        QUS_GQA_SMALL_T_DISPATCH(1, 2);
        break;
    case 2:
        QUS_GQA_SMALL_T_DISPATCH(2, 4);
        break;
    case 3:
        QUS_GQA_SMALL_T_DISPATCH(3, 4);
        break;
    case 4:
        QUS_GQA_SMALL_T_DISPATCH(4, 4);
        break;
    case 5:
        QUS_GQA_SMALL_T_DISPATCH(5, 4);
        break;
    case 6:
        QUS_GQA_SMALL_T_DISPATCH(6, 4);
        break;
    default:
        throw std::invalid_argument("gqa_attention_small_t_launch: unsupported T");
    }
#undef QUS_GQA_SMALL_T_DISPATCH

    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = 64;
    const int partial_splits   = gqa_small_t_split_upper_bound(window);
    const dim3 reduce_grid(kGqaQHeads, (kGqaHeadDim + kDChunk - 1) / kDChunk, q.ne[2]);
    gqa_attention_small_t_reduce_output_kernel<kDChunk><<<reduce_grid, kReduceBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(partial_acc.data),
        static_cast<const float*>(partial_m.data), static_cast<const float*>(partial_l.data),
        static_cast<const std::int32_t*>(pos.data), q.ne[2], partial_splits,
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void gqa_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                          const Tensor& positions, float scale, KVCache& kv, int layer,
                          Tensor* partial_acc, Tensor* partial_m, Tensor* partial_l, Tensor& out,
                          cudaStream_t stream) {
    if (gqa_attention_uses_small_t(q.ne[2])) {
        if (partial_acc == nullptr || partial_m == nullptr || partial_l == nullptr) {
            throw std::invalid_argument("gqa_attention: small-T route requires workspace");
        }
        gqa_attention_small_t_launch(q, k, v, positions, scale, kv, layer, *partial_acc, *partial_m,
                                     *partial_l, out, stream);
        return;
    }
    gqa_attention_prompt_launch(q, k, v, positions, scale, kv, layer, out, stream);
}

} // namespace qus::kernels::detail
