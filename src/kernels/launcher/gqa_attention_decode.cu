// qus::kernels - split-KV GQA small-T launcher and unified route dispatcher.
#include "kernels/launcher/gqa_attention.h"

#include "kernels/kernel/gqa_attention_decode.cuh"
#include "qus/core/device.h" // CUDA_CHECK
#include "qus/kernels/gqa_attention.h"

#include <cstdint>
#include <stdexcept>

namespace qus::kernels::detail {
namespace {

std::int32_t ceil_div_i32(std::int32_t x, std::int32_t y) {
    return (x + y - 1) / y;
}

std::int32_t gqa_small_t_split_upper_bound(std::int32_t max_context, std::int32_t tokens) {
    if (tokens <= 1) { return kGqaDecodeSplits; }
    if (max_context <= 0) { return kGqaDecodeSplits; }

    constexpr std::int32_t kMinSplits = 4;
    std::int32_t splits               = kMinSplits;
    if (tokens <= 5) {
        constexpr std::int32_t kSmallWindowLimit     = 4096;
        constexpr std::int32_t kSmallTargetKeysSplit = 32;
        const std::int32_t small_window =
            (max_context < kSmallWindowLimit) ? max_context : kSmallWindowLimit;
        const std::int32_t small_splits = ceil_div_i32(small_window, kSmallTargetKeysSplit);
        splits                          = (splits > small_splits) ? splits : small_splits;
        if (max_context > kSmallWindowLimit) {
            constexpr std::int32_t kLargeTargetKeysSplit = 480;
            const std::int32_t large_splits = ceil_div_i32(max_context, kLargeTargetKeysSplit);
            splits                          = (splits > large_splits) ? splits : large_splits;
        }
    } else {
        constexpr std::int32_t kTargetKeysSplit = 512;
        splits = ceil_div_i32(max_context, kTargetKeysSplit);
        splits = (splits > kMinSplits) ? splits : kMinSplits;
    }
    return (splits < kGqaDecodeSplits) ? splits : kGqaDecodeSplits;
}

template <int TokenTile, int WarpsPerCta>
void launch_tc_partial(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                       float scale, Tensor& cache_k, Tensor& cache_v, std::int32_t padded_context,
                       std::int32_t max_context, Tensor& partial_acc, Tensor& partial_m,
                       Tensor& partial_l, cudaStream_t stream) {
    constexpr int kBlock = 32 * WarpsPerCta;
    const int tokens     = q.ne[2];
    const int splits     = gqa_small_t_split_upper_bound(max_context, tokens);
    const dim3 grid(kGqaKVHeads, splits, 1);
    gqa_attention_small_t_tc_partial_kernel<TokenTile, WarpsPerCta><<<grid, kBlock, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const std::int32_t*>(pos.data),
        static_cast<__nv_bfloat16*>(cache_k.data), static_cast<__nv_bfloat16*>(cache_v.data),
        tokens, padded_context, max_context, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool gqa_attention_uses_small_t(std::int32_t tokens) { return tokens >= 1 && tokens <= 6; }

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& pos, float scale, KVCache& kv, int layer,
                                  Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                  Tensor& out, cudaStream_t stream) {
    Tensor& cache_k = kv.k[static_cast<std::uint32_t>(layer)];
    Tensor& cache_v = kv.v[static_cast<std::uint32_t>(layer)];

    const auto padded_context = static_cast<std::int32_t>(kv.padded_context);
    const auto max_context    = static_cast<std::int32_t>(kv.max_context);

    switch (q.ne[2]) {
    case 1:
        launch_tc_partial<1, 2>(q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context,
                                partial_acc, partial_m, partial_l, stream);
        break;
    case 2:
        launch_tc_partial<2, 4>(q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context,
                                partial_acc, partial_m, partial_l, stream);
        break;
    case 3:
        launch_tc_partial<3, 4>(q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context,
                                partial_acc, partial_m, partial_l, stream);
        break;
    case 4:
        launch_tc_partial<4, 4>(q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context,
                                partial_acc, partial_m, partial_l, stream);
        break;
    case 5:
        launch_tc_partial<5, 4>(q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context,
                                partial_acc, partial_m, partial_l, stream);
        break;
    case 6:
        launch_tc_partial<6, 4>(q, k, v, pos, scale, cache_k, cache_v, padded_context, max_context,
                                partial_acc, partial_m, partial_l, stream);
        break;
    default:
        throw std::invalid_argument("gqa_attention_small_t_launch: unsupported T");
    }

    constexpr int kReduceBlock = 256;
    constexpr int kDChunk      = 64;
    const int partial_splits   = gqa_small_t_split_upper_bound(max_context, q.ne[2]);
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
