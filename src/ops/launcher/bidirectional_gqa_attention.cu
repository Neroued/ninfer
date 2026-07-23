#include "ops/launcher/bidirectional_gqa_attention.h"

#include "core/device.h"
#include "ops/kernel/bidirectional_gqa_attention.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <int Tokens, class Launch>
void dispatch_token_case(Launch&& launch) {
    constexpr int Warps = (Tokens + 3) / 4;
    launch.template operator()<Tokens, Warps>();
}

template <class Launch>
void dispatch_tokens(std::int32_t tokens, Launch&& launch) {
    switch (tokens) {
#define NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE(TOKENS)                                                \
    case TOKENS:                                                                                   \
        dispatch_token_case<TOKENS>(launch);                                                       \
        return
        NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE(1);
        NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE(2);
        NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE(3);
        NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE(4);
        NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE(5);
        NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE(6);
        NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE(7);
        NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE(8);
        NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE(9);
        NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE(10);
        NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE(11);
        NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE(12);
        NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE(13);
        NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE(14);
        NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE(15);
        NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE(16);
#undef NINFER_BIDIRECTIONAL_GQA_TOKEN_CASE
    default:
        throw std::invalid_argument("bidirectional_gqa_attention: unsupported T");
    }
}

} // namespace

BidirectionalGqaPlan bidirectional_gqa_resolve_plan(std::int32_t tokens,
                                                    GqaContextExecutionEnvelope envelope) {
    if (tokens < 1 || tokens > 16) {
        throw std::invalid_argument("bidirectional_gqa_attention plan: T must be 1..16");
    }
    if (envelope.min_context > envelope.max_context) {
        throw std::invalid_argument("bidirectional_gqa_attention plan: invalid envelope");
    }
    const std::int32_t warps = (tokens + 3) / 4;
    const bool direct        = envelope.max_context == 0;
    const std::int32_t key_block =
        direct || tokens <= 8 || envelope.max_context <= 65536u ? 32 : 64;
    std::int32_t split_limit = 32;
    if (tokens <= 8) {
        split_limit =
            envelope.max_context <= 131072u ? 32 : (envelope.max_context <= 196608u ? 48 : 64);
    } else if (key_block == 64) {
        split_limit =
            envelope.max_context <= 131072u ? 32 : (envelope.max_context <= 196608u ? 38 : 40);
    }
    const std::uint32_t envelope_tiles =
        (envelope.max_context + static_cast<std::uint32_t>(key_block) - 1u) /
        static_cast<std::uint32_t>(key_block);
    const std::int32_t splits =
        direct ? 1 : std::min(split_limit, std::max(1, static_cast<std::int32_t>(envelope_tiles)));
    return {
        .route          = direct ? BidirectionalGqaRoute::Direct : BidirectionalGqaRoute::SplitKv,
        .tokens         = tokens,
        .warps          = warps,
        .key_block      = key_block,
        .split_capacity = splits,
    };
}

const char* bidirectional_gqa_route_name(BidirectionalGqaRoute route) {
    switch (route) {
    case BidirectionalGqaRoute::Direct:
        return "direct";
    case BidirectionalGqaRoute::SplitKv:
        return "split_kv";
    }
    return "unknown";
}

void bidirectional_gqa_attention_launch(const Tensor& q, const Tensor& query_k,
                                        const Tensor& query_v, const Tensor& context_length,
                                        float scale, const KVCacheLayerView& context,
                                        const BidirectionalGqaPlan& plan, Tensor& partial_acc,
                                        Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                        cudaStream_t stream) {
    dispatch_tokens(q.ne[2], [&]<int Tokens, int Warps>() {
        const bool direct = plan.route == BidirectionalGqaRoute::Direct;
        if (plan.warps != Warps || (plan.key_block != 32 && plan.key_block != 64) ||
            plan.split_capacity < 1 || plan.split_capacity > kBidirectionalGqaMaxSplit ||
            (direct && plan.split_capacity != 1)) {
            throw std::invalid_argument("bidirectional_gqa_attention: inconsistent plan");
        }
        const auto launch_key_block = [&]<int KeyBlock>() {
            constexpr std::size_t SmemBytes =
                2u * KeyBlock * kBidirectionalGqaHeadDim * sizeof(__nv_bfloat16);
            if (direct) {
                const dim3 direct_grid(kBidirectionalGqaKVHeads, 1, 1);
                bidirectional_gqa_split_partial_kernel<Tokens, Warps, KeyBlock, true>
                    <<<direct_grid, Warps * 32, SmemBytes, stream>>>(
                        static_cast<const __nv_bfloat16*>(q.data),
                        static_cast<const __nv_bfloat16*>(query_k.data),
                        static_cast<const __nv_bfloat16*>(query_v.data),
                        static_cast<const std::int32_t*>(context_length.data),
                        static_cast<const __nv_bfloat16*>(context.k.data),
                        static_cast<const __nv_bfloat16*>(context.v.data),
                        static_cast<int>(context.padded_context),
                        static_cast<int>(context.max_context), 1, scale,
                        static_cast<__nv_bfloat16*>(partial_acc.data),
                        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                        static_cast<__nv_bfloat16*>(out.data));
                CUDA_CHECK(cudaGetLastError());
                return;
            }

            const dim3 partial_grid(kBidirectionalGqaKVHeads, plan.split_capacity, 1);
            bidirectional_gqa_split_partial_kernel<Tokens, Warps, KeyBlock, false>
                <<<partial_grid, Warps * 32, SmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(query_k.data),
                    static_cast<const __nv_bfloat16*>(query_v.data),
                    static_cast<const std::int32_t*>(context_length.data),
                    static_cast<const __nv_bfloat16*>(context.k.data),
                    static_cast<const __nv_bfloat16*>(context.v.data),
                    static_cast<int>(context.padded_context), static_cast<int>(context.max_context),
                    plan.split_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                    static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                    static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());
            const dim3 reduce_grid(kBidirectionalGqaQHeads, Tokens, 1);
            bidirectional_gqa_reduce_kernel<Tokens, KeyBlock><<<reduce_grid, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(context_length.data),
                static_cast<int>(context.max_context), plan.split_capacity,
                static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());
        };
        if (plan.key_block == 32) {
            launch_key_block.template operator()<32>();
        } else {
            launch_key_block.template operator()<64>();
        }
    });
}

} // namespace ninfer::ops::detail
