#include "ops/softmax_attention/dense/context/launch.h"

#include "core/device.h"
#include "ops/softmax_attention/dense/context/kernel.cuh"

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
#define NINFER_CONTEXT_ATTENTION_TOKEN_CASE(TOKENS)                                                \
    case TOKENS:                                                                                   \
        dispatch_token_case<TOKENS>(launch);                                                       \
        return
        NINFER_CONTEXT_ATTENTION_TOKEN_CASE(1);
        NINFER_CONTEXT_ATTENTION_TOKEN_CASE(2);
        NINFER_CONTEXT_ATTENTION_TOKEN_CASE(3);
        NINFER_CONTEXT_ATTENTION_TOKEN_CASE(4);
        NINFER_CONTEXT_ATTENTION_TOKEN_CASE(5);
        NINFER_CONTEXT_ATTENTION_TOKEN_CASE(6);
        NINFER_CONTEXT_ATTENTION_TOKEN_CASE(7);
        NINFER_CONTEXT_ATTENTION_TOKEN_CASE(8);
        NINFER_CONTEXT_ATTENTION_TOKEN_CASE(9);
        NINFER_CONTEXT_ATTENTION_TOKEN_CASE(10);
        NINFER_CONTEXT_ATTENTION_TOKEN_CASE(11);
        NINFER_CONTEXT_ATTENTION_TOKEN_CASE(12);
        NINFER_CONTEXT_ATTENTION_TOKEN_CASE(13);
        NINFER_CONTEXT_ATTENTION_TOKEN_CASE(14);
        NINFER_CONTEXT_ATTENTION_TOKEN_CASE(15);
        NINFER_CONTEXT_ATTENTION_TOKEN_CASE(16);
#undef NINFER_CONTEXT_ATTENTION_TOKEN_CASE
    default:
        throw std::invalid_argument("context_softmax_attention: unsupported T");
    }
}

} // namespace

ContextAttentionPlan context_attention_resolve_plan(std::int32_t tokens,
                                                    ContextAttentionExecutionEnvelope envelope) {
    if (tokens < 1 || tokens > 16) {
        throw std::invalid_argument("context_softmax_attention plan: T must be 1..16");
    }
    if (envelope.min_context > envelope.max_context) {
        throw std::invalid_argument("context_softmax_attention plan: invalid envelope");
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
        .route          = direct ? ContextAttentionRoute::Direct : ContextAttentionRoute::SplitKv,
        .tokens         = tokens,
        .warps          = warps,
        .key_block      = key_block,
        .split_capacity = splits,
    };
}

const char* context_attention_route_name(ContextAttentionRoute route) {
    switch (route) {
    case ContextAttentionRoute::Direct:
        return "direct";
    case ContextAttentionRoute::SplitKv:
        return "split_kv";
    }
    return "unknown";
}

void context_attention_launch(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                              const Tensor& context_lengths, const Tensor& valid_columns,
                              const Tensor& table_rows, float scale,
                              const PagedKVBatchLayerView& context,
                              const ContextAttentionPlan& plan, Tensor& partial_acc,
                              Tensor& partial_m, Tensor& partial_l, Tensor& out,
                              cudaStream_t stream) {
    dispatch_tokens(q.ne[2], [&]<int Tokens, int Warps>() {
        const bool direct = plan.route == ContextAttentionRoute::Direct;
        if (plan.warps != Warps || plan.split_capacity < 1 ||
            plan.split_capacity > kContextQueryMaxSplit) {
            throw std::invalid_argument("context_softmax_attention: inconsistent plan");
        }
        if (direct) {
            if (plan.key_block != 32 || plan.split_capacity != 1) {
                throw std::invalid_argument("context_softmax_attention: inconsistent plan");
            }
            constexpr int KeyBlock = 32;
            constexpr std::size_t SmemBytes =
                2u * KeyBlock * kContextQueryHeadDim * sizeof(__nv_bfloat16);
            const dim3 direct_grid(kContextQueryKVHeads, 1, q.ne[3]);
            context_attention_split_partial_kernel<Tokens, Warps, KeyBlock, true>
                <<<direct_grid, Warps * 32, SmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(query_k.data),
                    static_cast<const __nv_bfloat16*>(query_v.data),
                    static_cast<const std::int32_t*>(context_lengths.data),
                    static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int32_t*>(table_rows.data),
                    static_cast<const __nv_bfloat16*>(context.k_pages.data),
                    static_cast<const __half*>(context.v_pages.data),
                    static_cast<const std::int32_t*>(context.block_tables.data),
                    context.k_pages.ne[2], context.block_tables.ne[0],
                    context.block_tables.ne[0] * kPagedKVPageSize, 1, scale,
                    static_cast<__nv_bfloat16*>(partial_acc.data),
                    static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                    static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());
            return;
        }

        if (plan.route != ContextAttentionRoute::SplitKv) {
            throw std::invalid_argument("context_softmax_attention: inconsistent plan");
        }

        const auto launch_split = [&]<int KeyBlock>() {
            constexpr std::size_t SmemBytes =
                2u * KeyBlock * kContextQueryHeadDim * sizeof(__nv_bfloat16);
            const dim3 partial_grid(kContextQueryKVHeads, plan.split_capacity, q.ne[3]);
            context_attention_split_partial_kernel<Tokens, Warps, KeyBlock, false>
                <<<partial_grid, Warps * 32, SmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(query_k.data),
                    static_cast<const __nv_bfloat16*>(query_v.data),
                    static_cast<const std::int32_t*>(context_lengths.data),
                    static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int32_t*>(table_rows.data),
                    static_cast<const __nv_bfloat16*>(context.k_pages.data),
                    static_cast<const __half*>(context.v_pages.data),
                    static_cast<const std::int32_t*>(context.block_tables.data),
                    context.k_pages.ne[2], context.block_tables.ne[0],
                    context.block_tables.ne[0] * kPagedKVPageSize, plan.split_capacity, scale,
                    static_cast<__nv_bfloat16*>(partial_acc.data),
                    static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                    static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());
            const dim3 reduce_grid(kContextQueryQHeads, Tokens, q.ne[3]);
            context_attention_reduce_kernel<Tokens, KeyBlock><<<reduce_grid, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(context_lengths.data),
                static_cast<const std::int32_t*>(valid_columns.data),
                context.block_tables.ne[0] * kPagedKVPageSize, plan.split_capacity,
                static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());
        };
        if (plan.key_block == 32) {
            launch_split.template operator()<32>();
            return;
        }
        if constexpr (Tokens > 8) {
            if (plan.key_block == 64) {
                launch_split.template operator()<64>();
                return;
            }
        }
        throw std::invalid_argument("context_softmax_attention: inconsistent plan");
    });
}

} // namespace ninfer::ops::detail
