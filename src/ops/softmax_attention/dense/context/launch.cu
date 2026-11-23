#include "ops/softmax_attention/dense/context/launch.h"

#include "core/device.h"
#include "ops/softmax_attention/kernels/context_query/paged.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

__global__ void context_attention_idle_kernel() {}

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

void launch_context_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                              const Tensor& context_lengths, const Tensor& valid_columns,
                              const Tensor& table_rows, float scale,
                              const ConstPagedKVBatchLayerView& context,
                              const ContextAttentionPlan& plan, Tensor& partial_acc,
                              Tensor& partial_m, Tensor& partial_l, Tensor& out,
                              cudaStream_t stream) {
    const int logical_pages = context.k_pages.records.page_offsets.ne[0];
    // This guard protects page lookups for both tile sizes. The caller supplies live lengths
    // within the selected bucket; at fixed page-table shape the guard is the same for all buckets.
    const int max_context =
        static_cast<int>(std::min<std::int64_t>(context_bucket_tokens(kContextAttentionMaxBucket),
                                                std::int64_t(logical_pages) * kPagedKVPageSize));
    dispatch_tokens(plan.tokens, [&]<int Tokens, int Warps>() {
        if (plan.split_capacity < 1 || plan.split_capacity > kContextQueryMaxSplit) {
            throw std::invalid_argument("context_softmax_attention: inconsistent plan");
        }
        const auto launch_partial = [&]<int KeyBlock, bool Direct>() {
            constexpr std::size_t SmemBytes =
                2u * KeyBlock * kContextQueryHeadDim * sizeof(__nv_bfloat16);
            const dim3 grid(kContextQueryKVHeads, plan.split_capacity, q.ne[3]);
            context_attention_split_partial_kernel<Tokens, Warps, KeyBlock, Direct>
                <<<grid, Warps * 32, SmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(query_k.data),
                    static_cast<const __nv_bfloat16*>(query_v.data),
                    static_cast<const std::int32_t*>(context_lengths.data),
                    static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int32_t*>(table_rows.data),
                    static_cast<const __nv_bfloat16*>(context.k_pages.data()),
                    static_cast<const __half*>(context.v_pages.data()),
                    static_cast<const std::int64_t*>(context.k_pages.records.page_offsets.data),
                    logical_pages, max_context, plan.split_capacity, scale,
                    static_cast<__nv_bfloat16*>(partial_acc.data),
                    static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                    static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());
        };
        if (plan.route == ContextAttentionRoute::Direct) {
            launch_partial.template operator()<32, true>();
            context_attention_idle_kernel<<<1, 1, 0, stream>>>();
            CUDA_CHECK(cudaGetLastError());
            return;
        }

        const auto launch_split = [&]<int KeyBlock>() {
            launch_partial.template operator()<KeyBlock, false>();
            const dim3 reduce_grid(kContextQueryQHeads * Tokens / 4, 1, q.ne[3]);
            context_attention_reduce_kernel<Tokens, KeyBlock><<<reduce_grid, 128, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(partial_acc.data),
                static_cast<const float*>(partial_m.data),
                static_cast<const float*>(partial_l.data),
                static_cast<const std::int32_t*>(context_lengths.data),
                static_cast<const std::int32_t*>(valid_columns.data), max_context,
                plan.split_capacity, static_cast<__nv_bfloat16*>(out.data));
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
