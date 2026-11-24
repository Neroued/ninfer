#include "ops/softmax_attention/sliding_window/launch.h"

#include "core/device.h"
#include "ops/common/host/operands.h"
#include "ops/softmax_attention/kernels/context_query/cyclic.cuh"

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
#define NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(TOKENS)                                         \
    case TOKENS:                                                                                   \
        dispatch_token_case<TOKENS>(launch);                                                       \
        return
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(1);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(2);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(3);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(4);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(5);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(6);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(7);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(8);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(9);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(10);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(11);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(12);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(13);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(14);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(15);
        NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE(16);
#undef NINFER_SLIDING_WINDOW_ATTENTION_TOKEN_CASE
    default:
        throw std::invalid_argument("sliding_window_attention: unsupported T");
    }
}

} // namespace

void launch_sliding_window_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                                     const Tensor& positions, const Tensor& valid_columns,
                                     float scale, const ConstCyclicKVCacheLayerView& context,
                                     const SlidingWindowAttentionPlan& plan, Tensor& partial_acc,
                                     Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                     cudaStream_t stream) {
    const bool direct = plan.route == SlidingWindowAttentionRoute::Direct;
    if ((plan.window != 2048 && plan.window != 4096) ||
        context.capacity != static_cast<std::uint32_t>(plan.window) || plan.key_block != 32 ||
        plan.reduce_warps != (plan.window == 2048 ? 4 : 1) || plan.split_capacity < 1 ||
        plan.split_capacity > kSlidingWindowMaxSplits || (direct && plan.split_capacity != 1))
        throw std::invalid_argument("sliding_window_attention: inconsistent plan");
    dispatch_tokens(plan.tokens, [&]<int Tokens, int Warps>() {
        constexpr int KeyBlock = 32;
        constexpr std::size_t SmemBytes =
            2u * KeyBlock * kContextQueryHeadDim * sizeof(__nv_bfloat16);
        if (plan.warps != Warps)
            throw std::invalid_argument("sliding_window_attention: inconsistent query layout");
        // Window determines visibility and ring addressing, not the MMA storage layout.
        const auto partial = [&]<bool Direct>() {
            const dim3 grid(kContextQueryKVHeads, Direct ? 1 : plan.split_capacity, q.ne[3]);
            sliding_window_attention_split_partial_kernel<Tokens, Warps, KeyBlock, Direct>
                <<<grid, Warps * 32, SmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(query_k.data),
                    static_cast<const __nv_bfloat16*>(query_v.data),
                    static_cast<const std::int32_t*>(positions.data),
                    static_cast<const std::int32_t*>(valid_columns.data),
                    static_cast<const std::int64_t*>(context.k.records.byte_offsets.data),
                    static_cast<const std::int64_t*>(context.v.records.byte_offsets.data),
                    common::operand::state_field_base<const __nv_bfloat16>(context.k),
                    common::operand::state_field_base<const __half>(context.v),
                    static_cast<int>(context.padded_capacity), plan.window - 1, plan.max_context,
                    Direct ? 1 : plan.split_capacity, scale, static_cast<float*>(partial_acc.data),
                    static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                    static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());
        };
        if (direct) {
            partial.template operator()<true>();
            return;
        }
        partial.template operator()<false>();
        const auto reduce = [&]<int ReduceWarps>() {
            constexpr int Rows = kContextQueryQHeads * Tokens;
            const dim3 grid((Rows + ReduceWarps - 1) / ReduceWarps, 1, q.ne[3]);
            sliding_window_attention_reduce_kernel<Tokens, KeyBlock, ReduceWarps>
                <<<grid, ReduceWarps * 32, 0, stream>>>(
                    static_cast<const float*>(partial_acc.data),
                    static_cast<const float*>(partial_m.data),
                    static_cast<const float*>(partial_l.data),
                    static_cast<const std::int32_t*>(positions.data),
                    static_cast<const std::int32_t*>(valid_columns.data), plan.window - 1,
                    plan.max_context, plan.split_capacity, static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());
        };
        if (plan.reduce_warps == 4)
            reduce.template operator()<4>();
        else
            reduce.template operator()<1>();
    });
}

} // namespace ninfer::ops::detail
