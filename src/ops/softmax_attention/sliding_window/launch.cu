#include "ops/softmax_attention/sliding_window/launch.h"

#include "core/device.h"
#include "ops/softmax_attention/sliding_window/kernel.cuh"

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

SlidingWindowAttentionPlan
sliding_window_attention_resolve_plan(std::uint32_t window, std::int32_t tokens, std::int32_t batch,
                                      SlidingWindowAttentionExecutionEnvelope envelope) {
    if (window != 2048 && window != 4096) {
        throw std::invalid_argument("sliding_window_attention plan: unsupported window");
    }
    if (tokens < 1 || tokens > 16 || batch < 1 || batch > 8) {
        throw std::invalid_argument(
            "sliding_window_attention plan: T must be 1..16 and B must be 1..8");
    }
    if (envelope.min_context > envelope.max_context) {
        throw std::invalid_argument("sliding_window_attention plan: invalid envelope");
    }
    // Keep enough independent KV-head CTAs for small batches, but limit repeated Q loads,
    // FP32 partial writes and merge traffic as the batch/query block grows.
    const int direct_context_limit = window == 2048 && tokens > 8 && batch >= 6 ? 128 : 96;
    const bool direct = envelope.max_context <= static_cast<std::uint32_t>(direct_context_limit);
    constexpr int key_block          = 32;
    const int reduce_warps           = window == 2048 ? 4 : 1;
    const std::uint32_t context_rows = std::min(envelope.max_context, window - 1u);
    const std::int32_t context_tiles = (context_rows + key_block - 1u) / key_block;
    int split_limit                  = 32;
    if (window == 2048) {
        if (tokens <= 4) {
            split_limit = batch <= 2 ? 32 : batch <= 6 ? 16 : 8;
        } else if (tokens <= 8) {
            split_limit = batch == 1 ? 32 : batch == 2 ? 16 : 8;
            if (batch >= 6) split_limit = std::min(split_limit, std::max(4, context_tiles / 4));
        } else {
            split_limit = batch <= 2 ? 16 : batch <= 5 ? 8 : 4;
            if (batch >= 2) {
                const int tiles_per_split = batch == 2 ? 2 : 4;
                split_limit = std::min(split_limit, std::max(4, context_tiles / tiles_per_split));
            }
        }
    }
    return {
        .route =
            direct ? SlidingWindowAttentionRoute::Direct : SlidingWindowAttentionRoute::SplitKv,
        .tokens         = tokens,
        .warps          = (tokens + 3) / 4,
        .key_block      = key_block,
        .reduce_warps   = reduce_warps,
        .split_capacity = direct ? 1 : std::min(split_limit, std::max(1, context_tiles)),
        .max_context    = static_cast<std::int32_t>(envelope.max_context),
        .window         = static_cast<std::int32_t>(window),
    };
}

const char* sliding_window_attention_route_name(SlidingWindowAttentionRoute route) {
    switch (route) {
    case SlidingWindowAttentionRoute::Direct:
        return "direct";
    case SlidingWindowAttentionRoute::SplitKv:
        return "split_kv";
    }
    return "unknown";
}

void sliding_window_attention_launch(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                                     const Tensor& positions, const Tensor& valid_columns,
                                     const Tensor& lanes, float scale,
                                     const CyclicKVCacheLayerView& context,
                                     const SlidingWindowAttentionPlan& plan, Tensor& partial_acc,
                                     Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                     cudaStream_t stream) {
    const bool direct = plan.route == SlidingWindowAttentionRoute::Direct;
    if ((plan.window != 2048 && plan.window != 4096) ||
        context.capacity != static_cast<std::uint32_t>(plan.window) ||
        plan.key_block != 32 || plan.reduce_warps != (plan.window == 2048 ? 4 : 1) ||
        plan.split_capacity < 1 || plan.split_capacity > kSlidingWindowMaxSplits ||
        (direct && plan.split_capacity != 1))
        throw std::invalid_argument("sliding_window_attention: inconsistent plan");
    dispatch_tokens(q.ne[2], [&]<int Tokens, int Warps>() {
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
                    static_cast<const std::int32_t*>(lanes.data),
                    static_cast<const __nv_bfloat16*>(context.k.data),
                    static_cast<const __half*>(context.v.data),
                    static_cast<int>(context.padded_capacity), plan.window - 1, plan.max_context,
                    Direct ? 1 : plan.split_capacity, scale,
                    static_cast<float*>(partial_acc.data), static_cast<float*>(partial_m.data),
                    static_cast<float*>(partial_l.data), static_cast<__nv_bfloat16*>(out.data));
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
