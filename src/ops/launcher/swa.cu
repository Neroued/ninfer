#include "ops/launcher/swa.h"

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
#define NINFER_SWA_TOKEN_CASE(TOKENS)                                                              \
    case TOKENS:                                                                                   \
        dispatch_token_case<TOKENS>(launch);                                                       \
        return
        NINFER_SWA_TOKEN_CASE(1);
        NINFER_SWA_TOKEN_CASE(2);
        NINFER_SWA_TOKEN_CASE(3);
        NINFER_SWA_TOKEN_CASE(4);
        NINFER_SWA_TOKEN_CASE(5);
        NINFER_SWA_TOKEN_CASE(6);
        NINFER_SWA_TOKEN_CASE(7);
        NINFER_SWA_TOKEN_CASE(8);
        NINFER_SWA_TOKEN_CASE(9);
        NINFER_SWA_TOKEN_CASE(10);
        NINFER_SWA_TOKEN_CASE(11);
        NINFER_SWA_TOKEN_CASE(12);
        NINFER_SWA_TOKEN_CASE(13);
        NINFER_SWA_TOKEN_CASE(14);
        NINFER_SWA_TOKEN_CASE(15);
        NINFER_SWA_TOKEN_CASE(16);
#undef NINFER_SWA_TOKEN_CASE
    default:
        throw std::invalid_argument("swa: unsupported T");
    }
}

} // namespace

SwaPlan swa_resolve_plan(std::int32_t tokens, SwaContextExecutionEnvelope envelope) {
    if (tokens < 1 || tokens > 16) { throw std::invalid_argument("swa plan: T must be 1..16"); }
    if (envelope.min_context > envelope.max_context) {
        throw std::invalid_argument("swa plan: invalid envelope");
    }
    // Graph envelopes whose longest context fits three key tiles avoid the second kernel and
    // workspace round trip. At four tiles, split-KV is already faster for every qualified T.
    constexpr std::uint32_t direct_context_limit = 96;
    const bool direct                            = envelope.max_context <= direct_context_limit;
    constexpr std::int32_t key_block             = 32;
    const std::uint32_t context_rows             = std::min(envelope.max_context, 4095u);
    const std::int32_t context_tiles =
        static_cast<std::int32_t>((context_rows + key_block - 1u) / key_block);
    constexpr std::int32_t split_limit = 32;
    return {
        .route          = direct ? SwaRoute::Direct : SwaRoute::SplitKv,
        .tokens         = tokens,
        .warps          = (tokens + 3) / 4,
        .key_block      = key_block,
        .split_capacity = direct ? 1 : std::min(split_limit, std::max(1, context_tiles)),
        .max_context    = static_cast<std::int32_t>(envelope.max_context),
    };
}

const char* swa_route_name(SwaRoute route) {
    switch (route) {
    case SwaRoute::Direct:
        return "direct";
    case SwaRoute::SplitKv:
        return "split_kv";
    }
    return "unknown";
}

void swa_launch(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                const Tensor& positions, float scale, const CyclicKVCacheLayerView& context,
                const SwaPlan& plan, Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                Tensor& out, cudaStream_t stream) {
    dispatch_tokens(q.ne[2], [&]<int Tokens, int Warps>() {
        const bool direct = plan.route == SwaRoute::Direct;
        if (plan.warps != Warps || (plan.key_block != 32 && plan.key_block != 64) ||
            plan.split_capacity < 1 || plan.split_capacity > kSwaMaxCandidateSplit ||
            (direct && plan.split_capacity != 1)) {
            throw std::invalid_argument("swa: inconsistent plan");
        }
        const auto launch_key_block = [&]<int KeyBlock>() {
            constexpr std::size_t SmemBytes =
                2u * KeyBlock * kBidirectionalGqaHeadDim * sizeof(__nv_bfloat16);
            if (direct) {
                const dim3 direct_grid(kBidirectionalGqaKVHeads, 1, 1);
                swa_split_partial_kernel<Tokens, Warps, KeyBlock, true>
                    <<<direct_grid, Warps * 32, SmemBytes, stream>>>(
                        static_cast<const __nv_bfloat16*>(q.data),
                        static_cast<const __nv_bfloat16*>(query_k.data),
                        static_cast<const __nv_bfloat16*>(query_v.data),
                        static_cast<const std::int32_t*>(positions.data),
                        static_cast<const __nv_bfloat16*>(context.k.data),
                        static_cast<const __nv_bfloat16*>(context.v.data),
                        static_cast<int>(context.padded_capacity), plan.max_context, 1, scale,
                        static_cast<__nv_bfloat16*>(partial_acc.data),
                        static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                        static_cast<__nv_bfloat16*>(out.data));
                CUDA_CHECK(cudaGetLastError());
                return;
            }

            const dim3 partial_grid(kBidirectionalGqaKVHeads, plan.split_capacity, 1);
            swa_split_partial_kernel<Tokens, Warps, KeyBlock, false>
                <<<partial_grid, Warps * 32, SmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const __nv_bfloat16*>(query_k.data),
                    static_cast<const __nv_bfloat16*>(query_v.data),
                    static_cast<const std::int32_t*>(positions.data),
                    static_cast<const __nv_bfloat16*>(context.k.data),
                    static_cast<const __nv_bfloat16*>(context.v.data),
                    static_cast<int>(context.padded_capacity), plan.max_context,
                    plan.split_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                    static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                    static_cast<__nv_bfloat16*>(out.data));
            CUDA_CHECK(cudaGetLastError());

            constexpr int ReduceWarps = 1;
            constexpr int ReduceRows  = kBidirectionalGqaQHeads * Tokens;
            const dim3 reduce_grid((ReduceRows + ReduceWarps - 1) / ReduceWarps, 1, 1);
            swa_reduce_kernel<Tokens, KeyBlock, ReduceWarps>
                <<<reduce_grid, ReduceWarps * 32, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(partial_acc.data),
                    static_cast<const float*>(partial_m.data),
                    static_cast<const float*>(partial_l.data),
                    static_cast<const std::int32_t*>(positions.data), plan.max_context,
                    plan.split_capacity, static_cast<__nv_bfloat16*>(out.data));
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
