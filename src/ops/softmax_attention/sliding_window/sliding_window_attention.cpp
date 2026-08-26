#include "ninfer/ops/sliding_window_attention.h"

#include "core/layout.h"
#include "ops/softmax_attention/sliding_window/launch.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 128;
constexpr std::int32_t kQHeads  = 32;
constexpr std::int32_t kKVHeads = 8;
constexpr std::int32_t kWindow  = 4096;
constexpr float kExpectedScale  = 0.08838834764831844055f;

void require_profile(AttentionHeadGeometry geometry, std::uint32_t window, const char* op) {
    if (!valid_attention_head_geometry(geometry) || geometry.head_dim != kHeadDim ||
        geometry.query_heads != kQHeads || geometry.kv_heads != kKVHeads) {
        throw std::invalid_argument(std::string(op) + ": unsupported head geometry");
    }
    if (window != static_cast<std::uint32_t>(kWindow)) {
        throw std::invalid_argument(std::string(op) + ": supported window is 4096");
    }
}

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

void validate_context(const CyclicKVCacheLayerView& context, const char* op) {
    if (context.num_kv_heads != kKVHeads || context.head_dim != kHeadDim ||
        context.capacity != kWindow || context.padded_capacity < context.capacity ||
        context.lane_capacity <= 0) {
        throw std::invalid_argument(std::string(op) + ": invalid cyclic context");
    }
    if (context.padded_capacity >
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string(op) + ": padded capacity exceeds int32");
    }
    const auto padded = static_cast<std::int32_t>(context.padded_capacity);
    if (context.k.dtype != DType::BF16 || context.v.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": context K/V must be BF16");
    }
    require_shape(context.k, kHeadDim, padded, kKVHeads, context.lane_capacity, op, "context k");
    require_shape(context.v, kHeadDim, padded, kKVHeads, context.lane_capacity, op, "context v");
    require_contiguous_nonnull(context.k, op, "context k");
    require_contiguous_nonnull(context.v, op, "context v");
}

struct PartialWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

template <class Allocator>
PartialWorkspace allocate_workspace(Allocator& workspace, std::int32_t tokens, std::int32_t splits,
                                    std::int32_t batch_size) {
    return {
        workspace.alloc(DType::BF16, {kHeadDim, kQHeads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {kQHeads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {kQHeads, tokens, splits * batch_size}),
    };
}

} // namespace

std::size_t sliding_window_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, std::uint32_t window,
    SlidingWindowAttentionExecutionEnvelope envelope, std::int32_t min_tokens,
    std::int32_t max_tokens, std::int32_t batch_size) {
    require_profile(geometry, window, "sliding_window_attention workspace");
    if (min_tokens < 1 || max_tokens < min_tokens || max_tokens > 16 || batch_size < 1 ||
        batch_size > 8 || envelope.min_context > envelope.max_context ||
        envelope.max_context >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(
            "sliding_window_attention workspace: invalid envelope or token interval");
    }
    const auto plan = detail::sliding_window_attention_resolve_plan(max_tokens, envelope);
    WorkspaceLayoutBuilder layout;
    (void)allocate_workspace(layout, max_tokens, plan.split_capacity, batch_size);
    return layout.peak_bytes(1);
}

void sliding_window_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& lanes, AttentionHeadGeometry geometry,
                              std::uint32_t window, float scale,
                              const CyclicKVCacheLayerView& context,
                              SlidingWindowAttentionExecutionEnvelope envelope,
                              WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "sliding_window_attention";
    require_profile(geometry, window, op);
    if (q.dtype != DType::BF16 || query_k.dtype != DType::BF16 || query_v.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("sliding_window_attention: q/k/v/out must be BF16");
    }
    if (positions.dtype != DType::I32 || valid_columns.dtype != DType::I32 ||
        lanes.dtype != DType::I32) {
        throw std::invalid_argument(
            "sliding_window_attention: positions/valid_columns/lanes must be I32");
    }
    const std::int32_t tokens = q.ne[2];
    const std::int32_t batch  = q.ne[3];
    if (tokens < 1 || tokens > 16) {
        throw std::invalid_argument("sliding_window_attention: optimized domain is T=1..16");
    }
    if (batch < 1 || batch > 8) {
        throw std::invalid_argument("sliding_window_attention: B must be 1..8");
    }
    require_shape(q, kHeadDim, kQHeads, tokens, batch, op, "q");
    require_shape(query_k, kHeadDim, kKVHeads, tokens, batch, op, "query k");
    require_shape(query_v, kHeadDim, kKVHeads, tokens, batch, op, "query v");
    require_shape(positions, tokens, batch, 1, 1, op, "positions");
    require_shape(valid_columns, batch, 1, 1, 1, op, "valid columns");
    require_shape(lanes, batch, 1, 1, 1, op, "lanes");
    require_shape(out, kHeadDim, kQHeads, tokens, batch, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(query_k, op, "query k");
    require_contiguous_nonnull(query_v, op, "query v");
    require_contiguous_nonnull(positions, op, "positions");
    require_contiguous_nonnull(valid_columns, op, "valid columns");
    require_contiguous_nonnull(lanes, op, "lanes");
    require_contiguous_nonnull(out, op, "out");
    validate_context(context, op);
    if (envelope.min_context > envelope.max_context ||
        envelope.max_context >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sliding_window_attention: invalid execution envelope");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1e-7f) {
        throw std::invalid_argument("sliding_window_attention: scale must be 1/sqrt(128)");
    }

    auto scope               = workspace.scope();
    const auto plan          = detail::sliding_window_attention_resolve_plan(tokens, envelope);
    PartialWorkspace partial = allocate_workspace(workspace, tokens, plan.split_capacity, batch);
    detail::sliding_window_attention_launch(q, query_k, query_v, positions, valid_columns, lanes,
                                            scale, context, plan, partial.acc, partial.m, partial.l,
                                            out, stream);
}

} // namespace ninfer::ops
