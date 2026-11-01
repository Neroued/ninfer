#include "ninfer/ops/sliding_window_attention.h"

#include "core/layout.h"
#include "ops/common/host/operands.h"
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
constexpr float kExpectedScale  = 0.08838834764831844055f;

bool registered_window(std::uint32_t window) { return window == 2048 || window == 4096; }

void require_profile(AttentionHeadGeometry geometry, std::uint32_t window, const char* op) {
    if (!valid_attention_head_geometry(geometry) || geometry.head_dim != kHeadDim ||
        geometry.query_heads != kQHeads || geometry.kv_heads != kKVHeads) {
        throw std::invalid_argument(std::string(op) + ": unsupported head geometry");
    }
    if (!registered_window(window)) {
        throw std::invalid_argument(std::string(op) + ": supported windows are 2048 and 4096");
    }
}

void validate_context(const ConstCyclicKVCacheLayerView& context, std::uint32_t window,
                      const char* op, Device device) {
    common::operand::device(context.device(), device, op, "cache");
    if (context.num_kv_heads != kKVHeads || context.head_dim != kHeadDim ||
        context.capacity != window || context.padded_capacity < context.capacity ||
        context.k.field.layers != 1 || context.v.field.layers != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid cyclic context");
    }
    if (context.padded_capacity >
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string(op) + ": padded capacity exceeds int32");
    }
    const auto padded = static_cast<std::int32_t>(context.padded_capacity);
    common::operand::state_field(context.k, DType::BF16, {kHeadDim, padded, kKVHeads}, 16, op,
                                 "context K", device);
    common::operand::state_field(context.v, DType::FP16, {kHeadDim, padded, kKVHeads}, 16, op,
                                 "context V", device);
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
        workspace.alloc(DType::FP32, {kHeadDim, kQHeads, tokens, splits * batch_size}),
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
    // Split capacity depends on the query width. The largest width need not have the
    // largest allocation (for example, a wider block can use fewer splits).
    std::size_t capacity = 0;
    for (int tokens = min_tokens; tokens <= max_tokens; ++tokens) {
        const auto plan =
            detail::select_sliding_window_attention_plan(window, tokens, batch_size, envelope);
        if (plan.route == detail::SlidingWindowAttentionRoute::Direct) continue;
        WorkspaceLayoutBuilder layout;
        (void)allocate_workspace(layout, tokens, plan.split_capacity, batch_size);
        capacity = std::max(capacity, layout.peak_bytes(1));
    }
    return capacity;
}

void sliding_window_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                              const Tensor& positions, const Tensor& valid_columns,
                              AttentionHeadGeometry geometry, std::uint32_t window, float scale,
                              const ConstCyclicKVCacheLayerView& context,
                              SlidingWindowAttentionExecutionEnvelope envelope,
                              MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument(
            "sliding_window_attention: workspace belongs to another device");
    }
    const auto stream        = execution.stream();
    constexpr const char* op = "sliding_window_attention";
    require_profile(geometry, window, op);
    if (q.dtype != DType::BF16 || query_k.dtype != DType::BF16 || query_v.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("sliding_window_attention: q/k/v/out must be BF16");
    }
    if (positions.dtype != DType::I32 || valid_columns.dtype != DType::I32) {
        throw std::invalid_argument(
            "sliding_window_attention: positions/valid_columns must be I32");
    }
    const std::int32_t tokens = q.ne[2];
    const std::int32_t batch  = q.ne[3];
    if (tokens < 1 || tokens > 16) {
        throw std::invalid_argument("sliding_window_attention: optimized domain is T=1..16");
    }
    if (batch < 1 || batch > 8) {
        throw std::invalid_argument("sliding_window_attention: B must be 1..8");
    }
    common::operand::shape(q, {kHeadDim, kQHeads, tokens, batch}, op, "q");
    common::operand::shape(query_k, {kHeadDim, kKVHeads, tokens, batch}, op, "query k");
    common::operand::shape(query_v, {kHeadDim, kKVHeads, tokens, batch}, op, "query v");
    common::operand::shape(positions, {tokens, batch, 1, 1}, op, "positions");
    common::operand::shape(valid_columns, {batch, 1, 1, 1}, op, "valid columns");
    common::operand::shape(out, {kHeadDim, kQHeads, tokens, batch}, op, "out");
    common::operand::contiguous_nonnull(q, op, "q", execution.device());
    common::operand::contiguous_nonnull(query_k, op, "query k", execution.device());
    common::operand::contiguous_nonnull(query_v, op, "query v", execution.device());
    common::operand::contiguous_nonnull(positions, op, "positions", execution.device());
    common::operand::contiguous_nonnull(valid_columns, op, "valid columns", execution.device());
    common::operand::contiguous_nonnull(out, op, "out", execution.device());
    validate_context(context, window, op, execution.device());
    common::operand::shape(context.k.records.byte_offsets, {batch, 1, 1, 1}, op, "K offsets");
    common::operand::shape(context.v.records.byte_offsets, {batch, 1, 1, 1}, op, "V offsets");
    if (envelope.min_context > envelope.max_context ||
        envelope.max_context >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("sliding_window_attention: invalid execution envelope");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1e-7f) {
        throw std::invalid_argument("sliding_window_attention: scale must be 1/sqrt(128)");
    }

    auto scope      = workspace.scope();
    const auto plan = detail::select_sliding_window_attention_plan(window, tokens, batch, envelope);
    PartialWorkspace partial{};
    if (plan.route == detail::SlidingWindowAttentionRoute::SplitKv) {
        partial = allocate_workspace(workspace, tokens, plan.split_capacity, batch);
    }
    detail::launch_sliding_window_attention(q, query_k, query_v, positions, valid_columns, scale,
                                            context, plan, partial.acc, partial.m, partial.l, out,
                                            stream);
}

} // namespace ninfer::ops
