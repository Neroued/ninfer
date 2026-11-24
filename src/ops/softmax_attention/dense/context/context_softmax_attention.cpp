#include "ninfer/ops/softmax_attention.h"

#include "core/layout.h"
#include "ops/common/host/operands.h"
#include "ops/softmax_attention/dense/context/launch.h"

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

void require_profile(AttentionHeadGeometry geometry, const char* op) {
    if (!valid_attention_head_geometry(geometry) || geometry.head_dim != kHeadDim ||
        geometry.query_heads != kQHeads || geometry.kv_heads != kKVHeads) {
        throw std::invalid_argument(std::string(op) + ": unsupported head geometry");
    }
}

std::uint32_t validate_context(const ConstPagedKVBatchLayerView& context, const char* op,
                               Device device) {
    common::operand::device(context.device(), device, op, "cache");
    if (context.storage != KvCacheStorage::BFloat16 || context.num_kv_heads != kKVHeads ||
        context.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid context geometry or dtype");
    }
    const auto& offsets = context.k_pages.records.page_offsets;
    if (offsets.ne[0] <= 0 || offsets.ne[1] <= 0 ||
        context.k_pages.records.page_tokens != kPagedKVPageSize) {
        throw std::invalid_argument(std::string(op) + ": invalid context capacity");
    }
    common::operand::paged_plane(context.k_pages, DType::BF16,
                                 {kHeadDim, kPagedKVPageSize, kKVHeads}, 16, op, "context K",
                                 device);
    common::operand::paged_plane(context.v_pages, DType::FP16,
                                 {kHeadDim, kPagedKVPageSize, kKVHeads}, 16, op, "context V",
                                 device);
    common::operand::same_paged_records(context.k_pages, context.v_pages, op, "context K/V");
    if (context.k_scale_pages.records.storage.data != nullptr ||
        context.v_scale_pages.records.storage.data != nullptr) {
        throw std::invalid_argument(std::string(op) + ": BF16 context must not have scales");
    }
    const std::uint64_t logical_capacity =
        static_cast<std::uint64_t>(context.k_pages.records.page_offsets.ne[0]) * kPagedKVPageSize;
    if (logical_capacity > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string(op) + ": logical context capacity exceeds int32");
    }
    return static_cast<std::uint32_t>(logical_capacity);
}

struct PartialWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

template <class Allocator>
PartialWorkspace allocate_workspace(Allocator& workspace, std::int32_t tokens,
                                    const detail::ContextAttentionPlan& plan,
                                    std::int32_t batch_size) {
    if (plan.route == detail::ContextAttentionRoute::Direct) return {};
    const int splits = plan.split_capacity;
    return {
        workspace.alloc(DType::BF16, {kHeadDim, kQHeads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {kQHeads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {kQHeads, tokens, splits * batch_size}),
    };
}

} // namespace

std::size_t context_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, ContextAttentionExecutionEnvelope envelope,
    std::int32_t min_tokens, std::int32_t max_tokens, std::int32_t batch_size) {
    require_profile(geometry, "context_softmax_attention workspace");
    if (min_tokens < 1 || max_tokens < min_tokens || max_tokens > 16 || batch_size < 1 ||
        batch_size > 8 || envelope.min_context > envelope.max_context ||
        envelope.max_context >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(
            "context_softmax_attention workspace: invalid envelope or token interval");
    }
    const auto endpoint_capacity = [&](std::int32_t tokens) {
        const auto plan = detail::select_context_attention_plan(tokens, envelope);
        WorkspaceLayoutBuilder layout;
        (void)allocate_workspace(layout, tokens, plan, batch_size);
        return layout.peak_bytes(1);
    };

    std::size_t maximum = 0;
    if (min_tokens <= 8) { maximum = endpoint_capacity(std::min(max_tokens, 8)); }
    if (max_tokens >= 9) { maximum = std::max(maximum, endpoint_capacity(max_tokens)); }
    return maximum;
}

void context_softmax_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                               const Tensor& context_lengths, const Tensor& valid_columns,
                               const Tensor& table_rows, AttentionHeadGeometry geometry,
                               float scale, const ConstPagedKVBatchLayerView& context,
                               ContextAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                               Tensor& out, CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument(
            "context_softmax_attention: workspace belongs to another device");
    }
    const auto stream        = execution.stream();
    constexpr const char* op = "context_softmax_attention";
    require_profile(geometry, op);
    if (q.dtype != DType::BF16 || query_k.dtype != DType::BF16 || query_v.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("context_softmax_attention: q/k/v/out must be BF16");
    }
    if (context_lengths.dtype != DType::I32 || valid_columns.dtype != DType::I32 ||
        table_rows.dtype != DType::I32) {
        throw std::invalid_argument(
            "context_softmax_attention: lengths/valid_columns/table_rows must be I32");
    }
    const std::int32_t tokens = q.ne[2];
    const std::int32_t batch  = q.ne[3];
    if (tokens < 1 || tokens > 16) {
        throw std::invalid_argument("context_softmax_attention: optimized domain is T=1..16");
    }
    if (batch < 1 || batch > 8) {
        throw std::invalid_argument("context_softmax_attention: B must be 1..8");
    }
    common::operand::shape(q, {kHeadDim, kQHeads, tokens, batch}, op, "q");
    common::operand::shape(query_k, {kHeadDim, kKVHeads, tokens, batch}, op, "query k");
    common::operand::shape(query_v, {kHeadDim, kKVHeads, tokens, batch}, op, "query v");
    common::operand::shape(context_lengths, {batch, 1, 1, 1}, op, "context lengths");
    common::operand::shape(valid_columns, {batch, 1, 1, 1}, op, "valid columns");
    common::operand::shape(table_rows, {batch, 1, 1, 1}, op, "table rows");
    common::operand::shape(out, {kHeadDim, kQHeads, tokens, batch}, op, "out");
    common::operand::contiguous_nonnull(q, op, "q", execution.device());
    common::operand::contiguous_nonnull(query_k, op, "query k", execution.device());
    common::operand::contiguous_nonnull(query_v, op, "query v", execution.device());
    common::operand::contiguous_nonnull(context_lengths, op, "context lengths", execution.device());
    common::operand::contiguous_nonnull(valid_columns, op, "valid columns", execution.device());
    common::operand::contiguous_nonnull(table_rows, op, "table rows", execution.device());
    common::operand::contiguous_nonnull(out, op, "out", execution.device());
    const std::uint32_t logical_capacity = validate_context(context, op, execution.device());
    if (envelope.min_context > envelope.max_context || envelope.max_context > logical_capacity) {
        throw std::invalid_argument("context_softmax_attention: invalid execution envelope");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1e-7f) {
        throw std::invalid_argument("context_softmax_attention: scale must be 1/sqrt(128)");
    }

    auto scope               = workspace.scope();
    const auto plan          = detail::select_context_attention_plan(tokens, envelope);
    PartialWorkspace partial = allocate_workspace(workspace, tokens, plan, batch);
    detail::launch_context_attention(q, query_k, query_v, context_lengths, valid_columns,
                                     table_rows, scale, context, plan, partial.acc, partial.m,
                                     partial.l, out, stream);
}

} // namespace ninfer::ops
