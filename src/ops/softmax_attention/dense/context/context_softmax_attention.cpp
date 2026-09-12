#include "ninfer/ops/softmax_attention.h"

#include "core/layout.h"
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

std::uint32_t validate_context(const PagedKVBatchLayerView& context, const char* op) {
    if (context.storage != KvCacheStorage::BFloat16 || context.num_kv_heads != kKVHeads ||
        context.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid context geometry or dtype");
    }
    const std::int32_t physical_pages = context.k_pages.ne[2];
    if (physical_pages <= 0 || context.v_pages.ne[2] != physical_pages ||
        context.block_tables.ne[0] <= 0 || context.block_tables.ne[1] <= 0) {
        throw std::invalid_argument(std::string(op) + ": invalid context capacity");
    }
    if (context.k_pages.dtype != DType::BF16 || context.v_pages.dtype != DType::FP16) {
        throw std::invalid_argument(std::string(op) + ": context K/V must be BF16");
    }
    require_shape(context.k_pages, kHeadDim, kPagedKVPageSize, physical_pages, kKVHeads, op,
                  "context k pages");
    require_shape(context.v_pages, kHeadDim, kPagedKVPageSize, physical_pages, kKVHeads, op,
                  "context v pages");
    require_contiguous_nonnull(context.k_pages, op, "context k pages");
    require_contiguous_nonnull(context.v_pages, op, "context v pages");
    require_shape(context.block_tables, context.block_tables.ne[0], context.block_tables.ne[1], 1,
                  1, op, "context block tables");
    if (context.block_tables.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": context block tables must be I32");
    }
    require_contiguous_nonnull(context.block_tables, op, "context block tables");
    if (context.k_scale_pages.data != nullptr || context.v_scale_pages.data != nullptr) {
        throw std::invalid_argument(std::string(op) + ": BF16 context must not have scales");
    }
    const std::uint64_t logical_capacity =
        static_cast<std::uint64_t>(context.block_tables.ne[0]) * kPagedKVPageSize;
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
PartialWorkspace allocate_workspace(Allocator& workspace, std::int32_t tokens, std::int32_t splits,
                                    std::int32_t batch_size) {
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
        const auto plan = detail::context_attention_resolve_plan(tokens, envelope);
        WorkspaceLayoutBuilder layout;
        (void)allocate_workspace(layout, tokens, plan.split_capacity, batch_size);
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
                               float scale, const PagedKVBatchLayerView& context,
                               ContextAttentionExecutionEnvelope envelope,
                               WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
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
    require_shape(q, kHeadDim, kQHeads, tokens, batch, op, "q");
    require_shape(query_k, kHeadDim, kKVHeads, tokens, batch, op, "query k");
    require_shape(query_v, kHeadDim, kKVHeads, tokens, batch, op, "query v");
    require_shape(context_lengths, batch, 1, 1, 1, op, "context lengths");
    require_shape(valid_columns, batch, 1, 1, 1, op, "valid columns");
    require_shape(table_rows, batch, 1, 1, 1, op, "table rows");
    require_shape(out, kHeadDim, kQHeads, tokens, batch, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(query_k, op, "query k");
    require_contiguous_nonnull(query_v, op, "query v");
    require_contiguous_nonnull(context_lengths, op, "context lengths");
    require_contiguous_nonnull(valid_columns, op, "valid columns");
    require_contiguous_nonnull(table_rows, op, "table rows");
    require_contiguous_nonnull(out, op, "out");
    const std::uint32_t logical_capacity = validate_context(context, op);
    if (envelope.min_context > envelope.max_context || envelope.max_context > logical_capacity) {
        throw std::invalid_argument("context_softmax_attention: invalid execution envelope");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1e-7f) {
        throw std::invalid_argument("context_softmax_attention: scale must be 1/sqrt(128)");
    }

    auto scope               = workspace.scope();
    const auto plan          = detail::context_attention_resolve_plan(tokens, envelope);
    PartialWorkspace partial = allocate_workspace(workspace, tokens, plan.split_capacity, batch);
    detail::context_attention_launch(q, query_k, query_v, context_lengths, valid_columns,
                                     table_rows, scale, context, plan, partial.acc, partial.m,
                                     partial.l, out, stream);
}

} // namespace ninfer::ops
