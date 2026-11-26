#include "ninfer/ops/sliding_window_attention.h"

#include "core/layout.h"
#include "ops/common/host/graph_profile.h"
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

constexpr std::int32_t kMaximumGridRows = 65535;
constexpr std::int32_t kHeadDim         = 128;
constexpr float kExpectedScale          = 0.08838834764831844055f;

template <bool Block>
constexpr const char* op_name =
    Block ? "block_sliding_window_attention" : "sliding_window_attention";

template <bool Block>
auto select_plan(AttentionHeadGeometry geometry, std::uint32_t window, ContextBucket bucket,
                 std::int32_t batch, std::int32_t tokens) {
    if constexpr (Block)
        return detail::select_block_sliding_window_attention_plan(geometry, window, bucket, batch,
                                                                  tokens);
    else
        return detail::select_sliding_window_attention_plan(geometry, window, bucket, batch,
                                                            tokens);
}

void validate_context(const ConstCyclicKVCacheLayerView& context, AttentionHeadGeometry geometry,
                      std::uint32_t window, const char* op, Device device) {
    common::operand::device(context.device(), device, op, "cache");
    if (context.storage != KvCacheStorage::BFloat16 || context.num_kv_heads != geometry.kv_heads ||
        context.head_dim != kHeadDim || context.capacity != window ||
        context.padded_capacity < context.capacity || context.k.field.layers != 1 ||
        context.v.field.layers != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid cyclic context");
    }
    if (context.padded_capacity >
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string(op) + ": padded capacity exceeds int32");
    }
    const auto padded = static_cast<std::int32_t>(context.padded_capacity);
    common::operand::state_field(context.k, DType::BF16, {kHeadDim, padded, geometry.kv_heads}, 16,
                                 op, "context K", device);
    common::operand::state_field(context.v, DType::FP16, {kHeadDim, padded, geometry.kv_heads}, 16,
                                 op, "context V", device);
}

struct PartialWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

template <bool Block, class Allocator>
PartialWorkspace allocate_workspace(Allocator& workspace, AttentionHeadGeometry geometry,
                                    std::int32_t tokens, std::int32_t splits,
                                    std::int32_t batch_size) {
    if (splits == 1) return {};
    const auto partial_dtype = Block ? DType::FP16 : DType::FP32;
    return {
        workspace.alloc(partial_dtype,
                        {kHeadDim, geometry.query_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {geometry.query_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {geometry.query_heads, tokens, splits * batch_size}),
    };
}

template <bool Block>
OpGraphProfile graph_profile(AttentionHeadGeometry geometry, std::uint32_t window,
                             ContextBucket bucket, std::int32_t batch_size, std::int32_t tokens) {
    const auto plan        = select_plan<Block>(geometry, window, bucket, batch_size, tokens);
    std::uint64_t schedule = 0;
    if constexpr (Block) schedule = std::uint64_t(plan.schedule);
    const std::uint64_t words[]{
        std::uint64_t(geometry.head_dim), std::uint64_t(geometry.query_heads),
        std::uint64_t(geometry.kv_heads), std::uint64_t(batch_size),
        std::uint64_t(plan.tokens),       std::uint64_t(plan.window),
        std::uint64_t(plan.reduce_warps), std::uint64_t(plan.split_capacity),
        std::uint64_t(plan.max_context),  schedule};
    return detail::make_op_graph_profile(words);
}

template <bool Block>
std::size_t workspace_capacity(AttentionHeadGeometry geometry, std::uint32_t window,
                               ContextBucketRange buckets, std::int32_t batch_size,
                               std::int32_t min_tokens, std::int32_t max_tokens) {
    buckets.validate(ContextBucket::M1, op_name<Block>);
    if (min_tokens < 1 || max_tokens < min_tokens || max_tokens > 16 || batch_size < 1) {
        throw std::invalid_argument(std::string(op_name<Block>) +
                                    ": invalid workspace B/T interval");
    }
    // Split capacity depends on the query width. The largest width need not have the
    // largest allocation (for example, a wider block can use fewer splits).
    std::size_t capacity = 0;
    buckets.for_each([&](ContextBucket bucket) {
        for (int tokens = min_tokens; tokens <= max_tokens; ++tokens) {
            const auto plan = select_plan<Block>(geometry, window, bucket, batch_size, tokens);
            WorkspaceLayoutBuilder layout;
            (void)allocate_workspace<Block>(layout, geometry, tokens, plan.split_capacity,
                                            std::min(batch_size, kMaximumGridRows));
            capacity = std::max(capacity, layout.peak_bytes(1));
        }
    });
    return capacity;
}

template <bool Block>
void execute(const Tensor& q, const Tensor& query_k, const Tensor& query_v, const Tensor& sequence,
             const Tensor& valid_columns, AttentionHeadGeometry geometry, std::uint32_t window,
             float scale, const ConstCyclicKVCacheLayerView& context, ContextBucket bucket,
             MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    constexpr const char* op       = op_name<Block>;
    constexpr const char* metadata = Block ? "context_lengths" : "positions";
    if (workspace.device() != execution.device())
        throw std::invalid_argument(std::string(op) + ": workspace belongs to another device");
    const auto stream = execution.stream();
    const int kQHeads = geometry.query_heads, kKVHeads = geometry.kv_heads;
    if (q.dtype != DType::BF16 || query_k.dtype != DType::BF16 || query_v.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/k/v/out must be BF16");
    }
    if (sequence.dtype != DType::I32 || valid_columns.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": " + metadata +
                                    "/valid_columns must be I32");
    }
    const std::int32_t tokens = q.ne[2];
    const std::int32_t batch  = q.ne[3];
    const auto plan           = select_plan<Block>(geometry, window, bucket, batch, tokens);
    common::operand::shape(q, {kHeadDim, kQHeads, tokens, batch}, op, "q");
    common::operand::shape(query_k, {kHeadDim, kKVHeads, tokens, batch}, op, "query k");
    common::operand::shape(query_v, {kHeadDim, kKVHeads, tokens, batch}, op, "query v");
    if constexpr (Block)
        common::operand::shape(sequence, {batch, 1, 1, 1}, op, metadata);
    else
        common::operand::shape(sequence, {tokens, batch, 1, 1}, op, metadata);
    common::operand::shape(valid_columns, {batch, 1, 1, 1}, op, "valid columns");
    common::operand::shape(out, {kHeadDim, kQHeads, tokens, batch}, op, "out");
    common::operand::contiguous_nonnull(q, op, "q", execution.device());
    common::operand::contiguous_nonnull(query_k, op, "query k", execution.device());
    common::operand::contiguous_nonnull(query_v, op, "query v", execution.device());
    common::operand::contiguous_nonnull(sequence, op, metadata, execution.device());
    common::operand::contiguous_nonnull(valid_columns, op, "valid columns", execution.device());
    common::operand::contiguous_nonnull(out, op, "out", execution.device());
    validate_context(context, geometry, window, op, execution.device());
    common::operand::shape(context.k.records.byte_offsets, {batch, 1, 1, 1}, op, "K offsets");
    common::operand::shape(context.v.records.byte_offsets, {batch, 1, 1, 1}, op, "V offsets");
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1e-7f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(128)");
    }

    constexpr auto launch = [] {
        if constexpr (Block)
            return detail::launch_block_sliding_window_attention;
        else
            return detail::launch_sliding_window_attention;
    }();
    for (std::int64_t begin = 0; begin < batch; begin += kMaximumGridRows) {
        const int rows = std::min<std::int64_t>(kMaximumGridRows, batch - begin);
        auto scope     = workspace.scope();
        auto partial =
            allocate_workspace<Block>(workspace, geometry, tokens, plan.split_capacity, rows);
        auto selected                   = context;
        selected.k.records.byte_offsets = context.k.records.byte_offsets.slice(0, begin, rows);
        selected.v.records.byte_offsets = context.v.records.byte_offsets.slice(0, begin, rows);
        auto output                     = out.slice(3, begin, rows);
        launch(q.slice(3, begin, rows), query_k.slice(3, begin, rows),
               query_v.slice(3, begin, rows), sequence.slice(Block ? 0 : 1, begin, rows),
               valid_columns.slice(0, begin, rows), scale, selected, plan, partial.acc, partial.m,
               partial.l, output, stream);
    }
}

} // namespace

OpGraphProfile sliding_window_attention_graph_profile(AttentionHeadGeometry geometry,
                                                      std::uint32_t window, ContextBucket bucket,
                                                      std::int32_t batch_size,
                                                      std::int32_t tokens) {
    return graph_profile<false>(geometry, window, bucket, batch_size, tokens);
}

std::size_t sliding_window_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, std::uint32_t window, ContextBucketRange buckets,
    std::int32_t batch_size, std::int32_t min_tokens, std::int32_t max_tokens) {
    return workspace_capacity<false>(geometry, window, buckets, batch_size, min_tokens, max_tokens);
}

void sliding_window_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                              const Tensor& positions, const Tensor& valid_columns,
                              AttentionHeadGeometry geometry, std::uint32_t window, float scale,
                              const ConstCyclicKVCacheLayerView& context, ContextBucket bucket,
                              MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    execute<false>(q, query_k, query_v, positions, valid_columns, geometry, window, scale, context,
                   bucket, workspace, out, execution);
}

OpGraphProfile block_sliding_window_attention_graph_profile(AttentionHeadGeometry geometry,
                                                            std::uint32_t window,
                                                            ContextBucket bucket,
                                                            std::int32_t batch_size,
                                                            std::int32_t tokens) {
    return graph_profile<true>(geometry, window, bucket, batch_size, tokens);
}

std::size_t block_sliding_window_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, std::uint32_t window, ContextBucketRange buckets,
    std::int32_t batch_size, std::int32_t min_tokens, std::int32_t max_tokens) {
    return workspace_capacity<true>(geometry, window, buckets, batch_size, min_tokens, max_tokens);
}

void block_sliding_window_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                                    const Tensor& context_lengths, const Tensor& valid_columns,
                                    AttentionHeadGeometry geometry, std::uint32_t window,
                                    float scale, const ConstCyclicKVCacheLayerView& context,
                                    ContextBucket bucket, MemoryArena& workspace, Tensor& out,
                                    CudaExecutionView execution) {
    execute<true>(q, query_k, query_v, context_lengths, valid_columns, geometry, window, scale,
                  context, bucket, workspace, out, execution);
}

} // namespace ninfer::ops
