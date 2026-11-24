#include "ninfer/ops/qsa.h"
#include "ops/common/host/graph_profile.h"
#include "ops/softmax_attention/qsa/indexer/plan.h"
#include "ops/softmax_attention/qsa/validation.h"

namespace ninfer::ops {
namespace {
constexpr detail::qsa::ViewValidation v{"qsa_indexer"};

} // namespace

std::size_t qsa_indexer_workspace_capacity_bytes(ContextBucketRange buckets,
                                                 std::int32_t batch_size, std::int32_t min_tokens,
                                                 std::int32_t max_tokens) {
    buckets.validate(ContextBucket::K256, "qsa_indexer");
    v.require(batch_size >= 1 && min_tokens >= 1 && max_tokens >= min_tokens, "T/B interval");
    WorkspaceLayoutBuilder layout;
    // Scratch rows grow with T until the first full window, then repeat across windows.
    buckets.for_each([&](ContextBucket bucket) {
        auto scope = layout.scope();
        detail::qsa::allocate_indexer_workspace(
            layout, detail::qsa::select_indexer_plan(max_tokens, batch_size, bucket));
    });
    return layout.peak_bytes(1);
}

OpGraphProfile qsa_indexer_graph_profile(ContextBucket bucket, std::int32_t batch_size,
                                         std::int32_t tokens, std::int32_t sm_count) {
    ContextBucketRange{bucket}.validate(ContextBucket::K256, "qsa_indexer");
    v.require(batch_size >= 1 && tokens >= 1 && sm_count >= 1, "T/B/SM count");
    const auto plan = detail::qsa::select_indexer_plan(tokens, batch_size, bucket);
    WorkspaceLayoutBuilder layout;
    const auto scratch         = detail::qsa::allocate_indexer_workspace(layout, plan);
    const auto score_bytes     = plan.partitions > 0 ? scratch.scores.bytes() : 0;
    const auto candidate_bytes = plan.partitions > 1 ? scratch.candidates.bytes() : 0;
    const int full_windows     = tokens / plan.window_tokens;
    const int tail_width       = tokens % plan.window_tokens;
    // Scores start at zero and their size is a multiple of 256, so candidates need no padding.
    std::array<std::uint64_t, 24> words{std::uint64_t(tokens),
                                        std::uint64_t(batch_size),
                                        std::uint64_t(plan.window_tokens),
                                        std::uint64_t(plan.columns),
                                        std::uint64_t(plan.partitions),
                                        std::uint64_t(plan.window_queries),
                                        std::uint64_t(full_windows),
                                        std::uint64_t(tail_width),
                                        0,
                                        score_bytes,
                                        candidate_bytes ? score_bytes : 0,
                                        candidate_bytes};
    // All full windows use one launch plan. A final partial window can choose another route.
    std::size_t next = 12;
    for (const int width : {full_windows ? plan.window_tokens : 0, tail_width}) {
        for (const bool tma_cache : {false, true}) {
            if (width > 0) {
                const auto window = detail::qsa::select_indexer_window(plan, width, batch_size,
                                                                       sm_count, tma_cache);
                words[next]       = window.score_workers;
                words[next + 1]   = window.select_workers;
                words[next + 2]   = std::uint64_t(window.query_tile) |
                                  (std::uint64_t(window.stages) << 8) |
                                  (std::uint64_t(window.score) << 16) |
                                  (std::uint64_t(window.reverse_selection) << 24) |
                                  (std::uint64_t(window.retain_keys) << 25);
            }
            next += 3;
        }
    }
    return detail::make_op_graph_profile(words);
}

namespace {
void indexer(const Tensor& query, const ConstQsaIndexCacheView& cache, const QsaBatchView& batch,
             ContextBucket bucket, MemoryArena& workspace, QsaBlockSelection& out,
             CudaExecutionView execution, const RaggedSequenceView* ragged = nullptr) {
    ContextBucketRange{bucket}.validate(ContextBucket::K256, "qsa_indexer");
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("QSA workspace belongs to another device");
    }
    const detail::qsa::ViewValidation v{"qsa_indexer", execution.device()};
    const int tokens = query.ne[2], rows = query.ne[3];
    v.require(tokens >= 1 && rows >= 1, "T/B");
    v.dense(query, DType::BF16, {128, 4, tokens, rows}, 16, "query");
    v.dense(batch.positions, DType::I32, {tokens, rows, 1, 1}, 4, "positions");
    v.metadata(batch.valid_columns, ragged ? ragged->num_seqs() : rows, "valid columns");
    v.metadata(batch.table_rows, ragged ? ragged->num_seqs() : rows, "table rows");
    common::operand::paged_plane(cache.keys, DType::BF16, {128, 16, 1}, 2, "qsa_indexer",
                                 "index pages", execution.device());
    v.require(cache.keys.records.page_tokens == 64 && cache.keys.plane.geometry.tokens_per_row == 4,
              "index record geometry");
    const auto& table = cache.keys.records.page_offsets;
    v.dense(out.block_ids, DType::I32, {512, tokens, rows, 1}, 4, "block ids");
    v.dense(out.block_counts, DType::I32, {tokens, rows, 1, 1}, 4, "block counts");
    const Tensor* inputs[]{&query,
                           &table,
                           &batch.positions,
                           &batch.valid_columns,
                           &batch.table_rows,
                           ragged ? &ragged->cu_seqlens : nullptr,
                           ragged ? &ragged->token_to_seq : nullptr};
    v.require(!v.overlap(out.block_ids, out.block_counts), "output overlap");
    for (const auto* dst : {&out.block_ids, &out.block_counts})
        for (const auto* src : inputs)
            if (src) v.require(!v.overlap(*dst, *src), "output/input overlap");
    const auto plan    = detail::qsa::select_indexer_plan(tokens, rows, bucket);
    auto scope         = workspace.scope();
    const auto scratch = detail::qsa::allocate_indexer_workspace(workspace, plan);
    for (const auto* work : {&scratch.scores, &scratch.candidates}) {
        if (!work->data) continue;
        for (const auto* src : inputs)
            if (src) v.require(!v.overlap(*work, *src), "workspace/input overlap");
        for (const auto* dst : {&out.block_ids, &out.block_counts})
            v.require(!v.overlap(*work, *dst), "workspace/output overlap");
    }
    detail::qsa::launch_indexer(query, cache, batch, plan, scratch, out, execution,
                                ragged && ragged->num_seqs() > 1 ? ragged : nullptr);
}
} // namespace

void qsa_indexer(const Tensor& query, const ConstQsaIndexCacheView& cache,
                 const QsaBatchView& batch, ContextBucket bucket, MemoryArena& workspace,
                 QsaBlockSelection& out, CudaExecutionView execution) {
    indexer(query, cache, batch, bucket, workspace, out, execution);
}

std::size_t qsa_indexer_ragged_workspace_capacity_bytes(ContextBucketRange buckets, int min_tokens,
                                                        int max_tokens, int min_segments,
                                                        int max_segments) {
    (void)common::operand::ragged_envelope(min_tokens, max_tokens, min_segments, max_segments,
                                           "qsa_indexer");
    return qsa_indexer_workspace_capacity_bytes(buckets, 1, std::max(min_tokens, min_segments),
                                                max_tokens);
}

void qsa_indexer(const Tensor& query, const ConstQsaIndexCacheView& cache,
                 const QsaRaggedBatchView& batch, ContextBucket bucket, MemoryArena& workspace,
                 QsaBlockSelection& out, CudaExecutionView execution) {
    namespace operand        = common::operand;
    constexpr const char* op = "qsa_indexer";
    const auto& seq          = batch.ragged;
    const auto device        = execution.device();
    operand::ragged_batch(seq, batch.positions, batch.table_rows, op, device);
    const int n = seq.total_tokens();
    operand::tensor(query, DType::BF16, {128, 4, n}, 16, op, "query", device);
    operand::tensor(out.block_ids, DType::I32, {512, n}, 4, op, "block ids", device);
    operand::tensor(out.block_counts, DType::I32, {n}, 4, op, "block counts", device);
    const QsaBatchView flat{batch.positions, seq.seq_lengths, batch.table_rows};
    indexer(query, cache, flat, bucket, workspace, out, execution, &seq);
}

} // namespace ninfer::ops
