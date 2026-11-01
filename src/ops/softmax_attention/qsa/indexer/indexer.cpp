#include "ninfer/ops/qsa.h"
#include "ops/softmax_attention/qsa/indexer/plan.h"
#include "ops/softmax_attention/qsa/validation.h"

namespace ninfer::ops {
namespace {
constexpr detail::qsa::ViewValidation v{"qsa_indexer"};

void profile(int lo, int hi, int batch, QsaIndexerExecutionEnvelope envelope) {
    v.require(lo > 0 && hi >= lo && batch >= 1 && batch <= 8, "T/B interval");
    v.require(envelope.max_visible_tokens <= detail::qsa::kIndexerMaxVisibleTokens,
              "visible-token capacity");
}

} // namespace

std::size_t qsa_indexer_workspace_capacity_bytes(std::int32_t min_tokens, std::int32_t max_tokens,
                                                 std::int32_t batch_size,
                                                 QsaIndexerExecutionEnvelope envelope) {
    profile(min_tokens, max_tokens, batch_size, envelope);
    // The allocation grows monotonically with T until the bounded window is full.
    WorkspaceLayoutBuilder layout;
    detail::qsa::allocate_indexer_workspace(
        layout,
        detail::qsa::select_indexer_plan(max_tokens, batch_size, envelope.max_visible_tokens),
        batch_size);
    return layout.peak_bytes(1);
}

void qsa_indexer(const Tensor& query, const ConstQsaIndexCacheView& cache,
                 const QsaBatchView& batch, QsaIndexerExecutionEnvelope envelope,
                 MemoryArena& workspace, QsaBlockSelection& out, CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("QSA workspace belongs to another device");
    }
    const detail::qsa::ViewValidation v{"qsa_indexer", execution.device()};
    const int tokens = query.ne[2], rows = query.ne[3];
    profile(tokens, tokens, rows, envelope);
    v.dense(query, DType::BF16, {128, 4, tokens, rows}, 16, "query");
    v.dense(batch.positions, DType::I32, {tokens, rows, 1, 1}, 4, "positions");
    v.metadata(batch.valid_columns, rows, "valid columns");
    v.metadata(batch.table_rows, rows, "table rows");
    common::operand::paged_plane(cache.keys, DType::BF16, {128, 16, 1}, 2, "qsa_indexer",
                                 "index pages", execution.device());
    v.require(cache.keys.records.page_tokens == 64 && cache.keys.plane.geometry.tokens_per_row == 4,
              "index record geometry");
    const auto& table = cache.keys.records.page_offsets;
    v.require(std::uint64_t(table.ne[0]) * 64 >= envelope.max_visible_tokens, "table capacity");
    v.dense(out.block_ids, DType::I32, {512, tokens, rows, 1}, 4, "block ids");
    v.dense(out.block_counts, DType::I32, {tokens, rows, 1, 1}, 4, "block counts");
    const Tensor* inputs[]{&query, &table, &batch.positions, &batch.valid_columns,
                           &batch.table_rows};
    v.require(!v.overlap(out.block_ids, out.block_counts), "output overlap");
    for (const auto* dst : {&out.block_ids, &out.block_counts})
        for (const auto* src : inputs) v.require(!v.overlap(*dst, *src), "output/input overlap");
    const auto plan = detail::qsa::select_indexer_plan(tokens, rows, envelope.max_visible_tokens);
    auto scope      = workspace.scope();
    const auto scratch = detail::qsa::allocate_indexer_workspace(workspace, plan, rows);
    for (const auto* work : {&scratch.scores, &scratch.candidates}) {
        if (!work->data) continue;
        for (const auto* src : inputs)
            v.require(!v.overlap(*work, *src), "workspace/input overlap");
        for (const auto* dst : {&out.block_ids, &out.block_counts})
            v.require(!v.overlap(*work, *dst), "workspace/output overlap");
    }
    detail::qsa::launch_indexer(query, cache, batch, plan, scratch, out, execution);
}
} // namespace ninfer::ops
