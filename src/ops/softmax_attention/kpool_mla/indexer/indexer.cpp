#include "ops/softmax_attention/kpool_mla/indexer/plan.h"
#include "ops/softmax_attention/kpool_mla/validation.h"

namespace ninfer::ops {
using namespace detail::kpool;

std::size_t kpool_indexer_workspace_capacity_bytes(const KpoolIndexerProfile& profile,
                                                   KpoolExecutionEnvelope envelope,
                                                   std::int32_t batch, std::int32_t lo,
                                                   std::int32_t hi, std::int32_t sms) {
    indexer_geometry(profile, envelope, batch, lo, hi, sms);
    return indexer_plan(lo, hi, batch, envelope).bytes;
}

void kpool_indexer(const Tensor& query, const Tensor& coefficients,
                   const ConstKpoolIndexCacheView& cache, const KpoolMlaBatchView& batch,
                   KpoolExecutionEnvelope envelope, MemoryArena& workspace,
                   KpoolBlockSelection& out, CudaExecutionView execution) {
    const int T = query.ne[2], B = query.ne[3];
    const auto device = execution.device();
    indexer_geometry({query.ne[1], query.ne[0]}, envelope, B, T, T,
                     execution.multiprocessor_count());
    require(std::int64_t(T) * B <= std::numeric_limits<int>::max(), "selection row extent");
    const int rows = T * B;
    std::array<operand::Range, 5> inputs;
    int n            = 0;
    const auto input = [&](const Tensor& t, DType type, std::initializer_list<int> dims,
                           int alignment, const char* name) {
        operand::tensor(t, type, dims, alignment, kOperation, name, device);
        inputs[n++] = operand::range(t, kOperation);
    };
    input(query, DType::BF16, {128, 32, T, B}, 16, "index query");
    input(coefficients, DType::FP32, {32, T, B}, 16, "head coefficients");
    input(batch.positions, DType::I32, {T, B}, 4, "positions");
    input(batch.valid_columns, DType::I32, {B}, 4, "valid columns");
    input(batch.table_rows, DType::I32, {B}, 4, "table rows");
    plane(cache.keys, 128, 4, device);
    const auto& table = cache.keys.records.page_offsets;
    require(std::uint64_t(table.ne[0]) * 64 >= envelope.max_visible_tokens, "page table capacity");
    const auto table_blocks = operand::blocks(table, kOperation);
    operand::tensor(out.pool_ids, DType::I32, {512, rows}, 4, kOperation, "pool IDs", device);
    operand::tensor(out.pool_counts, DType::I32, {rows}, 4, kOperation, "pool counts", device);
    operand::tensor(out.selection_end, DType::I32, {rows}, 4, kOperation, "selection end", device);
    std::array<operand::Range, 3> outputs{operand::range(out.pool_ids, kOperation),
                                          operand::range(out.pool_counts, kOperation),
                                          operand::range(out.selection_end, kOperation)};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < i; ++j) operand::disjoint(outputs[i], outputs[j], kOperation);
        for (const auto& src : inputs) operand::disjoint(outputs[i], src, kOperation);
        require(!operand::overlaps(table_blocks, outputs[i]), "output overlaps page table");
    }
    operand::device(workspace.device(), device, kOperation, "workspace");
    const auto plan = indexer_plan(T, T, B, envelope);
    auto scope      = workspace.scope();
    MemorySpan scratch{};
    if (plan.bytes) {
        scratch          = workspace.alloc_bytes(plan.bytes, 256);
        const auto range = operand::range(scratch.data, scratch.bytes, kOperation);
        for (const auto& src : inputs) operand::disjoint(range, src, kOperation);
        for (const auto& dst : outputs) operand::disjoint(range, dst, kOperation);
        require(!operand::overlaps(table_blocks, range), "workspace overlaps page table");
    }
    IndexerArguments a{};
    a.query        = static_cast<const __nv_bfloat16*>(query.data);
    a.keys         = static_cast<const __nv_bfloat16*>(cache.keys.data());
    a.coefficients = static_cast<const float*>(coefficients.data);
    a.positions    = static_cast<const int*>(batch.positions.data);
    a.valid        = static_cast<const int*>(batch.valid_columns.data);
    a.table_rows   = static_cast<const int*>(batch.table_rows.data);
    a.tables       = static_cast<const std::int64_t*>(table.data);
    a.table_stride = table.nb[1] / 8;
    if (plan.scores.region.bytes) a.scores = static_cast<float*>(plan.scores.bind(scratch).data);
    if (plan.candidates.region.bytes)
        a.candidates = static_cast<std::uint64_t*>(plan.candidates.bind(scratch).data);
    if (plan.merged.region.bytes)
        a.merged = static_cast<std::uint64_t*>(plan.merged.bind(scratch).data);
    a.ids          = static_cast<int*>(out.pool_ids.data);
    a.counts       = static_cast<int*>(out.pool_counts.data);
    a.ends         = static_cast<int*>(out.selection_end.data);
    a.tokens       = T;
    a.batch        = B;
    a.columns      = plan.columns;
    a.partitions   = plan.partitions;
    a.merge_groups = plan.merge_groups;
    launch_indexer(a, plan.windows, execution.multiprocessor_count(), execution.stream());
}
} // namespace ninfer::ops
