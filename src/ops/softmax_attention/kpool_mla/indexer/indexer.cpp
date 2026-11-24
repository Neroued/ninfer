#include "ops/softmax_attention/kpool_mla/indexer/plan.h"
#include "ops/softmax_attention/kpool_mla/validation.h"
#include "ops/common/host/graph_profile.h"

namespace ninfer::ops {
using namespace detail::kpool;

OpGraphProfile kpool_indexer_graph_profile(const KpoolIndexerProfile& profile,
                                           KvCacheVectorFormat index_format, ContextBucket bucket,
                                           int batch, int tokens, int sms) {
    ContextBucketRange{bucket}.validate(ContextBucket::M1, "kpool_indexer");
    indexer_geometry(profile, index_format, batch, tokens, tokens, sms);
    const bool fp8           = index_format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0;
    const auto p             = indexer_plan(tokens, batch, bucket, fp8);
    const int group          = indexer_group_size(tokens, batch);
    const int prepared_batch = fp8 ? group : 0;
    // The geometry and window budgets determine every score/selection binding.
    // Only the full and tail request groups and prepared chunks can differ.
    std::array<std::uint64_t, 23> words{std::uint64_t(tokens),
                                        std::uint64_t(batch),
                                        std::uint64_t(sms),
                                        std::uint64_t(index_format),
                                        std::uint64_t(p.columns),
                                        std::uint64_t(p.partitions),
                                        std::uint64_t(p.merge_groups),
                                        p.bytes,
                                        std::uint64_t(p.query_width),
                                        std::uint64_t(group),
                                        p.query_fp8.region.offset,
                                        p.weighted_coefficients.region.offset,
                                        p.candidates.region.offset,
                                        p.scores.region.offset,
                                        p.merged.region.offset};
    std::size_t word = 15;
    for (const int outer : {group, batch % group}) {
        if (!outer) break;
        const auto windows = indexer_windows(tokens, outer, p.columns, prepared_batch);
        const int tail     = tokens % p.query_width;
        words[word++]      = windows.group_size;
        words[word++]      = windows.row_budget;
        words[word++]      = indexer_graph_slots(tokens, p.query_width, outer, prepared_batch);
        words[word++]      = tail ? indexer_graph_slots(tokens, tail, outer, prepared_batch) : 0;
    }
    return detail::make_op_graph_profile(words);
}

std::size_t kpool_indexer_workspace_capacity_bytes(const KpoolIndexerProfile& profile,
                                                   KvCacheVectorFormat index_format,
                                                   ContextBucketRange buckets, std::int32_t batch,
                                                   std::int32_t lo, std::int32_t hi,
                                                   std::int32_t sms) {
    buckets.validate(ContextBucket::M1, "kpool_indexer");
    indexer_geometry(profile, index_format, batch, lo, hi, sms);
    const bool fp8    = index_format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0;
    std::size_t bytes = 0;
    buckets.for_each([&](ContextBucket bucket) {
        bytes = std::max(bytes, indexer_capacity(lo, hi, batch, bucket, fp8));
    });
    return bytes;
}

void kpool_indexer(const Tensor& query, const Tensor& coefficients,
                   const ConstKpoolIndexCacheView& cache, const KpoolMlaBatchView& batch,
                   ContextBucket bucket, MemoryArena& workspace, KpoolBlockSelection& out,
                   CudaExecutionView execution) {
    const int T = query.ne[2], B = query.ne[3];
    const auto device = execution.device();
    ContextBucketRange{bucket}.validate(ContextBucket::M1, "kpool_indexer");
    indexer_geometry({query.ne[1], query.ne[0]}, cache.format, B, T, T,
                     execution.multiprocessor_count());
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
    index_cache(cache, 128, device);
    const auto& table       = cache.keys.records.page_offsets;
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
    const bool fp8  = cache.format == KvCacheVectorFormat::Fp8E4m3fnGroup128Ue8m0;
    const auto plan = indexer_plan(T, B, bucket, fp8);
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
    a.keys         = fp8 ? nullptr : static_cast<const __nv_bfloat16*>(cache.keys.data());
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
    if (fp8) {
        Fp8IndexerArguments f{};
        f.index  = a;
        f.keys   = static_cast<const std::uint8_t*>(cache.keys.data());
        f.scales = static_cast<const std::uint8_t*>(cache.scales.data());
        if (plan.bytes) {
            f.query        = static_cast<std::uint8_t*>(plan.query_fp8.bind(scratch).data);
            f.coefficients = static_cast<float*>(plan.weighted_coefficients.bind(scratch).data);
        }
        launch_fp8_indexer(f, execution.multiprocessor_count(), execution.stream());
    } else {
        launch_indexer(a, execution.multiprocessor_count(), execution.stream());
    }
}

std::size_t kpool_indexer_ragged_workspace_capacity_bytes(
    const KpoolIndexerProfile& profile, KvCacheVectorFormat format, ContextBucketRange buckets,
    int min_tokens, int max_tokens, int min_segments, int max_segments, int sm_count) {
    const auto widths = common::operand::ragged_envelope(min_tokens, max_tokens, min_segments,
                                                         max_segments, "kpool_indexer");
    return kpool_indexer_workspace_capacity_bytes(profile, format, buckets, 1, widths.first,
                                                  widths.last, sm_count);
}

void kpool_indexer(const Tensor& query, const Tensor& coeff, const ConstKpoolIndexCacheView& cache,
                   const KpoolMlaRaggedBatchView& batch, ContextBucket bucket,
                   MemoryArena& workspace, KpoolBlockSelection& out, CudaExecutionView execution) {
    namespace operand        = common::operand;
    constexpr const char* op = "kpool_indexer";
    const auto& seq          = batch.ragged;
    const auto device        = execution.device();
    operand::ragged_batch(seq, batch.positions, batch.table_rows, op, device);
    const int n = seq.total_tokens();
    operand::tensor(query, DType::BF16, {128, 32, n}, 16, op, "query", device);
    operand::tensor(coeff, DType::FP32, {32, n}, 4, op, "head coefficients", device);
    operand::tensor(out.pool_ids, DType::I32, {512, n}, 4, op, "pool IDs", device);
    operand::tensor(out.pool_counts, DType::I32, {n}, 4, op, "pool counts", device);
    operand::tensor(out.selection_end, DType::I32, {n}, 4, op, "selection end", device);
    for (int s = 0; s < seq.num_seqs(); ++s) {
        const int begin = seq.segment_begin(s), len = seq.segment_length(s);
        const KpoolMlaBatchView segment{batch.positions.slice(0, begin, len),
                                        seq.seq_lengths.slice(0, s, 1),
                                        batch.table_rows.slice(0, s, 1)};
        KpoolBlockSelection result{out.pool_ids.slice(1, begin, len),
                                   out.pool_counts.slice(0, begin, len),
                                   out.selection_end.slice(0, begin, len)};
        kpool_indexer(query.slice(2, begin, len), coeff.slice(1, begin, len), cache, segment,
                      bucket, workspace, result, execution);
    }
}

} // namespace ninfer::ops
