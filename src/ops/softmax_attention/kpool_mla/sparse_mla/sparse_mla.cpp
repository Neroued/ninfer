#include "ops/softmax_attention/kpool_mla/sparse_mla/plan.h"
#include "ops/softmax_attention/kpool_mla/validation.h"
#include "ops/common/host/graph_profile.h"

#include <cmath>

namespace ninfer::ops {
using namespace detail::kpool;

KpoolSparseMlaProfile kpool_sparse_mla_profile(const KpoolMlaFactors& factors, int heads) {
    require(heads > 0 && factors.key_up.n % heads == 0 && factors.value_up.n % heads == 0 &&
                factors.key_up.k == factors.value_up.k,
            "MLA factor dimensions");
    return {heads,
            factors.key_up.n / heads,
            factors.value_up.n / heads,
            factors.key_up.k,
            kKpoolPoolSize,
            kKpoolSelectedPools,
            factors.key_up.qtype,
            factors.value_up.qtype};
}

OpGraphProfile kpool_sparse_mla_graph_profile(const KpoolSparseMlaProfile& profile,
                                              KpoolMlaStorage storage, ContextBucket bucket,
                                              int batch, int tokens, int sms) {
    ContextBucketRange{bucket}.validate(ContextBucket::M1, "kpool_sparse_mla");
    sparse_mla_geometry(profile, storage, batch, tokens, tokens, sms);
    std::array<std::uint64_t, 22> words{std::uint64_t(tokens),
                                        std::uint64_t(batch),
                                        std::uint64_t(sms),
                                        std::uint64_t(storage),
                                        std::uint64_t(profile.key_format),
                                        std::uint64_t(profile.value_format)};
    std::size_t at    = 6;
    const auto append = [&](int columns) {
        if (!columns) return;
        const auto p = sparse_mla_plan(profile, storage, columns, batch, bucket, sms);
        // Storage, window width and splits determine every region of the layout.
        for (const auto word :
             {std::uint64_t(p.factor), std::uint64_t(p.attention), std::uint64_t(p.splits),
              std::uint64_t(p.preferred_splits), std::uint64_t(p.tile),
              std::uint64_t(p.merge_value), std::uint64_t(p.fp8_query), std::uint64_t(p.bytes)})
            words[at++] = word;
    };
    const auto columns = std::int64_t(tokens) * batch;
    if (columns >= kMlaWindow) append(kMlaWindow);
    append(columns % kMlaWindow);
    return detail::make_op_graph_profile(words);
}

std::size_t kpool_sparse_mla_workspace_capacity_bytes(const KpoolSparseMlaProfile& profile,
                                                      KpoolMlaStorage storage,
                                                      ContextBucketRange buckets,
                                                      std::int32_t batch, std::int32_t lo,
                                                      std::int32_t hi, std::int32_t sms) {
    buckets.validate(ContextBucket::M1, "kpool_sparse_mla");
    sparse_mla_geometry(profile, storage, batch, lo, hi, sms);
    std::size_t peak = 0;
    buckets.for_each([&](ContextBucket bucket) {
        peak = std::max(peak, sparse_mla_capacity(profile, storage, lo, hi, batch, bucket, sms));
    });
    return peak;
}

namespace {
void execute(const Tensor& query, const KpoolMlaFactors& factors,
             const ConstKpoolMlaLatentCacheView& cache, const KpoolMlaBatchView& batch,
             const RaggedSequenceView* seq, const KpoolBlockSelection& selected,
             const Tensor& selection_rows, float attention_scale, ContextBucket bucket,
             MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    ContextBucketRange{bucket}.validate(ContextBucket::M1, "kpool_sparse_mla");
    const int T = query.ne[2], B = query.ne[3], sms = execution.multiprocessor_count();
    const auto profile = kpool_sparse_mla_profile(factors, query.ne[1]);
    sparse_mla_geometry(profile, cache.storage, B, T, T, sms);
    require(std::isfinite(attention_scale) && attention_scale > 0, "Sparse MLA scale");
    const auto device = execution.device();
    std::array<operand::Range, 16> inputs{};
    int n            = 0;
    const auto input = [&](const Tensor& t, DType type, std::initializer_list<int> shape,
                           int alignment, const char* name) {
        operand::tensor(t, type, shape, alignment, kOperation, name, device);
        inputs[n++] = operand::range(t, kOperation);
    };
    input(query, DType::BF16, {256, 64, T, B}, 16, "MLA query");
    for (const auto* factor : {&factors.key_up, &factors.value_up})
        for (const auto plane : weight_operands(*factor, 16384, 512, device))
            if (plane.begin != plane.end) inputs[n++] = plane;
    input(batch.positions, DType::I32, {T, B}, 4, "positions");
    const int segments = seq ? seq->num_seqs() : B;
    input(batch.valid_columns, DType::I32, {segments}, 4, "valid columns");
    input(batch.table_rows, DType::I32, {segments}, 4, "table rows");
    if (seq) {
        inputs[n++] = operand::range(seq->cu_seqlens, kOperation);
        inputs[n++] = operand::range(seq->token_to_seq, kOperation);
    }
    input(selection_rows, DType::I32, {T, B}, 4, "selection rows");
    const int selected_rows = selected.pool_ids.ne[1];
    input(selected.pool_ids, DType::I32, {512, selected_rows}, 4, "pool IDs");
    input(selected.pool_counts, DType::I32, {selected_rows}, 4, "pool counts");
    input(selected.selection_end, DType::I32, {selected_rows}, 4, "selection end");
    latent_cache(cache, 512, device);
    const auto& table       = cache.data.records.page_offsets;
    const auto table_blocks = operand::blocks(table, kOperation);
    operand::tensor(out, DType::BF16, {256, 64, T, B}, 16, kOperation, "MLA output", device);
    const auto output = operand::range(out, kOperation);
    for (int i = 0; i < n; ++i) operand::disjoint(output, inputs[i], kOperation);
    require(!operand::overlaps(table_blocks, output), "output overlaps page table");
    operand::device(workspace.device(), device, kOperation, "workspace");
    SparseMlaArguments a{};
    a.query        = static_cast<const __nv_bfloat16*>(query.data);
    a.key_up       = factors.key_up.qdata;
    a.value_up     = factors.value_up.qdata;
    a.key_scales   = static_cast<const std::uint8_t*>(factors.key_up.scales);
    a.value_scales = static_cast<const std::uint8_t*>(factors.value_up.scales);
    a.key_mxfp8    = factors.key_up.qtype == QType::MXFP8;
    a.value_mxfp8  = factors.value_up.qtype == QType::MXFP8;
    if (cache.storage == KpoolMlaStorage::BFloat16)
        a.latent = static_cast<const __nv_bfloat16*>(cache.data.data());
    else {
        a.latent_fp8    = static_cast<const std::uint8_t*>(cache.data.data());
        a.latent_scales = static_cast<const std::uint8_t*>(cache.scales.data());
    }
    a.latent_bytes   = cache.data.records.storage.bytes - cache.data.plane.tensor.region.offset;
    a.valid          = static_cast<const int*>(batch.valid_columns.data);
    a.table_rows     = static_cast<const int*>(batch.table_rows.data);
    a.selection_rows = static_cast<const int*>(selection_rows.data);
    a.ids            = static_cast<const int*>(selected.pool_ids.data);
    a.counts         = static_cast<const int*>(selected.pool_counts.data);
    a.ends           = static_cast<const int*>(selected.selection_end.data);
    a.tables         = static_cast<const std::int64_t*>(table.data);
    a.table_stride   = table.nb[1] / 8;
    a.out            = static_cast<__nv_bfloat16*>(out.data);
    a.tokens         = T;
    a.batch          = B;
    a.scale          = attention_scale;
    auto scope       = workspace.scope();
    const auto bytes = sparse_mla_capacity(profile, cache.storage, T, T, B, bucket, sms);
    MemorySpan scratch{};
    if (bytes) {
        scratch          = workspace.alloc_bytes(bytes, 256);
        const auto range = operand::range(scratch.data, scratch.bytes, kOperation);
        for (int i = 0; i < n; ++i) operand::disjoint(range, inputs[i], kOperation);
        operand::disjoint(range, output, kOperation);
        require(!operand::overlaps(table_blocks, range), "workspace overlaps page table");
    }
    if (segments > 1 && seq)
        launch_packed_sparse_mla({a, static_cast<const int*>(seq->token_to_seq.data)}, scratch,
                                 profile, cache.storage, bucket, sms, execution.stream());
    else
        launch_sparse_mla(a, scratch, profile, cache.storage, bucket, sms, execution.stream());
}
} // namespace

void kpool_sparse_mla(const Tensor& query, const KpoolMlaFactors& factors,
                      const ConstKpoolMlaLatentCacheView& cache, const KpoolMlaBatchView& batch,
                      const KpoolBlockSelection& selected, const Tensor& selection_rows,
                      float attention_scale, ContextBucket bucket, MemoryArena& workspace,
                      Tensor& out, CudaExecutionView execution) {
    execute(query, factors, cache, batch, nullptr, selected, selection_rows, attention_scale,
            bucket, workspace, out, execution);
}

std::size_t kpool_sparse_mla_ragged_workspace_capacity_bytes(
    const KpoolSparseMlaProfile& profile, KpoolMlaStorage storage, ContextBucketRange buckets,
    int min_tokens, int max_tokens, int min_segments, int max_segments, int sm_count) {
    const int packed =
        packed_extent_start(min_tokens, max_tokens, min_segments, max_segments, "kpool_sparse_mla");
    // Both layouts use the same plan for each possible window width. A packed
    // window has no inactive columns, so it uses the single-batch split policy.
    return kpool_sparse_mla_workspace_capacity_bytes(profile, storage, buckets, 1,
                                                     min_segments == 1 ? min_tokens : packed,
                                                     max_tokens, sm_count);
}

void kpool_sparse_mla(const Tensor& query, const KpoolMlaFactors& factors,
                      const ConstKpoolMlaLatentCacheView& cache,
                      const KpoolMlaRaggedBatchView& batch, const KpoolBlockSelection& selected,
                      const Tensor& selection_rows, float attention_scale, ContextBucket bucket,
                      MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    constexpr const char* op = "kpool_sparse_mla";
    const auto& seq          = batch.ragged;
    const auto device        = execution.device();
    operand::ragged_batch(seq, batch.positions, batch.table_rows, op, device);
    const int n = seq.total_tokens();
    operand::tensor(query, DType::BF16, {256, 64, n}, 16, op, "query", device);
    operand::tensor(out, DType::BF16, {256, 64, n}, 16, op, "output", device);
    operand::tensor(selection_rows, DType::I32, {n}, 4, op, "selection rows", device);
    const KpoolMlaBatchView flat{batch.positions.view({n, 1}), seq.seq_lengths, batch.table_rows};
    auto result = out.view({256, 64, n, 1});
    execute(query.view({256, 64, n, 1}), factors, cache, flat, &seq, selected,
            selection_rows.view({n, 1}), attention_scale, bucket, workspace, result, execution);
}

} // namespace ninfer::ops
