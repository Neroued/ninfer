#include "ninfer/ops/qsa.h"
#include "ops/softmax_attention/qsa/sparse_gqa/plan.h"
#include "ops/softmax_attention/qsa/validation.h"

namespace ninfer::ops {
namespace {
constexpr detail::qsa::ViewValidation v{"qsa_sparse_gqa"};

void profile(KvCacheStorage storage, int query_heads, int kv_heads, int lo, int hi, int batch,
             QsaSparseGqaExecutionEnvelope envelope, std::int32_t multiprocessor_count) {
    v.require((query_heads == 24 && kv_heads == 2) || (query_heads == 12 && kv_heads == 1),
              "query/KV heads");
    v.require(envelope.max_visible_tokens <= 262144, "visible-token capacity");
    v.require(storage == KvCacheStorage::BFloat16 || storage == KvCacheStorage::Fp8E4M3Row256 ||
                  storage == KvCacheStorage::Int8Group64 ||
                  storage == KvCacheStorage::Nvfp4Group16 ||
                  storage == KvCacheStorage::Fp8KeyNvfp4Value,
              "cache storage");
    v.require(lo > 0 && hi >= lo && batch >= 1 && batch <= 8, "T/B interval");
    v.require(multiprocessor_count > 0, "SM count");
}
} // namespace

std::size_t qsa_sparse_gqa_workspace_capacity_bytes(KvCacheStorage storage,
                                                    std::int32_t query_heads, std::int32_t kv_heads,
                                                    std::int32_t lo, std::int32_t hi,
                                                    std::int32_t batch,
                                                    QsaSparseGqaExecutionEnvelope envelope,
                                                    std::int32_t multiprocessor_count) {
    profile(storage, query_heads, kv_heads, lo, hi, batch, envelope, multiprocessor_count);
    std::size_t maximum = 0;
    // Above this bound every extent uses direct output. Below it, allocation is not monotonic
    // because the split count decreases with query parallelism.
    const int end = std::min(
        hi, (detail::qsa::sparse_gqa_direct_queries(kv_heads, multiprocessor_count) - 1) / batch);
    for (int t = lo; t <= end; ++t) {
        WorkspaceLayoutBuilder layout;
        detail::qsa::allocate_sparse_gqa_workspace(
            layout, t, batch, query_heads,
            detail::qsa::select_sparse_gqa_plan(storage, t, batch, kv_heads, envelope,
                                                multiprocessor_count));
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

void qsa_sparse_gqa(const Tensor& query, const Tensor& gate,
                    const ConstPagedKVBatchLayerView& cache, const QsaBatchView& batch,
                    const QsaBlockSelection& selected, QsaSparseGqaExecutionEnvelope envelope,
                    MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("QSA workspace belongs to another device");
    }
    const detail::qsa::ViewValidation v{"qsa_sparse_gqa", execution.device()};
    const int tokens = query.ne[2], rows = query.ne[3];
    const int query_heads = query.ne[1], kv_heads = cache.num_kv_heads;
    profile(cache.storage, query_heads, kv_heads, tokens, tokens, rows, envelope,
            execution.multiprocessor_count());
    v.dense(query, DType::BF16, {256, query_heads, tokens, rows}, 16, "query");
    v.dense(gate, DType::BF16, {256, query_heads, tokens, rows}, 16, "gate");
    v.dense(out, DType::BF16, {256, query_heads, tokens, rows}, 16, "output");
    v.require(cache.head_dim == 256, "cache geometry");
    const auto layout = paged_kv_storage_layout(cache.storage, 256);
    const auto plane  = [&](const ConstPagedPlaneView& data, const ConstPagedPlaneView& scales,
                           PagedKVVectorLayout format) {
        common::operand::paged_plane(
            data, format.data_dtype, {format.data_leading_extent, 64, kv_heads},
            dtype_size(format.data_dtype), "qsa_sparse_gqa", "cache data", execution.device());
        common::operand::same_paged_records(cache.k_pages, data, "qsa_sparse_gqa", "cache");
        v.require(data.records.page_tokens == 64 && data.plane.geometry.tokens_per_row == 1,
                   "record geometry");
        if (format.has_scale()) {
            common::operand::paged_plane(scales, format.scale_dtype,
                                          {format.scale_leading_extent, 64, kv_heads},
                                          dtype_size(format.scale_dtype), "qsa_sparse_gqa",
                                          "cache scales", execution.device());
            common::operand::same_paged_records(data, scales, "qsa_sparse_gqa", "scales");
        } else {
            v.require(!scales.data(), "absent scales");
        }
    };
    plane(cache.k_pages, cache.k_scale_pages, layout.key);
    plane(cache.v_pages, cache.v_scale_pages, layout.value);
    v.dense(batch.positions, DType::I32, {tokens, rows, 1, 1}, 4, "positions");
    v.metadata(batch.valid_columns, rows, "valid columns");
    v.metadata(batch.table_rows, rows, "table rows");
    v.dense(selected.block_ids, DType::I32, {512, tokens, rows, 1}, 4, "block ids");
    v.dense(selected.block_counts, DType::I32, {tokens, rows, 1, 1}, 4, "block counts");
    const Tensor* inputs[]{&query,
                           &gate,
                           &cache.k_pages.records.page_offsets,
                           &batch.positions,
                           &batch.valid_columns,
                           &batch.table_rows,
                           &selected.block_ids,
                           &selected.block_counts};
    for (const auto* input : inputs)
        if (input->data) v.require(!v.overlap(out, *input), "output/input overlap");
    const auto plan = detail::qsa::select_sparse_gqa_plan(
        cache.storage, tokens, rows, kv_heads, envelope, execution.multiprocessor_count());
    auto scope = workspace.scope();
    const auto scratch =
        detail::qsa::allocate_sparse_gqa_workspace(workspace, tokens, rows, query_heads, plan);
    for (const auto* work : {&scratch.numerator, &scratch.maximum, &scratch.sum}) {
        if (!work->data) continue;
        v.require(!v.overlap(*work, out), "workspace/output overlap");
        for (const auto* input : inputs)
            if (input->data) v.require(!v.overlap(*work, *input), "workspace/input overlap");
    }
    detail::qsa::launch_sparse_gqa(query, gate, cache, batch, selected, plan, scratch, out,
                                   execution);
}
} // namespace ninfer::ops
