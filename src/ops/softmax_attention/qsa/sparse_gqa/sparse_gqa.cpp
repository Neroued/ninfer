#include "ninfer/ops/qsa.h"
#include "ops/softmax_attention/qsa/sparse_gqa/plan.h"
#include "ops/softmax_attention/qsa/validation.h"

namespace ninfer::ops {
namespace {
constexpr detail::qsa::ViewValidation v{"qsa_sparse_gqa"};

void profile(KvCacheStorage storage, int lo, int hi, int batch,
             QsaSparseGqaExecutionEnvelope envelope, std::int32_t multiprocessor_count) {
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

std::size_t qsa_sparse_gqa_workspace_capacity_bytes(KvCacheStorage storage, std::int32_t lo,
                                                    std::int32_t hi, std::int32_t batch,
                                                    QsaSparseGqaExecutionEnvelope envelope,
                                                    std::int32_t multiprocessor_count) {
    profile(storage, lo, hi, batch, envelope, multiprocessor_count);
    std::size_t maximum = 0;
    // Above this bound every extent uses direct output. Below it, allocation is not monotonic
    // because the split count decreases with query parallelism.
    const int end = std::min(hi, (detail::qsa::kSparseGqaDirectQueries - 1) / batch);
    for (int t = lo; t <= end; ++t) {
        WorkspaceLayoutBuilder layout;
        detail::qsa::allocate_sparse_gqa_workspace(
            layout, t, batch,
            detail::qsa::select_sparse_gqa_plan(storage, t, batch, envelope, multiprocessor_count));
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

void qsa_sparse_gqa(const Tensor& query, const Tensor& gate, const PagedKVBatchLayerView& cache,
                    const QsaBatchView& batch, const QsaBlockSelection& selected,
                    QsaSparseGqaExecutionEnvelope envelope, MemoryArena& workspace, Tensor& out,
                    CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("QSA workspace belongs to another device");
    }
    const detail::qsa::ViewValidation v{"qsa_sparse_gqa", execution.device()};
    const int tokens = query.ne[2], rows = query.ne[3], pages = cache.k_pages.ne[3];
    profile(cache.storage, tokens, tokens, rows, envelope, execution.multiprocessor_count());
    v.dense(query, DType::BF16, {256, 24, tokens, rows}, 16, "query");
    v.dense(gate, DType::BF16, {256, 24, tokens, rows}, 16, "gate");
    v.dense(out, DType::BF16, {256, 24, tokens, rows}, 16, "output");
    v.require(cache.head_dim == 256 && cache.num_kv_heads == 2, "cache geometry");
    const auto layout = paged_kv_storage_layout(cache.storage, 256);
    const auto plane  = [&](const Tensor& data, const Tensor& scales, PagedKVVectorLayout format) {
        v.plane(data, format.data_dtype, {format.data_leading_extent, 64, 2, pages}, "cache data");
        if (format.has_scale())
            v.plane(scales, format.scale_dtype, {format.scale_leading_extent, 64, 2, pages},
                     "cache scales");
        else
            v.require(!scales.data, "absent scales");
    };
    plane(cache.k_pages, cache.k_scale_pages, layout.key);
    plane(cache.v_pages, cache.v_scale_pages, layout.value);
    v.plane(cache.block_tables, DType::I32,
            {cache.block_tables.ne[0], cache.block_tables.ne[1], 1, 1}, "block tables");
    v.dense(batch.positions, DType::I32, {tokens, rows, 1, 1}, 4, "positions");
    v.metadata(batch.valid_columns, rows, "valid columns");
    v.metadata(batch.table_rows, rows, "table rows");
    v.dense(selected.block_ids, DType::I32, {512, tokens, rows, 1}, 4, "block ids");
    v.dense(selected.block_counts, DType::I32, {tokens, rows, 1, 1}, 4, "block counts");
    const Tensor* inputs[]{&query,
                           &gate,
                           &cache.k_pages,
                           &cache.v_pages,
                           &cache.k_scale_pages,
                           &cache.v_scale_pages,
                           &cache.block_tables,
                           &batch.positions,
                           &batch.valid_columns,
                           &batch.table_rows,
                           &selected.block_ids,
                           &selected.block_counts};
    for (const auto* input : inputs)
        if (input->data) v.require(!v.overlap(out, *input), "output/input overlap");
    const auto plan    = detail::qsa::select_sparse_gqa_plan(cache.storage, tokens, rows, envelope,
                                                             execution.multiprocessor_count());
    auto scope         = workspace.scope();
    const auto scratch = detail::qsa::allocate_sparse_gqa_workspace(workspace, tokens, rows, plan);
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
