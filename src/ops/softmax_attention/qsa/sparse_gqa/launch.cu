#include "ops/softmax_attention/qsa/sparse_gqa/plan.h"

namespace ninfer::ops::detail::qsa {

void launch_sparse_gqa(const Tensor& query, const Tensor& gate,
                       const ConstPagedKVBatchLayerView& cache, const QsaBatchView& batch,
                       const QsaBlockSelection& selected, SparseGqaPlan plan,
                       const SparseGqaWorkspace& scratch, Tensor& out,
                       CudaExecutionView execution) {
    const auto i32  = [](const Tensor& t) { return static_cast<const int*>(t.data); };
    const auto page = [](const ConstPagedPlaneView& t) {
        return SparseGqaPage{static_cast<const std::uint8_t*>(t.data()),
                             std::int64_t(t.row_stride_bytes()),
                             std::int64_t(t.head_stride_bytes())};
    };
    SparseGqaArguments a{static_cast<const __nv_bfloat16*>(query.data),
                         static_cast<const __nv_bfloat16*>(gate.data),
                         static_cast<__nv_bfloat16*>(out.data),
                         page(cache.k_pages),
                         page(cache.v_pages),
                         page(cache.k_scale_pages),
                         page(cache.v_scale_pages),
                         i32(batch.positions),
                         i32(batch.valid_columns),
                         i32(batch.table_rows),
                         static_cast<const std::int64_t*>(cache.k_pages.records.page_offsets.data),
                         i32(selected.block_ids),
                         i32(selected.block_counts),
                         cache.k_pages.records.page_offsets.nb[1] / 8,
                         static_cast<float*>(scratch.numerator.data),
                         static_cast<float*>(scratch.maximum.data),
                         static_cast<float*>(scratch.sum.data),
                         query.ne[2],
                         query.ne[3],
                         plan.splits,
                         plan.key_rows,
                         0};
    const auto queries = std::int64_t(a.tokens) * a.batch;
    // This slab only bounds grid.x (including the per-head merge); scratch/output indexing
    // stays global and T is not a semantic launch limit.
    constexpr int slab = 1 << 24;
    for (a.begin = 0; a.begin < queries; a.begin += slab) {
        const int n = int(std::min<std::int64_t>(slab, queries - a.begin));
        const auto selected_launch =
            select_sparse_gqa_launch(cache, plan, n, a.batch, execution.multiprocessor_count());
        launch_sparse_gqa_instance(a, cache.storage, cache.num_kv_heads, plan, selected_launch, n,
                                   execution);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::qsa
