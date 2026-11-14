#include "ops/softmax_attention/qsa/indexer/plan.h"

namespace ninfer::ops::detail::qsa {
void launch_indexer(const Tensor& query, const ConstQsaIndexCacheView& cache,
                    const QsaBatchView& batch, IndexerPlan plan, const IndexerWorkspace& scratch,
                    QsaBlockSelection& out, CudaExecutionView execution) {
    const auto i32 = [](const Tensor& t) { return static_cast<const std::int32_t*>(t.data); };
    const IndexerArguments input{
        static_cast<const __nv_bfloat16*>(query.data),
        static_cast<const __nv_bfloat16*>(cache.keys.data()),
        i32(batch.positions),
        i32(batch.valid_columns),
        i32(batch.table_rows),
        static_cast<const std::int64_t*>(cache.keys.records.page_offsets.data),
        std::int64_t(cache.keys.row_stride_bytes() / 2),
        cache.keys.records.page_offsets.nb[1] / 8,
        static_cast<float*>(scratch.scores.data),
        static_cast<std::uint64_t*>(scratch.candidates.data),
        static_cast<std::int32_t*>(out.block_ids.data),
        static_cast<std::int32_t*>(out.block_counts.data),
        query.ne[2],
        query.ne[3],
        0,
        plan.window_tokens,
        plan.columns,
        plan.partitions};
    launch_indexer_windows(input, plan,
                           cache.keys.records.storage.bytes - cache.keys.plane.tensor.region.offset,
                           cache.keys.plane.tensor.region.alignment, execution);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::qsa
