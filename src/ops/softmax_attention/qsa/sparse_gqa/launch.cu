#include "ops/softmax_attention/qsa/sparse_gqa/plan.h"
#include "ops/common/host/operands.h"
#include "core/platform.h"

#include <algorithm>

namespace ninfer::ops::detail::qsa {

void launch_sparse_gqa(const Tensor& query, const Tensor& gate,
                       const ConstPagedKVBatchLayerView& cache, const QsaBatchView& batch,
                       const QsaBlockSelection& selected, ContextBucket bucket, SparseGqaPlan plan,
                       const SparseGqaWorkspace& scratch, Tensor& out, CudaExecutionView execution,
                       const Tensor* token_to_seq, std::int64_t window_begin, int window_queries) {
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
    a.min_split_tiles  = plan.min_split_tiles;
    a.direct_single    = plan.direct_single;
    const auto aligned = [](const ConstPagedPlaneView& t) {
        return common::operand::aligned(t.data(), 16) && t.row_stride_bytes() % 16 == 0 &&
               t.head_stride_bytes() % 16 == 0 && t.plane.tensor.region.alignment % 16 == 0;
    };
    const bool vector_cache = aligned(cache.k_pages) && aligned(cache.v_pages);
    if (token_to_seq) {
        a.begin           = window_begin;
        const auto launch = select_sparse_gqa_launch(
            cache.storage, cache.num_kv_heads, bucket, plan, window_queries, 1,
            execution.multiprocessor_count(), vector_cache);
        const bool adjacent =
            NINFER_TARGET_RTX5090 && bucket > ContextBucket::K2 && launch.splits > 1;
        launch_sparse_gqa_instance(
            RaggedSparseGqaArguments{a, i32(*token_to_seq), window_queries, adjacent},
            cache.storage, cache.num_kv_heads, plan, launch, window_queries, execution);
    } else {
        const auto queries = std::int64_t(a.tokens) * a.batch;
        for (a.begin = 0; a.begin < queries; a.begin += kSparseGqaSlab) {
            const int n = int(std::min<std::int64_t>(kSparseGqaSlab, queries - a.begin));
            const auto selected_launch =
                select_sparse_gqa_launch(cache.storage, cache.num_kv_heads, bucket, plan, n,
                                         a.batch, execution.multiprocessor_count(), vector_cache);
            launch_sparse_gqa_instance(a, cache.storage, cache.num_kv_heads, plan, selected_launch,
                                       n, execution);
        }
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::qsa
