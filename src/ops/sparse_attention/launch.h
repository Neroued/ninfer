#pragma once

// ninfer::ops::detail - private launch prototypes for Qwen Sparse Attention.

#include "ninfer/ops/sparse_attention.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

void qsa_append_launch(const Tensor& k, const Tensor& v, const Tensor& index_keys,
                       const Tensor& rope_positions, const Tensor& cache_positions,
                       const Tensor& valid_columns, const Tensor& table_rows,
                       const Tensor& key_norm, const QsaGeometry& geometry,
                       const PagedKVBatchLayerView& kv, const QsaIndexPlanes& index,
                       cudaStream_t stream);

// Selection runs in passes of whole column tiles (columns of one lane); a pass scores into one
// FP32 [max_blocks, kQsaSelectPassColumns] scratch row block.
struct QsaSelectPlan {
    std::int32_t tile_columns;   // columns per tile: 1, 2, 4 or 8
    std::int32_t tiles_per_lane; // ceil(W / tile_columns)
    std::int32_t pass_tiles;     // tiles per pass
    std::int32_t max_blocks;     // score row length
};

[[nodiscard]] QsaSelectPlan qsa_select_plan(std::uint32_t max_visible_keys, std::int32_t ratio,
                                            std::int32_t width, std::int32_t lanes);
[[nodiscard]] std::size_t qsa_select_scratch_bytes(const QsaSelectPlan& plan);

void qsa_select_launch(const Tensor& index_query, const Tensor& query_rope_positions,
                       const Tensor& cache_positions, const Tensor& valid_columns,
                       const Tensor& table_rows, const Tensor& query_norm,
                       const QsaIndexPlanes& index, const QsaGeometry& geometry,
                       const QsaSelectPlan& plan, float* scores, Tensor& selected, Tensor& counts,
                       cudaStream_t stream);

struct QsaAttentionSplit {
    std::int32_t splits;     // CTAs per (column, KV head)
    std::int32_t split_keys; // selected keys per split
};

[[nodiscard]] QsaAttentionSplit qsa_attention_split(std::int32_t columns, std::int32_t kv_heads,
                                                    std::int32_t max_selected);
// The partials of `columns` columns split as a call of `split_columns` (>= columns) would.
[[nodiscard]] std::size_t
qsa_attention_partial_bytes(std::int32_t columns, std::int32_t query_heads, std::int32_t kv_heads,
                            std::int32_t max_selected, std::int32_t split_columns);
// `partial` holds qsa_attention_partial_bytes; it is unused when the call does not split.
void qsa_attention_launch(const Tensor& q, const Tensor& selected, const Tensor& counts,
                          const Tensor& table_rows, const PagedKVBatchLayerView& kv, float scale,
                          float* partial, Tensor& out, cudaStream_t stream,
                          std::int32_t split_columns);

} // namespace ninfer::ops::detail
