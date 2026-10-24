#pragma once

#include "ops/softmax_attention/kpool_mla/prepare/plan.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail::kpool {

struct TailField {
    __nv_bfloat16* data         = nullptr;
    const std::int64_t* offsets = nullptr;
    std::int64_t column_stride  = 0; // BF16 elements; offsets remain bytes.
};

struct RawField {
    __nv_bfloat16* data        = nullptr;
    std::int64_t column_stride = 0, row_stride = 0;
};

struct PrepareArguments {
    const __nv_bfloat16* x;
    const __nv_bfloat16* weight;
    const __nv_bfloat16 *latent_norm, *query_norm, *key_norm, *key_bias, *ape;
    const int *positions, *valid, *table_rows, *record_offsets;
    const std::int64_t* page_offsets;
    std::int64_t table_stride;
    __nv_bfloat16 *latent, *index, *query;
    float* coefficients;
    TailField before[2], after[2];
    RawField records[2];
    float* projected;
    __nv_bfloat16 *gates, *carry;
    int tokens, batch, rows, query_rank, heads, norm_dim;
    int splits = 1, stage_columns = 0;
    int first_row = 0, first_column = 0, first_span = 0, last_row = 0, last_stop = 0;
    std::uint32_t token_multiplier = 0, token_shift = 0;
    float rms_epsilon, key_epsilon;
    bool update;
};

void launch_prepare(const PrepareArguments& args, int window_columns, cudaStream_t stream);
} // namespace ninfer::ops::detail::kpool
