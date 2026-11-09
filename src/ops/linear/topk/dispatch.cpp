#include "ops/linear/topk/dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {
LinearTopKProfile select_linear_topk_profile(QType qtype, std::int32_t head_rows,
                                             std::int32_t input_rows) {
    if (input_rows != kLinearTopKHidden) {
        throw std::invalid_argument("linear_topk: unsupported head profile");
    }
    if (head_rows == kLinearTopKFullRows && qtype == QType::Q8_G32_FP16) {
        return LinearTopKProfile::Q8Full;
    }
    if (head_rows == kLinearTopKFullRows && qtype == QType::FP8_E4M3FN_ROW_BF16) {
        return LinearTopKProfile::Fp8Full;
    }
    if (head_rows == kLinearTopKMappedRows && qtype == QType::Q4_G64_FP16) {
        return LinearTopKProfile::Q4Mapped;
    }
    throw std::invalid_argument("linear_topk: unsupported head profile");
}

LinearTopKPlan select_linear_topk(LinearTopKProfile profile, int columns) {
    if (columns < 1 || columns > kLinearTopKMaxChunkColumns)
        throw std::invalid_argument("linear_topk: invalid internal chunk size");
    switch (profile) {
    case LinearTopKProfile::Q8Full:
        if (columns <= 24) return {select_topk_q8_n248320_k5120_sliced(columns), 128};
        return select_topk_q8_n248320_k5120_mma(columns);
    case LinearTopKProfile::Fp8Full:
        if (columns <= 24) return {select_topk_fp8_n248320_k5120_sliced(columns), 128};
        return select_topk_fp8_n248320_k5120_mma(columns);
    case LinearTopKProfile::Q4Mapped:
        if (columns <= 16) return {select_topk_q4_n131072_k5120_sliced(columns), 16};
        return select_topk_q4_n131072_k5120_mma(columns);
    }
    throw std::logic_error("linear_topk: unknown profile");
}

std::size_t linear_topk_capacity(LinearTopKProfile profile, int head_rows, int min_columns,
                                 int max_columns) {
    if (min_columns < 1 || max_columns < min_columns)
        throw std::invalid_argument("linear_topk workspace: invalid column interval");
    WorkspaceLayoutBuilder layout;
    for (int columns = 1; columns <= kLinearTopKMaxChunkColumns; ++columns) {
        const bool reachable = linear_topk_chunk_reachable(columns, min_columns, max_columns);
        if (!reachable) continue;
        const auto plan = select_linear_topk(profile, columns);
        auto scope      = layout.scope();
        (void)allocate_linear_topk_workspace(layout, head_rows, columns, plan.rows_per_producer);
    }
    return layout.peak_bytes();
}
} // namespace ninfer::ops::detail
