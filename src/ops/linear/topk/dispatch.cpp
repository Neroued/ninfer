#include "ops/linear/topk/dispatch.h"

#include <stdexcept>
#include "core/platform.h"

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

int linear_topk_large_threshold(LinearTopKProfile profile) {
#if NINFER_TARGET_SPARK
    return profile == LinearTopKProfile::Q4Mapped ? 256 : 128;
#else
    return 256;
#endif
}

LinearTopKPlan select_linear_topk(LinearTopKProfile profile, int columns) {
    if (columns < 1 || columns > kLinearTopKMaxChunkColumns)
        throw std::invalid_argument("linear_topk: invalid internal chunk size");
    if (columns > 128) {
        if (profile == LinearTopKProfile::Q8Full) return select_topk_q8_n248320_k5120_large();
        if (profile == LinearTopKProfile::Q4Mapped) return select_topk_q4_n131072_k5120_large();
#if !NINFER_TARGET_SPARK
        return select_topk_fp8_n248320_k5120_large();
#else
        auto plan       = select_topk_fp8_n248320_k5120_mma(columns);
        plan.row_window = 8192;
        return plan;
#endif
    }
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
    for_each_linear_topk_extent(
        min_columns, max_columns, linear_topk_large_threshold(profile), [&](int columns) {
            const auto plan = select_linear_topk(profile, columns);
            auto scope      = layout.scope();
            if (plan.row_window && plan.row_window < head_rows)
                (void)layout.alloc(DType::I64, {16, columns}, 256);
            (void)allocate_linear_topk_workspace(
                layout, plan.row_window ? std::min(head_rows, plan.row_window) : head_rows, columns,
                plan.rows_per_producer);
        });
    return layout.peak_bytes();
}
} // namespace ninfer::ops::detail
