#pragma once

#include "ninfer/ops/kpool_mla.h"
#include "core/layout.h"

#include <algorithm>

namespace ninfer::ops::detail::kpool {

enum class ProjectionRoute {
    Gemv,
    Simt2,
    Simt4,
    Sliced8,
    Sliced16,
    SlicedHistory,
    Mma32,
    TmaSmall,
    TmaHistory,
    Tma128,
    Split32,
    Split64
};

struct PrepareGeometry {
    int rows, query_rank, heads;

    int norm_dim() const { return 640 + query_rank; }
};

PrepareGeometry prepare_geometry(const KpoolMlaPrepareProfile& profile, KpoolMlaStorage storage,
                                 int batch, int min_tokens, int max_tokens, int sm_count);
ProjectionRoute select_projection(int columns, int rows);
int prepare_window_columns();

inline int projection_splits(ProjectionRoute route) {
    return route == ProjectionRoute::Split32 || route == ProjectionRoute::Split64 ? 4 : 1;
}

struct PrepareWorkspace {
    TensorRegion projected;
    TensorRegion gates;
    TensorRegion carry;
    std::size_t bytes = 0;
    int columns       = 0;
};

inline PrepareWorkspace prepare_workspace(PrepareGeometry geometry, bool update, int tokens,
                                          int batch) {
    PrepareWorkspace result;
    const auto total = std::int64_t(tokens) * batch;
    result.columns   = static_cast<int>(std::min<std::int64_t>(total, prepare_window_columns()));
    LayoutBuilder builder;
    const auto projection_words = [&](int columns) {
        const int splits = projection_splits(select_projection(columns, geometry.rows));
        return (splits == 1 ? geometry.norm_dim() : splits * geometry.rows) * columns;
    };
    const int remainder = total % result.columns;
    const auto words =
        std::max(projection_words(result.columns), remainder ? projection_words(remainder) : 0);
    result.projected = builder.add_tensor(DType::FP32, {words}, 256, "Kpool projection");
    if (update) {
        if (projection_splits(select_projection(result.columns, geometry.rows)) == 1 ||
            (remainder && projection_splits(select_projection(remainder, geometry.rows)) == 1))
            result.gates =
                builder.add_tensor(DType::BF16, {128, result.columns}, 256, "Kpool gates");
        if (total > result.columns && result.columns % tokens != 0)
            result.carry = builder.add_tensor(DType::BF16, {128, 3, 2, batch}, 256, "Kpool carry");
    }
    result.bytes = builder.finish(1);
    return result;
}

inline std::size_t prepare_capacity(PrepareGeometry geometry, bool update, int lo, int hi,
                                    int batch) {
    // Below the fixed window, examine every shape transition. Above it, resource use repeats
    // with the final window's residue; the carry region is then always present.
    std::size_t result = 0;
    const int window   = prepare_window_columns();
    const auto end     = std::min(std::int64_t(hi), std::int64_t(lo) + 2 * window);
    for (std::int64_t t = lo; t <= end; ++t)
        result = std::max(result, prepare_workspace(geometry, update, t, batch).bytes);
    return result;
}
} // namespace ninfer::ops::detail::kpool
