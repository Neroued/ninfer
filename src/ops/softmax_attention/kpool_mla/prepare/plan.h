#pragma once

#include "ninfer/ops/kpool_mla.h"
#include "core/layout.h"

#include <algorithm>
#include <numeric>
#include <limits>

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
    Split64,
    Fp8Simt1,
    Fp8Simt4,
    Fp8Sliced8,
    Fp8Sliced16,
    Fp8Mma32,
    Fp8Mma64,
    Mxfp8A8Short,
    Mxfp8A8Medium,
    Mxfp8A8Large
};

inline bool projection_uses_a8(ProjectionRoute route) {
    switch (route) {
    case ProjectionRoute::Mxfp8A8Short:
    case ProjectionRoute::Mxfp8A8Medium:
    case ProjectionRoute::Mxfp8A8Large:
        return true;
    default:
        return false;
    }
}

struct PrepareGeometry {
    int rows, query_rank, heads;
    ProjectionProfile weights;

    int norm_dim() const { return 640 + query_rank; }
};

PrepareGeometry prepare_geometry(const KpoolMlaPrepareProfile& profile, KpoolMlaStorage storage,
                                 KvCacheVectorFormat index_format, int batch, int min_tokens,
                                 int max_tokens, int sm_count);

enum class PairedProjectionRoute { None, Simt1, Sliced8, Sliced16 };

struct PrepareProjectionPlan {
    ProjectionRoute first, second;
    bool quantize                = false;
    int splits                   = 1;
    PairedProjectionRoute paired = PairedProjectionRoute::None;
    bool skip_empty              = true;
    int last_column              = std::numeric_limits<int>::max();
};

PrepareProjectionPlan prepare_projection_plan(const PrepareGeometry&, int columns);
int prepare_window_columns();

inline int projection_splits(ProjectionRoute route) {
    return route == ProjectionRoute::Split32 || route == ProjectionRoute::Split64 ? 4 : 1;
}

struct PrepareWorkspace {
    TensorRegion projected;
    TensorRegion gates;
    TensorRegion carry;
    TensorRegion activation_codes, activation_scales;
    std::size_t bytes = 0;
    int columns       = 0;
};

inline PrepareWorkspace prepare_workspace(PrepareGeometry geometry, bool update, int tokens,
                                          int batch) {
    PrepareWorkspace result;
    const auto total = std::int64_t(tokens) * batch;
    result.columns   = static_cast<int>(std::min<std::int64_t>(total, prepare_window_columns()));
    LayoutBuilder builder;
    const int remainder         = total % result.columns;
    const auto full             = prepare_projection_plan(geometry, result.columns);
    const auto tail             = remainder ? prepare_projection_plan(geometry, remainder) : full;
    const auto projection_words = [&](int columns, const PrepareProjectionPlan& plan) {
        return (plan.splits == 1 ? geometry.norm_dim() : plan.splits * geometry.rows) * columns;
    };
    const auto words = std::max(projection_words(result.columns, full),
                                remainder ? projection_words(remainder, tail) : 0);
    result.projected = builder.add_tensor(DType::FP32, {words}, 256, "Kpool projection");
    if (update) {
        if (full.splits == 1 || (remainder && tail.splits == 1))
            result.gates =
                builder.add_tensor(DType::BF16, {128, result.columns}, 256, "Kpool gates");
        if (total > result.columns && result.columns % tokens != 0)
            result.carry = builder.add_tensor(DType::BF16, {128, 3, 2, batch}, 256, "Kpool carry");
    }
    if (full.quantize || (remainder && tail.quantize)) {
        result.activation_codes = builder.add_tensor(DType::FP8_E4M3FN, {4096, result.columns}, 256,
                                                     "Kpool projection activation");
        result.activation_scales = builder.add_tensor(DType::U8, {128, result.columns}, 256,
                                                      "Kpool projection activation scales");
    }
    result.bytes = builder.finish(1);
    return result;
}

inline std::size_t prepare_capacity(const PrepareGeometry& geometry, bool update, int lo, int hi,
                                    int batch) {
    std::size_t result = 0;
    const int window   = prepare_window_columns();
    const int period   = window / std::gcd(window, batch);
    const auto examine = [&](std::int64_t t) {
        if (t >= lo && t <= hi)
            result = std::max(result, prepare_workspace(geometry, update, int(t), batch).bytes);
    };
    const auto endpoint = [&](std::int64_t t) {
        examine(t);
        // A request divisor needs no carry. Its predecessor or earlier same-residue
        // window may need one even though the projection workspace is no larger.
        if (t > 0 && t <= window && window % t == 0) {
            examine(t - 1);
            examine(t - period);
        }
    };
    endpoint(hi);
    for (int first = 1; first <= window;) {
        const auto plan = prepare_projection_plan(geometry, first);
        const int last  = std::min(window, plan.last_column);
        examine(std::min<std::int64_t>(hi, last / batch));
        first = last + 1;
    }
    const std::int64_t begin = std::max<std::int64_t>({lo, window / batch + 1, hi - period + 1});
    if (begin > hi) return result;
    // Within each modular wrap, scratch is monotone up to the next Plan boundary.
    // A complete residue period has at most B wraps; the cost does not grow with T.
    for (std::int64_t wrap = begin * batch / window; wrap <= std::int64_t(hi) * batch / window;
         ++wrap) {
        const auto upper = [&](int residue) {
            const auto t = std::min<std::int64_t>(hi, (wrap * window + residue) / batch);
            if (t >= begin && t * batch / window == wrap) endpoint(t);
        };
        upper(0);
        for (int first = 1; first < window;) {
            const auto plan = prepare_projection_plan(geometry, first);
            const int last  = std::min(window - 1, plan.last_column);
            upper(last);
            first = last + 1;
        }
    }
    return result;
}
} // namespace ninfer::ops::detail::kpool
