#pragma once

// Shared pointwise comparison mechanics for Op tests. Each semantic Op owns the
// concrete criterion used by its suite; this file deliberately defines no
// cross-Op tolerance presets.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace ninfer::test {

struct PointwiseCriterion {
    double absolute;
    double relative;
};

struct PointwiseStats {
    double maximum_absolute_error    = 0.0;
    double maximum_relative_error    = 0.0;
    double maximum_criterion_ratio   = 0.0;
    std::int64_t maximum_error_index = -1;
    std::int64_t maximum_ratio_index = -1;
    double actual_at_maximum         = 0.0;
    double reference_at_maximum      = 0.0;
    std::int64_t first_violation     = -1;
    std::int64_t non_finite_count    = 0;
};

inline PointwiseStats compute_pointwise_stats(const double* actual, const double* reference,
                                              std::int64_t count,
                                              const PointwiseCriterion& criterion) {
    PointwiseStats stats;
    for (std::int64_t index = 0; index < count; ++index) {
        const double got      = actual[index];
        const double expected = reference[index];
        if (!std::isfinite(got) || !std::isfinite(expected)) {
            ++stats.non_finite_count;
            if (stats.first_violation < 0) stats.first_violation = index;
            continue;
        }

        const double absolute_error = std::abs(got - expected);
        const double scale          = std::max(std::abs(got), std::abs(expected));
        const double relative_error = scale == 0.0 ? 0.0 : absolute_error / scale;
        if (absolute_error > stats.maximum_absolute_error) {
            stats.maximum_absolute_error = absolute_error;
            stats.maximum_error_index    = index;
            stats.actual_at_maximum      = got;
            stats.reference_at_maximum   = expected;
        }
        stats.maximum_relative_error = std::max(stats.maximum_relative_error, relative_error);

        const double limit = criterion.absolute + criterion.relative * std::abs(expected);
        const double criterion_ratio =
            limit == 0.0 ? (absolute_error == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
                         : absolute_error / limit;
        if (criterion_ratio > stats.maximum_criterion_ratio) {
            stats.maximum_criterion_ratio = criterion_ratio;
            stats.maximum_ratio_index     = index;
        }
        if (absolute_error > limit && stats.first_violation < 0) { stats.first_violation = index; }
    }
    return stats;
}

inline bool pointwise_passes(const PointwiseStats& stats, std::int64_t count) {
    return count > 0 && stats.non_finite_count == 0 && stats.first_violation < 0;
}

struct ReductionCriterion {
    double relative_l2;
    double gross_absolute;
    double gross_relative_to_max_reference;
    // Optional absolute floor for RMSE <= absolute_rms + relative_l2 * reference RMS.
    double absolute_rms = 0.0;
};

struct ReductionStats {
    double relative_l2                = 0.0;
    double root_mean_squared_error    = 0.0;
    double reference_root_mean_square = 0.0;
    double maximum_absolute_error     = 0.0;
    double maximum_absolute_reference = 0.0;
    std::int64_t maximum_error_index  = -1;
    std::int64_t first_non_finite     = -1;
    double actual_at_maximum          = 0.0;
    double reference_at_maximum       = 0.0;
};

inline ReductionStats compute_reduction_stats(const double* actual, const double* reference,
                                              std::int64_t count) {
    ReductionStats stats;
    long double squared_error     = 0.0L;
    long double squared_reference = 0.0L;
    for (std::int64_t index = 0; index < count; ++index) {
        const double got      = actual[index];
        const double expected = reference[index];
        if (!std::isfinite(got) || !std::isfinite(expected)) {
            if (stats.first_non_finite < 0) stats.first_non_finite = index;
            continue;
        }

        const double error          = got - expected;
        const double absolute_error = std::abs(error);
        squared_error += static_cast<long double>(error) * error;
        squared_reference += static_cast<long double>(expected) * expected;
        stats.maximum_absolute_reference =
            std::max(stats.maximum_absolute_reference, std::abs(expected));
        if (absolute_error > stats.maximum_absolute_error) {
            stats.maximum_absolute_error = absolute_error;
            stats.maximum_error_index    = index;
            stats.actual_at_maximum      = got;
            stats.reference_at_maximum   = expected;
        }
    }
    stats.relative_l2 = std::sqrt(static_cast<double>(squared_error)) /
                        std::max(std::sqrt(static_cast<double>(squared_reference)), 1.0e-30);
    if (count > 0) {
        stats.root_mean_squared_error =
            std::sqrt(static_cast<double>(squared_error / static_cast<long double>(count)));
        stats.reference_root_mean_square =
            std::sqrt(static_cast<double>(squared_reference / static_cast<long double>(count)));
    }
    return stats;
}

inline double gross_error_limit(const ReductionStats& stats, const ReductionCriterion& criterion) {
    return criterion.gross_absolute +
           criterion.gross_relative_to_max_reference * stats.maximum_absolute_reference;
}

inline double reduction_rms_limit(const ReductionStats& stats,
                                  const ReductionCriterion& criterion) {
    return criterion.absolute_rms + criterion.relative_l2 * stats.reference_root_mean_square;
}

inline bool reduction_passes(const ReductionStats& stats, std::int64_t count,
                             const ReductionCriterion& criterion) {
    const bool norm_passes =
        criterion.absolute_rms == 0.0
            ? stats.relative_l2 <= criterion.relative_l2
            : stats.root_mean_squared_error <= reduction_rms_limit(stats, criterion);
    return count > 0 && stats.first_non_finite < 0 && norm_passes &&
           stats.maximum_absolute_error <= gross_error_limit(stats, criterion);
}

} // namespace ninfer::test
