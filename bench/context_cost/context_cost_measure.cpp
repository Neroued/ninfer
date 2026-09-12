#include "context_cost_measure.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace ninfer::bench::context_cost {
namespace {

double median_of(std::vector<double> values) {
    if (values.empty()) { return 0.0; }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle),
                     values.end());
    const double upper = values[middle];
    if ((values.size() & 1U) != 0U) { return upper; }
    const double lower =
        *std::max_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(middle));
    return 0.5 * (lower + upper);
}

bool solve_linear(
    std::array<std::array<double, kMaximumModelCoefficients>, kMaximumModelCoefficients>& matrix,
    std::array<double, kMaximumModelCoefficients>& right, std::size_t size,
    std::array<double, kMaximumModelCoefficients>& solution) {
    double largest = 0.0;
    for (std::size_t row = 0; row < size; ++row) {
        for (std::size_t column = 0; column < size; ++column) {
            largest = std::max(largest, std::abs(matrix[row][column]));
        }
    }
    if (!(largest > 0.0) || !std::isfinite(largest)) { return false; }
    const double pivot_floor = largest * 1.0e-12;

    for (std::size_t column = 0; column < size; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < size; ++row) {
            if (std::abs(matrix[row][column]) > std::abs(matrix[pivot][column])) { pivot = row; }
        }
        if (std::abs(matrix[pivot][column]) <= pivot_floor) { return false; }
        if (pivot != column) {
            std::swap(matrix[pivot], matrix[column]);
            std::swap(right[pivot], right[column]);
        }
        const double diagonal = matrix[column][column];
        for (std::size_t entry = column; entry < size; ++entry) {
            matrix[column][entry] /= diagonal;
        }
        right[column] /= diagonal;
        for (std::size_t row = 0; row < size; ++row) {
            if (row == column) { continue; }
            const double factor = matrix[row][column];
            if (factor == 0.0) { continue; }
            for (std::size_t entry = column; entry < size; ++entry) {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
            right[row] -= factor * right[column];
        }
    }
    for (std::size_t index = 0; index < size; ++index) { solution[index] = right[index]; }
    return true;
}

double floating_prediction(const RegressionPoint& point, const std::vector<double>& coefficients) {
    double result = 0.0;
    for (std::size_t index = 0; index < coefficients.size(); ++index) {
        result += point.features[index] * coefficients[index];
    }
    return result;
}

std::vector<double> nonnegative_fit(const std::vector<RegressionPoint>& points,
                                    std::size_t columns) {
    std::vector<const RegressionPoint*> training;
    for (const RegressionPoint& point : points) {
        if (!point.validation) { training.push_back(&point); }
    }
    if (training.size() < columns) {
        throw std::invalid_argument("cost model has fewer training points than coefficients");
    }

    std::array<double, kMaximumModelCoefficients> scales{};
    for (const RegressionPoint* point : training) {
        if (!(point->elapsed_ns > 0.0) || !std::isfinite(point->elapsed_ns)) {
            throw std::invalid_argument("cost model training time must be finite and positive");
        }
        for (std::size_t column = 0; column < columns; ++column) {
            if (!(point->features[column] >= 0.0) || !std::isfinite(point->features[column])) {
                throw std::invalid_argument("cost model features must be finite and nonnegative");
            }
            scales[column] = std::max(scales[column], point->features[column]);
        }
    }
    for (std::size_t column = 0; column < columns; ++column) {
        if (scales[column] == 0.0) {
            throw std::invalid_argument("cost model contains an unobservable coefficient");
        }
    }

    double best_error = std::numeric_limits<double>::infinity();
    std::vector<double> best(columns, 0.0);
    const std::uint32_t subset_count = 1U << static_cast<std::uint32_t>(columns);
    for (std::uint32_t mask = 1; mask < subset_count; ++mask) {
        std::array<std::size_t, kMaximumModelCoefficients> active{};
        std::size_t active_count = 0;
        for (std::size_t column = 0; column < columns; ++column) {
            if ((mask & (1U << static_cast<std::uint32_t>(column))) != 0U) {
                active[active_count++] = column;
            }
        }

        std::array<std::array<double, kMaximumModelCoefficients>, kMaximumModelCoefficients>
            normal{};
        std::array<double, kMaximumModelCoefficients> right{};
        for (const RegressionPoint* point : training) {
            const double inverse_time = 1.0 / point->elapsed_ns;
            std::array<double, kMaximumModelCoefficients> row{};
            for (std::size_t local = 0; local < active_count; ++local) {
                const std::size_t column = active[local];
                row[local]               = point->features[column] / scales[column] * inverse_time;
            }
            for (std::size_t left = 0; left < active_count; ++left) {
                right[left] += row[left];
                for (std::size_t right_column = 0; right_column < active_count; ++right_column) {
                    normal[left][right_column] += row[left] * row[right_column];
                }
            }
        }

        std::array<double, kMaximumModelCoefficients> local_solution{};
        if (!solve_linear(normal, right, active_count, local_solution)) { continue; }
        std::vector<double> candidate(columns, 0.0);
        bool feasible = true;
        for (std::size_t local = 0; local < active_count; ++local) {
            if (local_solution[local] < -1.0e-9 || !std::isfinite(local_solution[local])) {
                feasible = false;
                break;
            }
            candidate[active[local]] = std::max(0.0, local_solution[local]) / scales[active[local]];
        }
        if (!feasible) { continue; }

        double error = 0.0;
        for (const RegressionPoint* point : training) {
            const double relative =
                (floating_prediction(*point, candidate) - point->elapsed_ns) / point->elapsed_ns;
            error += relative * relative;
        }
        if (error < best_error) {
            best_error = error;
            best       = std::move(candidate);
        }
    }
    if (!std::isfinite(best_error)) {
        throw std::runtime_error("cost model non-negative fit is rank deficient");
    }
    return best;
}

std::uint64_t quantize(double coefficient, CoefficientEncoding encoding) {
    if (!(coefficient >= 0.0) || !std::isfinite(coefficient)) {
        throw std::invalid_argument("cost coefficient must be finite and nonnegative");
    }
    const long double scale =
        encoding == CoefficientEncoding::Q32Nanoseconds ? static_cast<long double>(kQ32One) : 1.0L;
    const long double value = std::round(static_cast<long double>(coefficient) * scale);
    if (value > static_cast<long double>(std::numeric_limits<std::uint64_t>::max())) {
        throw std::overflow_error("cost coefficient exceeds uint64");
    }
    return static_cast<std::uint64_t>(value);
}

double transfer_prediction(const TransferRegressionPoint& point,
                           const std::array<double, 3>& coefficients) {
    return std::max(coefficients[0] + coefficients[1] * point.copy_operations,
                    coefficients[2] * static_cast<double>(point.payload_bytes));
}

double quantized_transfer_prediction(const TransferRegressionPoint& point,
                                     const std::vector<std::uint64_t>& coefficients) {
    if (coefficients.size() != 3) {
        throw std::invalid_argument("transfer roofline requires three coefficients");
    }
    const long double operation_limited =
        static_cast<long double>(coefficients[0]) +
        static_cast<long double>(coefficients[1]) * point.copy_operations;
    const long double bandwidth_limited = static_cast<long double>(coefficients[2]) *
                                          point.payload_bytes / static_cast<long double>(kQ32One);
    return static_cast<double>(std::max(operation_limited, bandwidth_limited));
}

FitMetrics metrics_for(const std::vector<PointPrediction>& predictions, bool validation,
                       double ordering_significance) {
    std::vector<const PointPrediction*> selected;
    std::vector<double> errors;
    for (const PointPrediction& point : predictions) {
        if (point.validation != validation) { continue; }
        selected.push_back(&point);
        errors.push_back(point.relative_error);
    }
    FitMetrics metrics;
    if (errors.empty()) { return metrics; }
    std::sort(errors.begin(), errors.end());
    metrics.median_relative_error = median_of(errors);
    const std::size_t p95 =
        static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(errors.size()))) - 1U;
    metrics.p95_relative_error     = errors[std::min(p95, errors.size() - 1U)];
    metrics.maximum_relative_error = errors.back();

    for (std::size_t left = 0; left < selected.size(); ++left) {
        for (std::size_t right = left + 1; right < selected.size(); ++right) {
            const PointPrediction* cheaper = selected[left];
            const PointPrediction* dearer  = selected[right];
            if (cheaper->elapsed_ns > dearer->elapsed_ns) { std::swap(cheaper, dearer); }
            if (dearer->elapsed_ns <= cheaper->elapsed_ns * (1.0 + ordering_significance)) {
                continue;
            }
            ++metrics.ordering_pairs;
            if (dearer->predicted_ns < cheaper->predicted_ns) {
                ++metrics.ordering_failures;
            } else if (dearer->predicted_ns == cheaper->predicted_ns) {
                // A compressed feature model may deliberately leave two physically different
                // points unresolved. Runtime sends this exact numerical tie to its deterministic
                // semantic tie-break; only a strict inversion invalidates the fit.
                ++metrics.ordering_ties;
            }
        }
    }
    return metrics;
}

} // namespace

SampleSummary summarize_samples(std::vector<double> elapsed_ns) {
    if (elapsed_ns.empty()) { throw std::invalid_argument("timing sample set must not be empty"); }
    for (const double sample : elapsed_ns) {
        if (!(sample > 0.0) || !std::isfinite(sample)) {
            throw std::invalid_argument("timing samples must be finite and positive");
        }
    }
    SampleSummary out;
    out.minimum_ns = *std::min_element(elapsed_ns.begin(), elapsed_ns.end());
    out.maximum_ns = *std::max_element(elapsed_ns.begin(), elapsed_ns.end());
    out.median_ns  = median_of(elapsed_ns);
    std::vector<double> deviations;
    deviations.reserve(elapsed_ns.size());
    for (const double sample : elapsed_ns) {
        deviations.push_back(std::abs(sample - out.median_ns));
    }
    out.mad_ns     = median_of(std::move(deviations));
    out.elapsed_ns = std::move(elapsed_ns);
    return out;
}

double predict_quantized(const RegressionPoint& point,
                         const std::vector<std::uint64_t>& coefficients,
                         const std::vector<CoefficientEncoding>& encodings) {
    if (coefficients.size() != encodings.size() || coefficients.size() > point.features.size()) {
        throw std::invalid_argument("cost model quantized shape is inconsistent");
    }
    long double result = 0.0L;
    for (std::size_t index = 0; index < coefficients.size(); ++index) {
        const long double scale = encodings[index] == CoefficientEncoding::Q32Nanoseconds
                                      ? static_cast<long double>(kQ32One)
                                      : 1.0L;
        result += static_cast<long double>(coefficients[index]) *
                  static_cast<long double>(point.features[index]) / scale;
    }
    return static_cast<double>(result);
}

FitResult fit_nonnegative_model(const std::vector<RegressionPoint>& points,
                                std::vector<std::string> coefficient_names,
                                std::vector<CoefficientEncoding> encodings, double maximum_p95,
                                double ordering_significance) {
    if (coefficient_names.empty() || coefficient_names.size() > kMaximumModelCoefficients ||
        encodings.size() != coefficient_names.size()) {
        throw std::invalid_argument("cost model coefficient schema is invalid");
    }
    if (!(maximum_p95 >= 0.0) || !(ordering_significance >= 0.0)) {
        throw std::invalid_argument("cost model validation thresholds must be nonnegative");
    }
    bool has_validation = false;
    for (const RegressionPoint& point : points) {
        if (!(point.elapsed_ns > 0.0) || !std::isfinite(point.elapsed_ns)) {
            throw std::invalid_argument("cost model point time must be finite and positive");
        }
        has_validation = has_validation || point.validation;
    }
    if (!has_validation) { throw std::invalid_argument("cost model requires held-out points"); }

    FitResult result;
    result.coefficient_names = std::move(coefficient_names);
    result.encodings         = std::move(encodings);
    result.coefficients      = nonnegative_fit(points, result.coefficient_names.size());
    result.quantized_coefficients.reserve(result.coefficients.size());
    for (std::size_t index = 0; index < result.coefficients.size(); ++index) {
        result.quantized_coefficients.push_back(
            quantize(result.coefficients[index], result.encodings[index]));
    }
    result.predictions.reserve(points.size());
    for (const RegressionPoint& point : points) {
        const double predicted =
            predict_quantized(point, result.quantized_coefficients, result.encodings);
        result.predictions.push_back(PointPrediction{
            .label          = point.label,
            .elapsed_ns     = point.elapsed_ns,
            .predicted_ns   = predicted,
            .relative_error = std::abs(predicted - point.elapsed_ns) / point.elapsed_ns,
            .validation     = point.validation,
        });
    }
    result.training   = metrics_for(result.predictions, false, ordering_significance);
    result.validation = metrics_for(result.predictions, true, ordering_significance);
    result.accepted   = result.training.p95_relative_error <= maximum_p95 &&
                      result.training.ordering_failures == 0 &&
                      result.validation.p95_relative_error <= maximum_p95 &&
                      result.validation.ordering_failures == 0;
    return result;
}

FitResult fit_transfer_roofline(const std::vector<TransferRegressionPoint>& points,
                                double maximum_p95, double ordering_significance) {
    if (!(maximum_p95 >= 0.0) || !(ordering_significance >= 0.0)) {
        throw std::invalid_argument("transfer validation thresholds must be nonnegative");
    }
    std::vector<const TransferRegressionPoint*> training;
    bool has_validation   = false;
    double batch_high     = std::numeric_limits<double>::infinity();
    double operation_high = std::numeric_limits<double>::infinity();
    double bandwidth_high = std::numeric_limits<double>::infinity();
    for (const TransferRegressionPoint& point : points) {
        if (point.payload_bytes == 0 || point.copy_operations == 0 || !(point.elapsed_ns > 0.0) ||
            !std::isfinite(point.elapsed_ns)) {
            throw std::invalid_argument("transfer point has invalid physical work or timing");
        }
        if (point.validation) {
            has_validation = true;
            continue;
        }
        training.push_back(&point);
        batch_high = std::min(batch_high, point.elapsed_ns);
        // Each roofline coefficient is bounded by elapsed/work for every exact point. Use the
        // tightest scale, with headroom below, so small-work cases cannot make the initial grid
        // too coarse to discover the launch-cost basin.
        operation_high = std::min(operation_high, point.elapsed_ns / point.copy_operations);
        bandwidth_high = std::min(bandwidth_high, point.elapsed_ns / point.payload_bytes);
    }
    if (training.size() < 3 || !has_validation) {
        throw std::invalid_argument("transfer roofline requires training and held-out points");
    }
    batch_high *= 1.25;
    operation_high *= 1.25;
    bandwidth_high *= 1.25;

    std::array<double, 3> lower{};
    std::array<double, 3> upper{batch_high, operation_high, bandwidth_high};
    std::array<double, 3> best{};
    double best_error          = std::numeric_limits<double>::infinity();
    constexpr std::size_t grid = 17;
    for (int refinement = 0; refinement < 5; ++refinement) {
        std::array<double, 3> step{};
        for (std::size_t dimension = 0; dimension < step.size(); ++dimension) {
            step[dimension] = (upper[dimension] - lower[dimension]) / (grid - 1U);
        }
        for (std::size_t batch = 0; batch < grid; ++batch) {
            for (std::size_t operation = 0; operation < grid; ++operation) {
                for (std::size_t bandwidth = 0; bandwidth < grid; ++bandwidth) {
                    const std::array<double, 3> candidate{
                        lower[0] + step[0] * batch,
                        lower[1] + step[1] * operation,
                        lower[2] + step[2] * bandwidth,
                    };
                    double error = 0.0;
                    for (const TransferRegressionPoint* point : training) {
                        const double relative =
                            (transfer_prediction(*point, candidate) - point->elapsed_ns) /
                            point->elapsed_ns;
                        error += relative * relative;
                    }
                    if (error < best_error) {
                        best_error = error;
                        best       = candidate;
                    }
                }
            }
        }
        for (std::size_t dimension = 0; dimension < best.size(); ++dimension) {
            lower[dimension] = std::max(0.0, best[dimension] - step[dimension]);
            upper[dimension] = best[dimension] + step[dimension];
        }
    }

    FitResult result;
    result.coefficient_names = {"batch_ns", "operation_ns", "ns_per_byte_q32"};
    result.encodings         = {CoefficientEncoding::IntegerNanoseconds,
                                CoefficientEncoding::IntegerNanoseconds,
                                CoefficientEncoding::Q32Nanoseconds};
    result.coefficients.assign(best.begin(), best.end());
    result.quantized_coefficients = {
        quantize(best[0], result.encodings[0]),
        quantize(best[1], result.encodings[1]),
        quantize(best[2], result.encodings[2]),
    };
    result.predictions.reserve(points.size());
    for (const TransferRegressionPoint& point : points) {
        const double predicted =
            quantized_transfer_prediction(point, result.quantized_coefficients);
        result.predictions.push_back(PointPrediction{
            .label          = point.label,
            .elapsed_ns     = point.elapsed_ns,
            .predicted_ns   = predicted,
            .relative_error = std::abs(predicted - point.elapsed_ns) / point.elapsed_ns,
            .validation     = point.validation,
        });
    }
    result.training   = metrics_for(result.predictions, false, ordering_significance);
    result.validation = metrics_for(result.predictions, true, ordering_significance);
    result.accepted   = result.training.p95_relative_error <= maximum_p95 &&
                      result.training.ordering_failures == 0 &&
                      result.validation.p95_relative_error <= maximum_p95 &&
                      result.validation.ordering_failures == 0;
    return result;
}

std::string_view encoding_name(CoefficientEncoding encoding) noexcept {
    switch (encoding) {
    case CoefficientEncoding::IntegerNanoseconds:
        return "integer_ns";
    case CoefficientEncoding::Q32Nanoseconds:
        return "q32_ns";
    }
    return "unknown";
}

} // namespace ninfer::bench::context_cost
