#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::bench::context_cost {

inline constexpr std::size_t kMaximumModelCoefficients = 5;
inline constexpr std::uint64_t kQ32One                 = 1ULL << 32U;

struct SampleSummary {
    std::vector<double> elapsed_ns;
    double median_ns  = 0.0;
    double mad_ns     = 0.0;
    double minimum_ns = 0.0;
    double maximum_ns = 0.0;
};

[[nodiscard]] SampleSummary summarize_samples(std::vector<double> elapsed_ns);

enum class CoefficientEncoding : std::uint8_t {
    IntegerNanoseconds,
    Q32Nanoseconds,
};

struct RegressionPoint {
    std::string label;
    std::array<double, kMaximumModelCoefficients> features{};
    double elapsed_ns = 0.0;
    bool validation   = false;
};

struct TransferRegressionPoint {
    std::string label;
    std::uint64_t payload_bytes   = 0;
    std::uint32_t copy_operations = 0;
    double elapsed_ns             = 0.0;
    bool validation               = false;
};

struct FitMetrics {
    double median_relative_error    = 0.0;
    double p95_relative_error       = 0.0;
    double maximum_relative_error   = 0.0;
    std::uint32_t ordering_failures = 0;
    std::uint32_t ordering_ties     = 0;
    std::uint32_t ordering_pairs    = 0;
};

struct PointPrediction {
    std::string label;
    double elapsed_ns     = 0.0;
    double predicted_ns   = 0.0;
    double relative_error = 0.0;
    bool validation       = false;
};

struct FitResult {
    std::vector<std::string> coefficient_names;
    std::vector<CoefficientEncoding> encodings;
    std::vector<double> coefficients;
    std::vector<std::uint64_t> quantized_coefficients;
    std::vector<PointPrediction> predictions;
    FitMetrics training;
    FitMetrics validation;
    bool accepted = false;
};

/**
 * Fits a non-negative, fixed-dimension linear model. The objective is relative squared error so
 * small latency points remain relevant beside bandwidth-sized transfers. Quantized coefficients
 * are evaluated again for the returned metrics and acceptance decision.
 */
[[nodiscard]] FitResult fit_nonnegative_model(const std::vector<RegressionPoint>& points,
                                              std::vector<std::string> coefficient_names,
                                              std::vector<CoefficientEncoding> encodings,
                                              double maximum_p95           = 0.15,
                                              double ordering_significance = 0.20);

[[nodiscard]] FitResult fit_transfer_roofline(const std::vector<TransferRegressionPoint>& points,
                                              double maximum_p95           = 0.35,
                                              double ordering_significance = 0.20);

[[nodiscard]] double predict_quantized(const RegressionPoint& point,
                                       const std::vector<std::uint64_t>& coefficients,
                                       const std::vector<CoefficientEncoding>& encodings);

[[nodiscard]] std::string_view encoding_name(CoefficientEncoding encoding) noexcept;

} // namespace ninfer::bench::context_cost
