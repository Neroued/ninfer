#include "context_cost_measure.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace cost = ninfer::bench::context_cost;

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

double exact_time(double batch, double runs, double bytes) {
    return 1200.0 * batch + 275.0 * runs + 0.018 * bytes;
}

double transfer_time(std::uint32_t operations, std::uint64_t bytes) {
    return std::max(1200.0 + 275.0 * operations, 0.018 * static_cast<double>(bytes));
}

} // namespace

int main() {
    int failures                      = 0;
    const cost::SampleSummary summary = cost::summarize_samples({90.0, 100.0, 110.0, 100.0, 105.0});
    failures += check(summary.median_ns == 100.0 && summary.mad_ns == 5.0 &&
                          summary.minimum_ns == 90.0 && summary.maximum_ns == 110.0,
                      "sample summary did not compute median/MAD/extrema");

    struct Shape {
        double runs;
        double bytes;
        bool validation;
    };

    const std::vector<Shape> shapes{
        {1, 4096, false},    {4, 65536, false},    {16, 1048576, false},
        {2, 8388608, false}, {8, 33554432, false}, {12, 67108864, false},
        {3, 524288, true},   {7, 4194304, true},   {10, 16777216, true},
    };
    std::vector<cost::RegressionPoint> points;
    for (std::size_t index = 0; index < shapes.size(); ++index) {
        const Shape shape = shapes[index];
        points.push_back(cost::RegressionPoint{
            .label      = "point-" + std::to_string(index),
            .features   = {1.0, shape.runs, shape.bytes, 0.0},
            .elapsed_ns = exact_time(1.0, shape.runs, shape.bytes),
            .validation = shape.validation,
        });
    }
    const cost::FitResult fit = cost::fit_nonnegative_model(
        points, {"batch_ns", "additional_run_ns", "ns_per_byte_q32"},
        {cost::CoefficientEncoding::IntegerNanoseconds,
         cost::CoefficientEncoding::IntegerNanoseconds, cost::CoefficientEncoding::Q32Nanoseconds});
    failures += check(fit.accepted, "exact held-out model was rejected");
    failures +=
        check(fit.quantized_coefficients.size() == 3 && fit.quantized_coefficients[0] == 1200 &&
                  fit.quantized_coefficients[1] == 275,
              "integer coefficients were not recovered");
    failures +=
        check(fit.validation.p95_relative_error < 1.0e-6 && fit.validation.ordering_failures == 0,
              "quantized held-out predictions are inaccurate");

    const std::vector<cost::RegressionPoint> inconsistent_training{
        {.label = "training-fast", .features = {1.0}, .elapsed_ns = 1.0, .validation = false},
        {.label = "training-slow", .features = {1.0}, .elapsed_ns = 100.0, .validation = false},
        {.label = "held-out", .features = {1.0}, .elapsed_ns = 1.0, .validation = true},
    };
    const cost::FitResult inconsistent_fit = cost::fit_nonnegative_model(
        inconsistent_training, {"fixed_ns"}, {cost::CoefficientEncoding::IntegerNanoseconds}, 0.20);
    failures +=
        check(!inconsistent_fit.accepted && inconsistent_fit.training.p95_relative_error > 0.20 &&
                  inconsistent_fit.validation.p95_relative_error <= 0.20,
              "training error was not part of fit acceptance");

    struct TransferShape {
        std::uint32_t operations;
        std::uint64_t bytes;
        bool validation;
    };

    const std::vector<TransferShape> transfer_shapes{
        {1, 4096, false},    {4, 65536, false},     {16, 262144, false},    {64, 1048576, false},
        {4, 8388608, false}, {32, 33554432, false}, {256, 67108864, false}, {8, 524288, true},
        {24, 4194304, true}, {128, 16777216, true}, {16, 50331648, true},
    };
    std::vector<cost::TransferRegressionPoint> transfer_points;
    for (std::size_t index = 0; index < transfer_shapes.size(); ++index) {
        const TransferShape shape = transfer_shapes[index];
        transfer_points.push_back(cost::TransferRegressionPoint{
            .label           = "transfer-" + std::to_string(index),
            .payload_bytes   = shape.bytes,
            .copy_operations = shape.operations,
            .elapsed_ns      = transfer_time(shape.operations, shape.bytes),
            .validation      = shape.validation,
        });
    }
    const cost::FitResult transfer_fit = cost::fit_transfer_roofline(transfer_points);
    failures += check(transfer_fit.accepted, "exact transfer roofline was rejected");
    failures += check(transfer_fit.quantized_coefficients.size() == 3 &&
                          transfer_fit.validation.p95_relative_error < 0.03 &&
                          transfer_fit.validation.ordering_failures == 0,
                      "transfer roofline held-out predictions are inaccurate");

    // The roofline intentionally cannot distinguish these operation-dominated points. Their
    // materially different observations must be reported as an unresolved tie, not an inversion.
    std::vector<cost::TransferRegressionPoint> tie_points = transfer_points;
    tie_points.push_back(cost::TransferRegressionPoint{.label           = "transfer-tie-cheaper",
                                                       .payload_bytes   = 1000,
                                                       .copy_operations = 8,
                                                       .elapsed_ns      = transfer_time(8, 1000),
                                                       .validation      = true});
    tie_points.push_back(cost::TransferRegressionPoint{.label           = "transfer-tie-dearer",
                                                       .payload_bytes   = 2000,
                                                       .copy_operations = 8,
                                                       .elapsed_ns      = 4500.0,
                                                       .validation      = true});
    const cost::FitResult tie_fit = cost::fit_transfer_roofline(tie_points);
    failures += check(tie_fit.accepted && tie_fit.validation.ordering_failures == 0 &&
                          tie_fit.validation.ordering_ties != 0,
                      "transfer roofline ties were treated as ordering failures");

    // A large bandwidth-dominated point must not set the launch-cost search scale. Using the
    // largest elapsed/work ratio makes the operation grid too coarse and collapses this exact
    // three-coefficient roofline to zero operation and bandwidth costs.
    const std::vector<cost::TransferRegressionPoint> separated_scale_points{
        {.label           = "small-o1",
         .payload_bytes   = 1,
         .copy_operations = 1,
         .elapsed_ns      = 1100.0,
         .validation      = false},
        {.label           = "small-o8",
         .payload_bytes   = 1,
         .copy_operations = 8,
         .elapsed_ns      = 1800.0,
         .validation      = false},
        {.label           = "large-o1",
         .payload_bytes   = 1'000'000'000'000ULL,
         .copy_operations = 1,
         .elapsed_ns      = 1'000'000'000.0,
         .validation      = false},
        {.label           = "large-o8",
         .payload_bytes   = 1'000'000'000'000ULL,
         .copy_operations = 8,
         .elapsed_ns      = 1'000'000'000.0,
         .validation      = false},
        {.label           = "held-small",
         .payload_bytes   = 1,
         .copy_operations = 4,
         .elapsed_ns      = 1400.0,
         .validation      = true},
        {.label           = "held-large",
         .payload_bytes   = 500'000'000'000ULL,
         .copy_operations = 4,
         .elapsed_ns      = 500'000'000.0,
         .validation      = true},
    };
    const cost::FitResult separated_scale_fit =
        cost::fit_transfer_roofline(separated_scale_points, 0.01);
    failures += check(separated_scale_fit.accepted &&
                          separated_scale_fit.validation.p95_relative_error < 0.01 &&
                          separated_scale_fit.quantized_coefficients[1] != 0 &&
                          separated_scale_fit.quantized_coefficients[2] != 0,
                      "transfer roofline search lost a coefficient across separated scales");

    return failures == 0 ? 0 : 1;
}
