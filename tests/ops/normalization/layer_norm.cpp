#include "ops/suite.h"
#include "ninfer/ops/layer_norm.h"
#include "cuda_fixture.h"
#include "ops/normalization/support.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kVisionWidth = 1152;
constexpr float kEps                = 1.0e-6F;

constexpr ReductionCriterion layer_norm_bf16_criterion() {
    return {/*relative_l2*/ 1.9e-3, /*gross_absolute*/ 2.0e-4,
            /*gross_relative_to_max_reference*/ 2.7e-3};
}

std::vector<double> layer_norm_oracle(const std::vector<float>& x, const std::vector<float>& weight,
                                      const std::vector<float>& bias, std::int32_t rows) {
    std::vector<double> output(x.size());
    for (std::int32_t row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * kVisionWidth;
        double mean            = 0.0;
        for (std::int32_t column = 0; column < kVisionWidth; ++column) {
            mean += static_cast<double>(x[base + column]);
        }
        mean /= static_cast<double>(kVisionWidth);

        double squared_deviation = 0.0;
        for (std::int32_t column = 0; column < kVisionWidth; ++column) {
            const double centered = static_cast<double>(x[base + column]) - mean;
            squared_deviation += centered * centered;
        }
        const double inverse =
            1.0 / std::sqrt(squared_deviation / static_cast<double>(kVisionWidth) + kEps);
        for (std::int32_t column = 0; column < kVisionWidth; ++column) {
            output[base + column] = (static_cast<double>(x[base + column]) - mean) * inverse *
                                        static_cast<double>(weight[column]) +
                                    static_cast<double>(bias[column]);
        }
    }
    return output;
}

int run_case(const char* label, std::int32_t rows, std::uint32_t seed, bool near_constant = false,
             bool bf16x2_unaligned = false) {
    const std::size_t n = static_cast<std::size_t>(kVisionWidth) * rows;
    std::vector<float> x(n), weight(kVisionWidth), bias(kVisionWidth);
    fill_uniform(x, seed, -4.0F, 4.0F);
    fill_uniform(weight, seed + 1U, -1.5F, 1.5F);
    fill_uniform(bias, seed + 2U, -1.0F, 1.0F);
    if (near_constant) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::size_t base = static_cast<std::size_t>(row) * kVisionWidth;
            for (std::int32_t column = 0; column < kVisionWidth; ++column) {
                x[base + column] = 1.0F + static_cast<float>((column % 5) - 2) / 128.0F;
            }
        }
    }
    round_to_bf16(x);
    round_to_bf16(weight);
    round_to_bf16(bias);
    const std::vector<double> reference = layer_norm_oracle(x, weight, bias, rows);

    norm::DeviceInput device_x      = norm::make_input(x, bf16x2_unaligned);
    norm::DeviceInput device_weight = norm::make_input(weight, bf16x2_unaligned);
    norm::DeviceInput device_bias   = norm::make_input(bias, bf16x2_unaligned);
    const std::size_t leading       = bf16x2_unaligned ? sizeof(std::uint16_t) : 0;
    GuardedDeviceBuffer output(leading + n * sizeof(std::uint16_t));
    output.fill(0xff);
    void* output_data = static_cast<std::uint8_t*>(output.data()) + leading;

    Tensor tx(device_x.memory, DType::BF16, {kVisionWidth, rows});
    Tensor tw(device_weight.memory, DType::BF16, {kVisionWidth});
    Tensor tb(device_bias.memory, DType::BF16, {kVisionWidth});
    Tensor tout(output.span().subspan(leading), DType::BF16, {kVisionWidth, rows});
    ops::layer_norm(tx, tw, tb, kEps, tout, ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    int failures = verify_reduction(label, from_device_bf16(output_data, n), reference,
                                    layer_norm_bf16_criterion());
    failures +=
        norm::verify_output_storage(std::string(label) + " output", output, bf16x2_unaligned);
    failures += norm::verify_preserved(std::string(label) + " preserves x", device_x);
    failures += norm::verify_preserved(std::string(label) + " preserves weight", device_weight);
    failures += norm::verify_preserved(std::string(label) + " preserves bias", device_bias);
    return failures;
}

} // namespace

void add_layer_norm_cases(ninfer::test::OpSuite& suite) {
    suite.add("layer_norm/vision/1152/1",
              [] { return run_case("layer_norm vision [1152,1]", 1, 3101U); },
              {{"execution", "eager"}});
    suite.add("layer_norm/vision/1152/256",
              [] { return run_case("layer_norm vision [1152,256]", 256, 3102U); },
              {{"execution", "eager"}});
    suite.add("layer_norm/vision/1152/4096",
              [] { return run_case("layer_norm vision [1152,4096]", 4096, 3103U); },
              {{"execution", "eager"}});
    suite.add("layer_norm/near-constant/1152/4",
              [] { return run_case("layer_norm near-constant [1152,4]", 4, 3201U, true); },
              {{"execution", "eager"}});
    suite.add("layer_norm/unaligned/1152/7",
              [] { return run_case("layer_norm unaligned [1152,7]", 7, 3202U, false, true); },
              {{"execution", "eager"}});
}
