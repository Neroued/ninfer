#include "ops/suite.h"
#include "ninfer/ops/l2norm.h"
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

constexpr float kEps = 1.0e-6F;

constexpr ReductionCriterion l2norm_bf16_criterion() {
    return {/*relative_l2*/ 1.9e-3, /*gross_absolute*/ 2.0e-7,
            /*gross_relative_to_max_reference*/ 3.2e-3};
}

struct Shape {
    std::int32_t d;
    std::int32_t heads;
    std::int32_t tokens;

    std::size_t elements() const {
        return static_cast<std::size_t>(d) * static_cast<std::size_t>(heads) *
               static_cast<std::size_t>(tokens);
    }
};

std::vector<double> l2norm_oracle(const std::vector<float>& x, const Shape& shape) {
    std::vector<double> output(x.size());
    const auto rows = static_cast<std::int64_t>(shape.heads) * shape.tokens;
    for (std::int64_t row = 0; row < rows; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * shape.d;
        double sum_squares     = 0.0;
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double value = x[base + column];
            sum_squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum_squares + kEps);
        for (std::int32_t column = 0; column < shape.d; ++column) {
            output[base + column] = static_cast<double>(x[base + column]) * inverse;
        }
    }
    return output;
}

int run_case(const char* label, const Shape& shape, std::uint32_t seed, float scale,
             bool bf16x2_unaligned = false) {
    const std::size_t n = shape.elements();
    std::vector<float> x(n);
    fill_uniform(x, seed, -scale, scale);
    round_to_bf16(x);
    const std::vector<double> reference = l2norm_oracle(x, shape);

    norm::DeviceInput device_x = norm::make_input(x, bf16x2_unaligned);
    const std::size_t leading  = bf16x2_unaligned ? sizeof(std::uint16_t) : 0;
    GuardedDeviceBuffer output(leading + n * sizeof(std::uint16_t));
    output.fill(0xff);
    void* output_data = static_cast<std::uint8_t*>(output.data()) + leading;

    Tensor tx(device_x.memory, DType::BF16, {shape.d, shape.heads, shape.tokens});
    Tensor tout(output.span().subspan(leading), DType::BF16, {shape.d, shape.heads, shape.tokens});
    ops::l2norm(tx, kEps, tout, ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    int failures = verify_reduction(label, from_device_bf16(output_data, n), reference,
                                    l2norm_bf16_criterion());
    failures +=
        norm::verify_output_storage(std::string(label) + " output", output, bf16x2_unaligned);
    failures += norm::verify_preserved(std::string(label) + " preserves x", device_x);
    return failures;
}

} // namespace

void add_l2norm_cases(ninfer::test::OpSuite& suite) {
    suite.add("l2norm/128/16/1",
              [] { return run_case("l2norm [128,16,1]", {128, 16, 1}, 2101U, 4.0F); },
              {{"execution", "eager"}});
    suite.add("l2norm/128/16/7",
              [] { return run_case("l2norm [128,16,7]", {128, 16, 7}, 2102U, 4.0F); },
              {{"execution", "eager"}});
    suite.add("l2norm/128/16/1024",
              [] { return run_case("l2norm [128,16,1024]", {128, 16, 1024}, 2103U, 4.0F); },
              {{"execution", "eager"}});
    suite.add(
        "l2norm/unaligned/128/16/7",
        [] { return run_case("l2norm unaligned [128,16,7]", {128, 16, 7}, 2201U, 4.0F, true); },
        {{"execution", "eager"}});
    suite.add("l2norm/near-zero/128/16/1",
              [] { return run_case("l2norm near-zero [128,16,1]", {128, 16, 1}, 2202U, 1.0e-7F); },
              {{"execution", "eager"}});
}
