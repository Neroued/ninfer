#include "ops/suite.h"
#include "ninfer/ops/embedding.h"
#include "cuda_fixture.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// Full-matrix measurements against the direct FP64 ideal peak at 3.891e-3 relative error. One
// strict pointwise criterion covers every output element in both registered cases; there is no
// tail or normwise escape path.
constexpr PointwiseCriterion kVisionPositionEmbeddingTolerance{
    /*absolute=*/1.0e-5,
    /*relative=*/3.95e-3,
};

std::vector<std::uint16_t> as_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

int run_case(std::int32_t patches, std::uint32_t seed) {
    constexpr std::int32_t kDimensions = 1152;
    constexpr std::int32_t kTableRows  = 2304;

    std::vector<float> table_values(static_cast<std::size_t>(kDimensions) * kTableRows);
    std::vector<float> x_values(static_cast<std::size_t>(kDimensions) * patches);
    std::vector<float> weights(static_cast<std::size_t>(4) * patches);
    std::vector<std::int32_t> indices(static_cast<std::size_t>(4) * patches);
    fill_uniform(table_values, seed, -2.0f, 2.0f);
    fill_uniform(x_values, seed + 1u, -8.0f, 8.0f);
    fill_uniform(weights, seed + 2u, 0.05f, 1.0f);
    round_to_bf16(table_values);
    round_to_bf16(x_values);

    for (std::int32_t patch = 0; patch < patches; ++patch) {
        double sum = 0.0;
        for (std::int32_t corner = 0; corner < 4; ++corner) {
            const std::size_t control = static_cast<std::size_t>(patch) * 4 + corner;
            indices[control] =
                static_cast<std::int32_t>((patch * 37 + corner * 101 + 11) % kTableRows);
            sum += static_cast<double>(weights[control]);
        }
        for (std::int32_t corner = 0; corner < 4; ++corner) {
            const std::size_t control = static_cast<std::size_t>(patch) * 4 + corner;
            weights[control] = static_cast<float>(static_cast<double>(weights[control]) / sum);
        }
    }

    std::vector<double> reference(x_values.size());
    for (std::int32_t patch = 0; patch < patches; ++patch) {
        for (std::int32_t dimension = 0; dimension < kDimensions; ++dimension) {
            double position = 0.0;
            for (std::int32_t corner = 0; corner < 4; ++corner) {
                const std::size_t control = static_cast<std::size_t>(patch) * 4 + corner;
                position +=
                    static_cast<double>(
                        table_values[static_cast<std::size_t>(indices[control]) * kDimensions +
                                     dimension]) *
                    static_cast<double>(weights[control]);
            }
            const std::size_t output = static_cast<std::size_t>(patch) * kDimensions + dimension;
            reference[output]        = static_cast<double>(x_values[output]) + position;
        }
    }

    const auto table_bits = as_bf16_bits(table_values);
    const auto x_bits     = as_bf16_bits(x_values);
    GuardedDeviceBuffer device_table(table_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_indices(indices.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_weights(weights.size() * sizeof(float));
    GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
    device_table.copy_from_host(table_bits.data(), table_bits.size() * sizeof(std::uint16_t));
    device_indices.copy_from_host(indices.data(), indices.size() * sizeof(std::int32_t));
    device_weights.copy_from_host(weights.data(), weights.size() * sizeof(float));
    device_x.copy_from_host(x_bits.data(), x_bits.size() * sizeof(std::uint16_t));

    Tensor table_tensor(device_table.span(), DType::BF16, {kDimensions, kTableRows});
    Tensor indices_tensor(device_indices.span(), DType::I32, {4, patches});
    Tensor weights_tensor(device_weights.span(), DType::FP32, {4, patches});
    Tensor x_tensor(device_x.span(), DType::BF16, {kDimensions, patches});
    ops::embedding_interpolate_add(table_tensor, indices_tensor, weights_tensor, x_tensor,
                                   ninfer::test::cuda_environment().default_execution);
    cuda_synchronize();

    const std::string label =
        "embedding_interpolate_add D=1152 R=2304 P=" + std::to_string(patches);
    int failures = verify_pointwise(label.c_str(), from_device_bf16(device_x.data(), x_bits.size()),
                                    reference, kVisionPositionEmbeddingTolerance);
    failures += verify_exact((label + " preserves table").c_str(),
                             from_device<std::uint16_t>(device_table.data(), table_bits.size()),
                             table_bits);
    failures +=
        verify_exact((label + " preserves indices").c_str(),
                     from_device<std::int32_t>(device_indices.data(), indices.size()), indices);
    failures += verify_exact((label + " preserves weights").c_str(),
                             from_device<float>(device_weights.data(), weights.size()), weights);
    failures += device_table.verify_guards((label + " table").c_str());
    failures += device_indices.verify_guards((label + " indices").c_str());
    failures += device_weights.verify_guards((label + " weights").c_str());
    failures += device_x.verify_guards((label + " x").c_str());
    return failures;
}

} // namespace

void add_embedding_interpolate_add_cases(ninfer::test::OpSuite& suite) {
    suite.add("interpolate_add/17", [] { return run_case(17, 1u); }, {{"execution", "eager"}});
    suite.add("interpolate_add/1024", [] { return run_case(1024, 11u); }, {{"execution", "eager"}});
}
