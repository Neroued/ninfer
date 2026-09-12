#include "ninfer/ops/rmsnorm_pack_tail.h"
#include "core/decode_graph.h"

#include "ops/norm_test_common.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::norm;

constexpr std::int32_t kRows = 5120;
constexpr ReductionCriterion kCriterion{/*relative_l2*/ 1.85e-3,
                                        /*gross_absolute*/ 1.0e-5,
                                        /*gross_relative_to_max_reference*/ 3.4e-3};

std::vector<double> oracle(const std::vector<float>& input, const std::vector<float>& weight,
                           std::int32_t width, std::int32_t batch) {
    std::vector<double> output(static_cast<std::size_t>(kRows) * (width - 1) * batch);
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t tail = 0; tail < (width - 1); ++tail) {
            const std::int32_t input_row  = b * width + tail + 1;
            const std::int32_t output_row = b * (width - 1) + tail;
            const std::size_t input_base  = static_cast<std::size_t>(input_row) * kRows;
            const std::size_t output_base = static_cast<std::size_t>(output_row) * kRows;
            double sum_squares            = 0.0;
            for (std::int32_t d = 0; d < kRows; ++d) {
                const double value = input[input_base + d];
                sum_squares += value * value;
            }
            const double inverse = 1.0 / std::sqrt(sum_squares / static_cast<double>(kRows) + kEps);
            for (std::int32_t d = 0; d < kRows; ++d) {
                output[output_base + d] = static_cast<double>(input[input_base + d]) * inverse *
                                          static_cast<double>(weight[d]);
            }
        }
    }
    return output;
}

int run_case(std::int32_t width, std::int32_t batch, std::uint32_t seed, bool graph = false) {
    const std::size_t input_count  = static_cast<std::size_t>(kRows) * width * batch;
    const std::size_t output_count = static_cast<std::size_t>(kRows) * (width - 1) * batch;
    std::vector<float> input(input_count);
    std::vector<float> weight(kRows);
    fill_uniform(input, seed, -2.0F, 2.0F);
    fill_uniform(weight, seed + 1U, 0.25F, 1.75F);
    for (std::int32_t b = 0; b < batch; ++b) {
        const std::size_t anchor_base = static_cast<std::size_t>(b * width) * kRows;
        for (std::int32_t d = 0; d < kRows; ++d) {
            input[anchor_base + d] =
                16.0F + static_cast<float>(b) + static_cast<float>(d % 31) * (1.0F / 32.0F);
        }
    }
    round_to_bf16(input);
    round_to_bf16(weight);
    const std::vector<double> expected = oracle(input, weight, width, batch);

    DeviceInput device_input  = make_input(input, false);
    DeviceInput device_weight = make_input(weight, false);
    GuardedDeviceBuffer output(output_count * sizeof(std::uint16_t));
    output.fill(0xff);

    Tensor input_tensor(device_input.data, DType::BF16, {kRows, width, batch});
    Tensor weight_tensor(device_weight.data, DType::BF16, {kRows});
    Tensor output_tensor(output.data(), DType::BF16, {kRows, (width - 1) * batch});
    if (graph) {
        cuda_check(cudaStreamSynchronize(nullptr), "finish fixture initialization");
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        {
            DecodeGraphDefinition definition;
            DecodeGraphExecutable executable;
            definition.capture(stream, [&] {
                ops::rmsnorm_pack_tail(input_tensor, weight_tensor, output_tensor, stream);
            });
            executable.instantiate(definition);
            for (int replay = 0; replay < 2; ++replay) {
                executable.launch(stream);
                cuda_check(cudaStreamSynchronize(stream), "graph replay synchronize");
            }
        }
        cuda_check(cudaStreamDestroy(stream), "destroy stream");
    } else {
        ops::rmsnorm_pack_tail(input_tensor, weight_tensor, output_tensor, nullptr);
        cuda_synchronize();
    }

    const std::string label = "rmsnorm_pack_tail W=" + std::to_string(width) +
                              " B=" + std::to_string(batch) + (graph ? " graph" : " eager");
    int failures = verify_reduction(label, from_device_bf16(output.data(), output_count), expected,
                                    kCriterion);
    failures += output.verify_guards(label + " output guards");
    failures += verify_preserved(label + " preserves input", device_input);
    failures += verify_preserved(label + " preserves weight", device_weight);
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        for (int width = 2; width <= 16; ++width)
            for (int batch = 1; batch <= 8; ++batch)
                failures += run_case(width, batch, 1700U + width * 8 + batch);
        failures += run_case(3, 3, 1701U, true);
        failures += run_case(16, 8, 1708U, true);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " rmsnorm_pack_tail\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "rmsnorm_pack_tail: " << error.what() << '\n';
        return 1;
    }
}
