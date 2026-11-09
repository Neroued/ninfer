#include "ops/suite.h"
#include "ninfer/ops/rmsnorm_pack_tail.h"
#include "cuda_fixture.h"
#include "core/decode_graph.h"

#include "ops/normalization/support.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::norm;

constexpr ReductionCriterion kCriterion{/*relative_l2*/ 1.85e-3,
                                        /*gross_absolute*/ 1.0e-5,
                                        /*gross_relative_to_max_reference*/ 3.4e-3};

std::vector<double> oracle(const std::vector<float>& input, const std::vector<float>& weight,
                           std::int32_t width, std::int32_t batch, float epsilon) {
    const int rows = weight.size();
    std::vector<double> output(static_cast<std::size_t>(rows) * (width - 1) * batch);
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t tail = 0; tail < (width - 1); ++tail) {
            const std::int32_t input_row  = b * width + tail + 1;
            const std::int32_t output_row = b * (width - 1) + tail;
            const std::size_t input_base  = static_cast<std::size_t>(input_row) * rows;
            const std::size_t output_base = static_cast<std::size_t>(output_row) * rows;
            double sum_squares            = 0.0;
            for (std::int32_t d = 0; d < rows; ++d) {
                const double value = input[input_base + d];
                sum_squares += value * value;
            }
            const double inverse =
                1.0 / std::sqrt(sum_squares / static_cast<double>(rows) + epsilon);
            for (std::int32_t d = 0; d < rows; ++d) {
                output[output_base + d] = static_cast<double>(input[input_base + d]) * inverse *
                                          static_cast<double>(weight[d]);
            }
        }
    }
    return output;
}

int run_case(std::int32_t width, std::int32_t batch, std::uint32_t seed, bool graph = false,
             int rows = 5120, float epsilon = 1.e-6F, float magnitude = 2.F) {
    const std::size_t input_count  = static_cast<std::size_t>(rows) * width * batch;
    const std::size_t output_count = static_cast<std::size_t>(rows) * (width - 1) * batch;
    std::vector<float> input(input_count);
    std::vector<float> weight(rows);
    fill_uniform(input, seed, -magnitude, magnitude);
    fill_uniform(weight, seed + 1U, 0.25F, 1.75F);
    for (std::int32_t b = 0; b < batch; ++b) {
        const std::size_t anchor_base = static_cast<std::size_t>(b * width) * rows;
        for (std::int32_t d = 0; d < rows; ++d)
            input[anchor_base + d] = std::numeric_limits<float>::quiet_NaN();
    }
    round_to_bf16(input);
    round_to_bf16(weight);

    DeviceInput device_input  = make_input(input, false);
    DeviceInput device_weight = make_input(weight, false);
    GuardedDeviceBuffer output(output_count * sizeof(std::uint16_t));
    output.fill(0xff);

    Tensor input_tensor(device_input.memory, DType::BF16, {rows, width, batch});
    Tensor weight_tensor(device_weight.memory, DType::BF16, {rows});
    Tensor output_tensor(output.span(), DType::BF16, {rows, (width - 1) * batch});
    int failures            = 0;
    const std::string label = "rmsnorm_pack_tail H=" + std::to_string(rows) +
                              " W=" + std::to_string(width) + " B=" + std::to_string(batch);
    const auto check = [&] {
        return verify_reduction(label, from_device_bf16(output.data(), output_count),
                                oracle(input, weight, width, batch, epsilon), kCriterion);
    };
    if (graph) {
        cuda_check(cudaStreamSynchronize(nullptr), "finish fixture initialization");
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "create stream");
        const auto stream_execution =
            ninfer::test::cuda_environment().context.execution_view(stream);
        {
            DecodeGraphDefinition definition;
            DecodeGraphExecutable executable;
            definition.capture(stream, [&] {
                ops::rmsnorm_pack_tail(input_tensor, weight_tensor, epsilon, output_tensor,
                                       stream_execution);
            });
            executable.instantiate(definition);
            for (int replay = 0; replay < 2; ++replay) {
                if (replay) {
                    for (auto& value : input) value = -value;
                    for (std::size_t i = 0; i < input.size(); ++i)
                        device_input.expected[i] = f32_to_bf16(input[i]);
                    cuda_check(cudaMemcpyAsync(device_input.storage.data(),
                                               device_input.expected.data(),
                                               device_input.expected.size() * sizeof(std::uint16_t),
                                               cudaMemcpyHostToDevice, stream),
                               "update input");
                }
                executable.launch(stream);
                cuda_check(cudaStreamSynchronize(stream), "graph replay synchronize");
                failures += check();
            }
        }
        cuda_check(cudaStreamDestroy(stream), "destroy stream");
    } else {
        ops::rmsnorm_pack_tail(input_tensor, weight_tensor, epsilon, output_tensor,
                               ninfer::test::cuda_environment().default_execution);
        cuda_synchronize();
        failures += check();
    }

    failures += output.verify_guards(label + " output guards");
    failures += verify_preserved(label + " preserves input", device_input);
    failures += verify_preserved(label + " preserves weight", device_weight);
    return failures;
}

int contract() {
    const auto e = cuda_environment().default_execution;
    DeviceBuffer storage(e.device(), 32768);
    Tensor x(storage.span().subspan(0, 16384), DType::BF16, {4096, 2}),
        w(storage.span().subspan(16384, 8192), DType::BF16, {4096}),
        y(storage.span().subspan(24576, 8192), DType::BF16, {4096, 1});
    int failures = 0;
    auto reject  = [&](float epsilon) {
        try {
            ops::rmsnorm_pack_tail(x, w, epsilon, y, e);
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    for (float invalid : {0.F, -1.F, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()})
        reject(invalid);
    x.ne[0] = 4100;
    reject(1.e-5F);
    x.ne[0] = 4096;
    y.ne[0] = 5120;
    reject(1.e-5F);
    y.ne[0] = 4096;
    y.data  = x.data;
    reject(1.e-5F);
    return failures;
}

} // namespace

void add_rmsnorm_pack_tail_cases(ninfer::test::OpSuite& suite) {
    for (int width = 2; width <= 16; ++width)
        for (int batch = 1; batch <= 8; ++batch)
            suite.add("rmsnorm_pack_tail/w" + std::to_string(width) + "/b" + std::to_string(batch),
                      [=] { return run_case(width, batch, 1700U + width * 8 + batch); },
                      {{"execution", "eager"}});
    suite.add("rmsnorm_pack_tail/graph/w3/b3", [] { return run_case(3, 3, 1701U, true); },
              {{"execution", "graph"}});
    suite.add("rmsnorm_pack_tail/graph/w16/b8", [] { return run_case(16, 8, 1708U, true); },
              {{"execution", "graph"}});
    for (const auto [width, batch] : {std::pair{2, 1}, {4, 3}, {8, 4}, {8, 5}, {16, 8}}) {
        suite.add("rmsnorm_pack_tail/glm/w" + std::to_string(width) + "/b" + std::to_string(batch),
                  [=] { return run_case(width, batch, 1800U, false, 4096, 1.e-5F); });
    }
    suite.add("rmsnorm_pack_tail/glm/graph",
              [] { return run_case(8, 8, 1801U, true, 4096, 1.e-5F); }, {{"execution", "graph"}});
    suite.add("rmsnorm_pack_tail/glm/epsilon",
              [] { return run_case(2, 1, 1802U, false, 4096, 1.e-5F, 1.e-4F); });
    suite.add("rmsnorm_pack_tail/epsilon",
              [] { return run_case(3, 2, 1803U, false, 5120, 3.e-4F, .01F); });
    suite.add("rmsnorm_pack_tail/contract", contract);
}
