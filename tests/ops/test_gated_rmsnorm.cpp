#include "ninfer/ops/gated_rmsnorm.h"
#include "core/device.h"
#include "ops/norm_test_common.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::norm;

namespace {

constexpr ReductionCriterion gated_rmsnorm_bf16_criterion() {
    return {/*relative_l2*/ 1.85e-3, /*gross_absolute*/ 4.5e-5,
            /*gross_relative_to_max_reference*/ 2.8e-3};
}

std::vector<double> gated_rmsnorm_oracle(const std::vector<float>& input,
                                         const std::vector<float>& weight,
                                         const std::vector<float>& gate, const Shape& shape) {
    std::vector<double> output(input.size());
    const auto row_count = static_cast<std::int64_t>(shape.rows) * shape.tokens;
    for (std::int64_t row = 0; row < row_count; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * shape.d;
        double sum_squares     = 0.0;
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double value = input[base + column];
            sum_squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum_squares / static_cast<double>(shape.d) + kEps);
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double gate_value = gate[base + column];
            const double silu       = gate_value / (1.0 + std::exp(-gate_value));
            output[base + column]   = static_cast<double>(input[base + column]) * inverse *
                                    static_cast<double>(weight[column]) * silu;
        }
    }
    return output;
}

int run_case(const char* label, const Shape& shape, std::uint32_t seed, float input_scale = 4.0F,
             bool bf16x2_unaligned = false, bool graph = false) {
    const std::size_t count = shape.elements();
    std::vector<float> input(count), weight(shape.d), gate(count);
    fill_uniform(input, seed, -input_scale, input_scale);
    fill_uniform(weight, seed + 1U, 0.25F, 1.75F);
    fill_uniform(gate, seed + 2U, -5.0F, 5.0F);
    gate.front() = 0.0F;
    gate.back() = -0.0F;
    round_to_bf16(input);
    round_to_bf16(weight);
    round_to_bf16(gate);
    std::vector<double> reference = gated_rmsnorm_oracle(input, weight, gate, shape);

    DeviceInput device_input  = make_input(input, bf16x2_unaligned);
    DeviceInput device_weight = make_input(weight, bf16x2_unaligned);
    DeviceInput device_gate   = make_input(gate, bf16x2_unaligned);
    const std::size_t leading = bf16x2_unaligned ? sizeof(std::uint16_t) : 0;
    GuardedDeviceBuffer output(leading + count * sizeof(std::uint16_t));
    output.fill(0xff);
    void* output_data = static_cast<std::uint8_t*>(output.data()) + leading;

    Tensor input_tensor = tensor_for(device_input.data, shape);
    Tensor weight_tensor(device_weight.data, DType::BF16, {shape.d});
    Tensor gate_tensor   = tensor_for(device_gate.data, shape);
    Tensor output_tensor = tensor_for(output_data, shape);
    ops::gated_rmsnorm(input_tensor, weight_tensor, gate_tensor, kEps, output_tensor, nullptr);
    cuda_synchronize();

    if (graph) {
        cudaStream_t stream;
        cudaGraph_t captured;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ops::gated_rmsnorm(input_tensor, weight_tensor, gate_tensor, kEps, output_tensor, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (std::size_t i = 0; i < gate.size(); ++i) {
            gate[i] = -gate[i];
            device_gate.expected[i + leading / sizeof(std::uint16_t)] = f32_to_bf16(gate[i]);
        }
        CUDA_CHECK(cudaMemcpyAsync(device_gate.storage.p, device_gate.expected.data(),
            device_gate.storage.bytes, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
        CUDA_CHECK(cudaStreamDestroy(stream));
        reference = gated_rmsnorm_oracle(input, weight, gate, shape);
    }

    int failures = verify_reduction(label, from_device_bf16(output_data, count), reference,
                                    gated_rmsnorm_bf16_criterion());
    failures += verify_output_storage(std::string(label) + " output", output, bf16x2_unaligned);
    failures += verify_preserved(std::string(label) + " preserves input", device_input);
    failures += verify_preserved(std::string(label) + " preserves weight", device_weight);
    failures += verify_preserved(std::string(label) + " preserves gate", device_gate);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case("gated_rmsnorm [128,48,1]", {128, 48}, 1401U);
    for (int columns : {2, 7, 16, 48, 56, 57, 64, 96, 128}) {
        const std::string label = "gated_rmsnorm target [128,48," + std::to_string(columns) + "]";
        failures += run_case(label.c_str(), {128, 48, columns}, 1420U + columns,
                             4.0F, false, columns == 56 || columns == 57 || columns == 128);
    }
    failures += run_case("gated_rmsnorm [128,32,7]", {128, 32, 7}, 1402U);
    failures += run_case("gated_rmsnorm [128,32,128]", {128, 32, 128}, 1403U);
    failures += run_case("gated_rmsnorm near-zero [128,32]", {128, 32}, 1404U, 1.0e-5F);
    failures += run_case("gated_rmsnorm unaligned [128,48]", {128, 48}, 1405U, 4.0F, true);
    std::cout << (failures ? "FAIL" : "OK") << " gated_rmsnorm\n";
    return failures ? 1 : 0;
}
