#include "core/device.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion sigmoid_mul_bf16_criterion() {
    return {/*absolute*/ 2.0e-5, /*relative*/ 4.05e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<double> sigmoid_mul_oracle(const std::vector<float>& gate,
                                       const std::vector<float>& x) {
    std::vector<double> expected(gate.size());
    for (std::size_t i = 0; i < gate.size(); ++i) {
        const double g = gate[i];
        expected[i]    = static_cast<double>(x[i]) / (1.0 + std::exp(-g));
    }
    return expected;
}

int run_case(const char* label, std::int32_t rows, std::int32_t columns, std::uint32_t seed, bool graph = false) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), x(count);
    fill_uniform(gate, seed, -12.0f, 12.0f);
    fill_uniform(x, seed + 1, -8.0f, 8.0f);
    round_to_bf16(gate);
    round_to_bf16(x);

    const auto expected  = sigmoid_mul_oracle(gate, x);
    const auto gate_bits = encode_bf16(gate);
    const auto x_bits    = encode_bf16(x);
    GuardedDeviceBuffer device_gate(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_x(count * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_x.copy_from_host(x_bits.data(), device_x.bytes());

    Tensor gate_tensor(device_gate.data(), DType::BF16, {rows, columns});
    Tensor x_tensor(device_x.data(), DType::BF16, {rows, columns});
    ops::sigmoid_mul(gate_tensor, x_tensor, nullptr);
    cuda_synchronize();

    if (graph) {
        cudaStream_t stream;
        cudaGraph_t captured;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        ops::sigmoid_mul(gate_tensor, x_tensor, stream);
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&executable, captured, nullptr, nullptr, 0));
        for (int replay = 0; replay < 2; ++replay) {
            CUDA_CHECK(cudaMemcpyAsync(device_x.data(), x_bits.data(), device_x.bytes(),
                cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    int failures = verify_pointwise(label, from_device_bf16(device_x.data(), count), expected,
                                    sigmoid_mul_bf16_criterion());
    failures += verify_exact("sigmoid_mul gate unchanged",
                             from_device<std::uint16_t>(device_gate.data(), count), gate_bits);
    failures += device_gate.verify_guards("sigmoid_mul gate");
    failures += device_x.verify_guards("sigmoid_mul x");
    return failures;
}

int run_edge_case() {
    std::vector<float> gate{-40.0f, -12.0f, -1.0f, 0.0f, 1.0f, 12.0f, 40.0f};
    std::vector<float> x{7.0f, -6.0f, 5.0f, -4.0f, 3.0f, -2.0f, 1.0f};
    round_to_bf16(gate);
    round_to_bf16(x);

    const auto expected  = sigmoid_mul_oracle(gate, x);
    const auto gate_bits = encode_bf16(gate);
    const auto x_bits    = encode_bf16(x);
    GuardedDeviceBuffer device_gate(gate_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_x.copy_from_host(x_bits.data(), device_x.bytes());

    Tensor gate_tensor(device_gate.data(), DType::BF16, {static_cast<int>(gate.size())});
    Tensor x_tensor(device_x.data(), DType::BF16, {static_cast<int>(x.size())});
    ops::sigmoid_mul(gate_tensor, x_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise("sigmoid_mul edge values",
                                    from_device_bf16(device_x.data(), expected.size()), expected,
                                    sigmoid_mul_bf16_criterion());
    failures +=
        verify_exact("sigmoid_mul edge gate unchanged",
                     from_device<std::uint16_t>(device_gate.data(), gate_bits.size()), gate_bits);
    failures += device_gate.verify_guards("sigmoid_mul edge gate");
    failures += device_x.verify_guards("sigmoid_mul edge x");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case("sigmoid_mul [6144,1]", 6144, 1, 101u);
    for (int columns : {2, 8, 16, 48, 64, 128})
        failures += run_case("sigmoid_mul target", 6144, columns, 102u + columns, columns == 128);
    failures += run_case("sigmoid_mul [4096,17]", 4096, 17, 201u);
    failures += run_case("sigmoid_mul [4096,128]", 4096, 128, 301u);
    failures += run_edge_case();
    std::cout << (failures ? "FAIL" : "OK") << " sigmoid_mul\n";
    return failures ? 1 : 0;
}
