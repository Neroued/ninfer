#include "ops/suite.h"
#include "ninfer/ops/rmsnorm.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include "ops/normalization/support.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::norm;

namespace {

enum class Gate { Silu, Sigmoid };
enum class Pattern {
    Random,
    ZeroGate,
    ZeroInput,
    GateEdges,
    NegativeGate,
    DirectGain,
    RoundingGate
};

const char* gate_name(Gate gate) {
    return gate == Gate::Silu ? "silu_gated_rmsnorm" : "sigmoid_gated_rmsnorm";
}

constexpr ReductionCriterion silu_gated_rmsnorm_bf16_criterion() {
    // BF16 RNE alone can approach 2^-8 relative error in a small tensor or one element.
    // The same profile-wide bound covers output storage and FP32 arithmetic in every route.
    return {/*relative_l2*/ 3.95e-3, /*gross_absolute*/ 4.5e-5,
            /*gross_relative_to_max_reference*/ 3.95e-3};
}

// BF16 RNE can incur almost 2^-8 relative error. The gross bound includes that output
// representation error and FP32 reduction/rsqrt/nonlinearity error, including gate tails.
constexpr ReductionCriterion sigmoid_gated_rmsnorm_bf16_criterion() {
    return {/*relative_l2*/ 3.95e-3, /*gross_absolute*/ 1.0e-5,
            /*gross_relative_to_max_reference*/ 3.95e-3};
}

struct GatedShape {
    std::int32_t d;
    std::int32_t heads;
    std::int32_t tokens = 1;
    std::int32_t batch  = 1;

    std::size_t elements() const { return static_cast<std::size_t>(d) * heads * tokens * batch; }

    Tensor tensor(MemorySpan memory) const {
        return Tensor(memory, DType::BF16, {d, heads, tokens, batch});
    }
};

struct Case {
    GatedShape shape;
    std::uint32_t seed;
    float input_scale      = 4.0F;
    bool unaligned         = false;
    bool graph             = false;
    Pattern pattern        = Pattern::Random;
    float eps              = kEps;
    bool shared_input_gate = false;
};

void invoke(Gate gate, const Tensor& x, const Tensor& w, const Tensor& z, float eps, Tensor& out,
            CudaExecutionView execution) {
    if (gate == Gate::Silu)
        ops::silu_gated_rmsnorm(x, w, z, eps, out, execution);
    else
        ops::sigmoid_gated_rmsnorm(x, w, z, eps, out, execution);
}

std::vector<double> gated_rmsnorm_oracle(const std::vector<float>& input,
                                         const std::vector<float>& weight,
                                         const std::vector<float>& gate, const GatedShape& shape,
                                         float eps, Gate kind) {
    std::vector<double> output(input.size());
    const std::size_t row_count = input.size() / shape.d;
    for (std::size_t row = 0; row < row_count; ++row) {
        const std::size_t base = row * shape.d;
        double sum_squares     = 0.0;
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double value = input[base + column];
            sum_squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum_squares / shape.d + static_cast<double>(eps));
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double gate_value = gate[base + column];
            const double sigmoid    = 1.0 / (1.0 + std::exp(-gate_value));
            const double activation = kind == Gate::Silu ? gate_value * sigmoid : sigmoid;
            output[base + column]   = static_cast<double>(input[base + column]) * inverse *
                                    static_cast<double>(weight[column]) * activation;
        }
    }
    return output;
}

int run_case(CudaExecutionView execution, Gate kind, const char* description, const Case& c) {
    const std::size_t count = c.shape.elements();
    std::vector<float> input(count), weight(c.shape.d), gate(count);
    fill_uniform(input, c.seed, -c.input_scale, c.input_scale);
    fill_uniform(weight, c.seed + 1U, 0.25F, 1.75F);
    fill_uniform(gate, c.seed + 2U, -5.0F, 5.0F);
    gate.front() = 0.0F;
    gate.back()  = -0.0F;
    if (c.pattern == Pattern::ZeroGate) std::fill(gate.begin(), gate.end(), 0.0F);
    if (c.pattern == Pattern::ZeroInput) std::fill(input.begin(), input.end(), 0.0F);
    if (c.pattern == Pattern::NegativeGate) std::fill(gate.begin(), gate.end(), -12.0F);
    if (c.pattern == Pattern::GateEdges) {
        constexpr std::array edges{-40.0F, -12.0F,  -1.0F, -1.0e-4F, -0.0F,
                                   0.0F,   1.0e-4F, 1.0F,  12.0F,    40.0F};
        for (std::size_t i = 0; i < count; ++i) gate[i] = edges[i % edges.size()];
    }
    if (c.pattern == Pattern::DirectGain) {
        fill_uniform(weight, c.seed + 1U, -1.5F, 1.5F);
        weight.front() = 0.0F;
        weight.back()  = -1.0F;
    }
    if (c.pattern == Pattern::RoundingGate) {
        std::fill(input.begin(), input.end(), 1.0F);
        std::fill(weight.begin(), weight.end(), 1.0F);
        std::fill(gate.begin(), gate.end(), 0.0078125F);
    }
    round_to_bf16(input);
    round_to_bf16(weight);
    round_to_bf16(gate);
    if (c.shared_input_gate) gate = input;

    DeviceInput device_input  = make_input(input, c.unaligned);
    DeviceInput device_weight = make_input(weight, c.unaligned);
    DeviceInput device_gate   = make_input(gate, c.unaligned);
    const std::size_t leading = c.unaligned ? sizeof(std::uint16_t) : 0;
    GuardedDeviceBuffer output(leading + count * sizeof(std::uint16_t));
    void* output_data   = static_cast<std::uint8_t*>(output.data()) + leading;
    Tensor input_tensor = c.shape.tensor(device_input.memory);
    Tensor weight_tensor(device_weight.memory, DType::BF16, {c.shape.d});
    Tensor gate_tensor =
        c.shape.tensor(c.shared_input_gate ? device_input.memory : device_gate.memory);
    Tensor output_tensor = c.shape.tensor(output.span().subspan(leading));
    const auto launch    = [&] {
        invoke(kind, input_tensor, weight_tensor, gate_tensor, c.eps, output_tensor, execution);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    const std::string label = std::string(gate_name(kind)) + " " + description + " [" +
                              std::to_string(c.shape.d) + "," + std::to_string(c.shape.heads) +
                              "," + std::to_string(c.shape.tokens) + "," +
                              std::to_string(c.shape.batch) + "]";
    int failures = 0;
    for (int phase = 0; phase < (c.graph ? 2 : 1); ++phase) {
        if (phase) {
            for (std::size_t i = 0; i < input.size(); ++i)
                input[i] = -input[i] * (i % 3 == 0 ? 0.5F : 1.0F);
            for (auto& value : gate) value = -value;
            for (auto& value : weight) value = -value;
            if (c.shared_input_gate) gate = input;
            const auto update = [&](DeviceInput& device, const std::vector<float>& values) {
                for (std::size_t i = 0; i < values.size(); ++i)
                    device.expected[i + leading / 2] = f32_to_bf16(values[i]);
                CUDA_CHECK(cudaMemcpy(device.storage.data(), device.expected.data(),
                                      device.storage.size(), cudaMemcpyHostToDevice));
            };
            update(device_input, input);
            update(device_weight, weight);
            update(device_gate, gate);
        }
        output.fill(0xff);
        cuda_synchronize();
        if (c.graph) {
            if (phase == 0) {
                definition.capture(execution.stream(), launch);
                graph.instantiate(definition);
            }
            graph.launch(execution.stream());
        } else {
            launch();
        }
        cuda_synchronize(execution.stream());
        const auto reference = gated_rmsnorm_oracle(input, weight, gate, c.shape, c.eps, kind);
        const auto actual    = from_device_bf16(output_data, count);
        const auto criterion = kind == Gate::Silu ? silu_gated_rmsnorm_bf16_criterion()
                                                  : sigmoid_gated_rmsnorm_bf16_criterion();
        const int numerical_failures = verify_reduction(
            (label + " phase=" + std::to_string(phase)).c_str(), actual, reference, criterion);
        failures += numerical_failures;
        failures += verify_output_storage(label + " output", output, c.unaligned);
        failures += verify_preserved(label + " preserves input", device_input);
        failures += verify_preserved(label + " preserves weight", device_weight);
        failures += verify_preserved(label + " preserves gate", device_gate);
    }
    return failures;
}

template <class Exception, class Function>
int rejects(const std::string& label, Function&& call) {
    try {
        call();
    } catch (const Exception&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": unexpected exception: " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected rejection\n";
    return 1;
}

int run_validation(CudaExecutionView execution, Gate kind) {
    DeviceBuffer input(ninfer::test::cuda_environment().context.device(), 128 * 48 * 2);
    DeviceBuffer weight(ninfer::test::cuda_environment().context.device(), 128 * 2);
    DeviceBuffer gate(ninfer::test::cuda_environment().context.device(), 128 * 48 * 2);
    DeviceBuffer output(ninfer::test::cuda_environment().context.device(), 128 * 48 * 2);
    const std::array<Tensor, 4> tensors{
        Tensor(input.span(), DType::BF16, {128, 48}), Tensor(weight.span(), DType::BF16, {128}),
        Tensor(gate.span(), DType::BF16, {128, 48}), Tensor(output.span(), DType::BF16, {128, 48})};
    const std::string name = gate_name(kind);
    const auto call        = [&](std::array<Tensor, 4> t, float eps = kEps) {
        invoke(kind, t[0], t[1], t[2], eps, t[3], execution);
    };
    int failures = 0;
    for (int i = 0; i < 4; ++i) {
        auto t     = tensors;
        t[i].dtype = DType::FP32;
        failures += rejects<std::invalid_argument>(name + " wrong dtype", [&] { call(t); });
        t         = tensors;
        t[i].data = nullptr;
        failures += rejects<std::invalid_argument>(name + " null input/output", [&] { call(t); });
        t          = tensors;
        t[i].nb[0] = 4;
        failures += rejects<std::invalid_argument>(name + " noncontiguous", [&] { call(t); });
        for (int extent : {0, -1}) {
            t          = tensors;
            t[i].ne[0] = extent;
            failures +=
                rejects<std::invalid_argument>(name + " nonpositive extent", [&] { call(t); });
        }
    }
    for (int i : {2, 3}) {
        auto t     = tensors;
        t[i].ne[1] = 47;
        failures += rejects<std::invalid_argument>(name + " shape mismatch", [&] { call(t); });
    }
    auto t     = tensors;
    t[1].ne[1] = 2;
    failures += rejects<std::invalid_argument>(name + " weight rank", [&] { call(t); });
    for (float eps : {0.0F, -1.0F, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()})
        failures +=
            rejects<std::invalid_argument>(name + " invalid eps", [&] { call(tensors, eps); });
    for (const auto device : {Device::cpu(), Device::cuda(execution.device().index() + 1)}) {
        t           = tensors;
        t[0].device = device;
        failures += rejects<std::invalid_argument>(name + " operand device", [&] { call(t); });
    }
    for (int i : {0, 1, 2}) {
        for (std::size_t offset : {std::size_t{0}, std::size_t{2}}) {
            t         = tensors;
            t[3].data = static_cast<std::byte*>(t[i].data) + offset;
            failures += rejects<std::invalid_argument>(name + " output overlap", [&] { call(t); });
        }
    }
    t = tensors;
    for (auto& extent : t[0].ne) extent = std::numeric_limits<std::int32_t>::max();
    failures += rejects<std::overflow_error>(name + " element count overflow", [&] { call(t); });
    t         = tensors;
    t[0].data = reinterpret_cast<void*>(std::numeric_limits<std::uintptr_t>::max() - 1);
    failures += rejects<std::overflow_error>(name + " byte range overflow", [&] { call(t); });
    return failures;
}

} // namespace

void add_gated_rmsnorm_cases(ninfer::test::OpSuite& suite) {
    const auto add = [&](Gate kind, const char* label, Case c) {
        const auto& s = c.shape;
        const auto id = std::string(gate_name(kind)) + "/" + label + "/" + std::to_string(s.d) +
                        "x" + std::to_string(s.heads) + "x" + std::to_string(s.tokens) + "x" +
                        std::to_string(s.batch);
        suite.add(id,
                  [=, label = std::string(label)] {
                      return run_case(cuda_environment().context.execution_view(), kind,
                                      label.c_str(), c);
                  },
                  {{"execution", c.graph ? "graph" : "eager"}});
    };
    for (Gate kind : {Gate::Silu, Gate::Sigmoid}) {
        add(kind, "single column", {{128, 48}, 1401U});
        for (int columns :
             {2, 4, 7, 8, 9, 16, 48, 55, 56, 57, 58, 64, 96, 127, 128, 129, 512, 1024, 2048})
            add(kind, "D128/H48 columns",
                {{128, 48, columns},
                 1420U + columns,
                 4.0F,
                 false,
                 columns == 56 || columns == 57 || columns == 128 || columns == 1024});
        for (int batch = 1; batch <= 8; ++batch)
            for (int width : {1, 7})
                add(kind, "T/B",
                    {{128, 48, width, batch},
                     1500U + width * 8 + batch,
                     4.0F,
                     false,
                     width == 7 && (batch == 1 || batch == 8)});
        for (int columns : {7, 11, 12, 13, 83, 84, 85, 86, 87, 128, 191, 192, 193, 1024})
            add(kind, "D128/H32 columns",
                {{128, 32, columns}, 1402U + columns, 4.0F, false, columns == 86});
        add(kind, "near-zero", {{128, 32}, 1404U, 1.0e-5F});
        add(kind, "BF16x2 unaligned", {{128, 48}, 1405U, 4.0F, true, true});
        for (int d : {1, 63, 64, 127, 129, 192, 256, 511, 512, 2048, 2560, 3072, 3073, 4096, 5120,
                      8192, 8193})
            add(kind, "general D", {{d, 3, 5}, 1600U + d, 4.0F, false, d == 129 || d == 5120});
        for (int delta : {-1, 0, 1})
            suite.add(std::string(gate_name(kind)) + "/wide/sms" + std::to_string(delta), [=] {
                auto& device   = cuda_environment().context;
                const int rows = device.multiprocessor_count() + delta;
                return run_case(device.execution_view(), kind, "wide rows",
                                {{4096, rows}, 1700U + rows});
            });
        add(kind, "zero gate", {{128, 48, 3}, 1801U, 4.0F, false, true, Pattern::ZeroGate});
        add(kind, "zero input", {{128, 48}, 1802U, 4.0F, false, false, Pattern::ZeroInput});
        add(kind, "gate edges", {{128, 48, 7}, 1803U, 4.0F, false, true, Pattern::GateEdges});
        add(kind, "negative gate tail",
            {{128, 48, 4}, 1804U, 4.0F, false, false, Pattern::NegativeGate});
        add(kind, "direct gains", {{128, 48, 4}, 1805U, 4.0F, false, true, Pattern::DirectGain});
        add(kind, "eps-dominated",
            {{128, 48, 7}, 1806U, 1.0e-4F, false, true, Pattern::Random, 1.0e-3F});
        add(kind, "shared read-only backing",
            {{128, 48, 4}, 1807U, 4.0F, false, true, Pattern::Random, kEps, true});
        add(kind, "BF16 rounding edge", {{1, 1}, 1808U, 4.0F, false, false, Pattern::RoundingGate});
        suite.add(std::string(gate_name(kind)) + "/contract", [=] {
            return run_validation(cuda_environment().context.execution_view(), kind);
        });
    }
}
