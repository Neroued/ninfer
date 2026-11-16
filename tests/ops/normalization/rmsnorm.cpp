#include "ops/suite.h"
#include "ninfer/ops/rmsnorm.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "ops/normalization/support.h"
#include "core/device.h"
#include "core/decode_graph.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::norm;

namespace {

// BF16 RNE alone can incur almost 2^-8 relative error; the gross bound must
// contain that storage error as well as FP32 reduction/rsqrt error.
constexpr ReductionCriterion rmsnorm_bf16_criterion() {
    return {/*relative_l2*/ 1.85e-3, /*gross_absolute*/ 1.0e-5,
            /*gross_relative_to_max_reference*/ 3.95e-3};
}

constexpr ReductionCriterion rmsnorm_inverse_criterion() {
    return {/*relative_l2*/ 2.0e-6, /*gross_absolute*/ 1.0e-7,
            /*gross_relative_to_max_reference*/ 4.0e-6};
}

std::vector<double> rmsnorm_oracle(const std::vector<float>& input,
                                   const std::vector<float>& weight, const Shape& shape,
                                   bool unit_offset, float eps,
                                   std::vector<double>* inverse_output = nullptr) {
    std::vector<double> output(input.size());
    const auto row_count = static_cast<std::int64_t>(shape.rows) * shape.tokens;
    if (inverse_output) inverse_output->resize(row_count);
    for (std::int64_t row = 0; row < row_count; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * shape.d;
        double sum_squares     = 0.0;
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double value = input[base + column];
            sum_squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum_squares / static_cast<double>(shape.d) + eps);
        if (inverse_output) (*inverse_output)[row] = inverse;
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double gain     = static_cast<double>(weight[column]) + (unit_offset ? 1.0 : 0.0);
            output[base + column] = static_cast<double>(input[base + column]) * inverse * gain;
        }
    }
    return output;
}

int run_case(const char* label, const Shape& shape, bool unit_offset, std::uint32_t seed,
             float input_scale = 4.0F, std::size_t leading_words = 0, bool replay = false,
             int width = 0, bool heterogeneous_quarters = false, float eps = kEps,
             bool with_inverse = false) {
    const std::size_t count = shape.elements();
    std::vector<float> input(count), weight(shape.d);
    fill_uniform(input, seed, -input_scale, input_scale);
    if (heterogeneous_quarters) {
        // One RMS is taken across the complete residual, whose four streams need not
        // have the same energy. Independent stream normalization changes this result.
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] *= std::ldexp(1.F, 2 * int((i % shape.d) / (shape.d / 4)) - 3);
    }
    fill_uniform(weight, seed + 1U, -1.5F, 1.5F);
    round_to_bf16(input);
    round_to_bf16(weight);
    weight[0] =
        unit_offset ? -1.0F : 0.0F; // Exact zero gain, without an intermediate BF16 gain cast.

    DeviceInput device_input  = make_input(input, leading_words);
    DeviceInput device_weight = make_input(weight, leading_words);
    const std::size_t leading = leading_words * sizeof(std::uint16_t);
    GuardedDeviceBuffer output(leading + count * sizeof(std::uint16_t));
    output.fill(0xff);
    void* output_data = static_cast<std::uint8_t*>(output.data()) + leading;

    Tensor input_tensor = tensor_for(device_input.memory, shape);
    Tensor weight_tensor(device_weight.memory, DType::BF16, {shape.d});
    Tensor output_tensor = tensor_for(output.span().subspan(leading), shape);
    if (width) {
        input_tensor  = Tensor(device_input.memory, DType::BF16,
                               {shape.d, shape.rows, width, shape.tokens / width});
        output_tensor = Tensor(output.span().subspan(leading), DType::BF16,
                               {shape.d, shape.rows, width, shape.tokens / width});
    }
    const auto rows = static_cast<std::size_t>(shape.rows) * shape.tokens;
    std::optional<GuardedDeviceBuffer> inverse;
    Tensor inverse_tensor;
    if (with_inverse) {
        // Only natural FP32 alignment is required, including on vectorized BF16 routes.
        inverse.emplace((rows + 1) * sizeof(float));
        inverse_tensor = Tensor(inverse->span().subspan(sizeof(float)), DType::FP32,
                                {input_tensor.ne[1], input_tensor.ne[2], input_tensor.ne[3]});
    }
    DeviceContext device;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    const auto launch = [&] {
        if (with_inverse)
            ops::rmsnorm(input_tensor, weight_tensor, eps, unit_offset, output_tensor,
                         inverse_tensor, device.execution_view());
        else
            ops::rmsnorm(input_tensor, weight_tensor, eps, unit_offset, output_tensor,
                         device.execution_view());
    };
    int failures = 0;
    for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
        if (phase) {
            for (auto& value : input) value *= with_inverse ? -0.5F : -1.0F;
            for (auto& value : weight) value *= 0.5F;
            const std::size_t offset = leading_words;
            for (std::size_t i = 0; i < input.size(); ++i)
                device_input.expected[i + offset] = f32_to_bf16(input[i]);
            for (std::size_t i = 0; i < weight.size(); ++i)
                device_weight.expected[i + offset] = f32_to_bf16(weight[i]);
            CUDA_CHECK(cudaMemcpy(device_input.storage.data(), device_input.expected.data(),
                                  device_input.expected.size() * 2, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(device_weight.storage.data(), device_weight.expected.data(),
                                  device_weight.expected.size() * 2, cudaMemcpyHostToDevice));
        }
        output.fill(0xff);
        if (inverse) inverse->fill(0xff);
        cuda_synchronize();
        if (replay && phase == 0) {
            definition.capture(device.stream(), launch);
            graph.instantiate(definition);
        }
        if (replay)
            graph.launch(device.stream());
        else
            launch();
        cuda_synchronize(device.stream());
        std::vector<double> inverse_reference;
        const auto reference = rmsnorm_oracle(input, weight, shape, unit_offset, eps,
                                              with_inverse ? &inverse_reference : nullptr);
        failures += verify_reduction(label, from_device_bf16(output_data, count), reference,
                                     rmsnorm_bf16_criterion());
        failures += verify_output_storage(std::string(label) + " output", output, leading_words);
        if (inverse) {
            const auto values = from_device<float>(inverse_tensor.data, rows);
            failures += verify_reduction(std::string(label) + " inverse",
                                         std::vector<double>(values.begin(), values.end()),
                                         inverse_reference, rmsnorm_inverse_criterion());
            failures += verify_exact((std::string(label) + " inverse prefix").c_str(),
                                     from_device<std::uint32_t>(inverse->data(), 1),
                                     std::vector<std::uint32_t>{0xffffffffU});
            failures += inverse->verify_guards(std::string(label) + " inverse");
        }
        failures += verify_preserved(std::string(label) + " preserves input", device_input);
        failures += verify_preserved(std::string(label) + " preserves weight", device_weight);
    }
    return failures;
}

} // namespace

int validate_contract() {
    DeviceBuffer input(ninfer::test::cuda_environment().context.device(), 128 * 4);
    DeviceBuffer weight(ninfer::test::cuda_environment().context.device(), 128 * 2);
    DeviceBuffer output(ninfer::test::cuda_environment().context.device(), 128 * 4);
    Tensor x(input.span(), DType::BF16, {128, 2}), w(weight.span(), DType::BF16, {128}),
        y(output.span(), DType::BF16, {128, 2});
    int failed        = 0;
    const auto reject = [&](const Tensor& a, const Tensor& b, Tensor out) {
        try {
            ops::rmsnorm(a, b, 1.0e-6F, false, out,
                         ninfer::test::cuda_environment().default_execution);
        } catch (const std::invalid_argument&) { return 0; }
        std::cerr << "rmsnorm overlap was accepted\n";
        return 1;
    };
    auto alias = y;
    alias.data = x.data;
    failed += reject(x, w, alias);
    alias.data = static_cast<std::byte*>(x.data) + 2;
    failed += reject(x, w, alias);
    alias.data = w.data;
    failed += reject(x, w, alias);
    auto tied = w;
    tied.data = x.data;
    failed += reject(x, tied, y);
    auto empty        = Tensor::metadata(DType::BF16, {128, 1});
    auto empty_weight = Tensor::metadata(DType::BF16, {128});
    empty.ne[1]       = 0;
    ops::rmsnorm(empty, empty_weight, 1.0e-6F, false, empty,
                 ninfer::test::cuda_environment().default_execution);
    return failed;
}

int validate_inverse_contract() {
    DeviceContext device;
    const auto execution = device.execution_view();
    DeviceBuffer input(execution.device(), 128 * 4), weight(execution.device(), 128 * 2),
        output(execution.device(), 128 * 4);
    GuardedDeviceBuffer inverse_storage(6 * sizeof(float));
    Tensor x(input.span(), DType::BF16, {128, 2}), w(weight.span(), DType::BF16, {128}),
        y(output.span(), DType::BF16, {128, 2}), inverse(inverse_storage.span(), DType::FP32, {2});
    const auto reject = [&](Tensor out, Tensor inv, const char* label) {
        try {
            ops::rmsnorm(x, w, kEps, false, out, inv, execution);
        } catch (const std::invalid_argument&) { return 0; }
        std::cerr << "rmsnorm inverse accepted " << label << '\n';
        return 1;
    };
    int failed = 0;
    for (void* pointer :
         {x.data, w.data, y.data, static_cast<void*>(static_cast<std::byte*>(x.data) + 4)}) {
        auto alias = inverse;
        alias.data = pointer;
        failed += reject(y, alias, "overlapping inverse");
    }
    failed += reject(x, inverse, "overlapping output");
    auto invalid  = inverse;
    invalid.dtype = DType::BF16;
    failed += reject(y, invalid, "BF16 inverse");
    invalid       = inverse;
    invalid.ne[0] = 1;
    invalid.ne[1] = 2;
    failed += reject(y, invalid, "reshaped inverse");
    invalid = inverse;
    invalid.nb[0] *= 2;
    failed += reject(y, invalid, "strided inverse");
    invalid      = inverse;
    invalid.data = static_cast<std::byte*>(inverse.data) + 2;
    failed += reject(y, invalid, "unaligned inverse");
    invalid      = inverse;
    invalid.data = nullptr;
    failed += reject(y, invalid, "null inverse");
    invalid        = inverse;
    invalid.device = Device::cpu();
    failed += reject(y, invalid, "inverse on another device");

    auto empty          = Tensor::metadata(DType::BF16, {128, 2, 1});
    empty.ne[2]         = 0;
    auto empty_inverse  = Tensor::metadata(DType::FP32, {2, 1});
    empty_inverse.ne[1] = 0;
    ops::rmsnorm(empty, w, kEps, false, empty, empty_inverse, execution);
    auto wrong_empty_inverse  = empty_inverse;
    wrong_empty_inverse.ne[0] = 1;
    try {
        ops::rmsnorm(empty, w, kEps, false, empty, wrong_empty_inverse, execution);
        std::cerr << "rmsnorm inverse accepted incorrect empty shape\n";
        ++failed;
    } catch (const std::invalid_argument&) {}
    wrong_empty_inverse       = empty_inverse;
    wrong_empty_inverse.dtype = DType::BF16;
    try {
        ops::rmsnorm(empty, w, kEps, false, empty, wrong_empty_inverse, execution);
        std::cerr << "rmsnorm inverse accepted incorrect empty dtype\n";
        ++failed;
    } catch (const std::invalid_argument&) {}

    auto no_features  = Tensor::metadata(DType::BF16, {1, 2, 3});
    no_features.ne[0] = 0;
    no_features.nb[1] = no_features.nb[2] = no_features.nb[3] = 0;
    auto no_weight                                            = Tensor::metadata(DType::BF16, {1});
    no_weight.ne[0]                                           = 0;
    no_weight.nb[1] = no_weight.nb[2] = no_weight.nb[3] = 0;
    Tensor untouched_inverse(inverse_storage.span(), DType::FP32, {2, 3});
    inverse_storage.fill(0x5a);
    ops::rmsnorm(no_features, no_weight, kEps, true, no_features, untouched_inverse, execution);
    cuda_synchronize();
    cudaGraph_t empty_graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(device.stream(), cudaStreamCaptureModeThreadLocal));
    ops::rmsnorm(empty, w, kEps, false, empty, empty_inverse, execution);
    ops::rmsnorm(no_features, no_weight, kEps, true, no_features, untouched_inverse, execution);
    CUDA_CHECK(cudaStreamEndCapture(device.stream(), &empty_graph));
    std::size_t nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(empty_graph, nullptr, &nodes));
    CUDA_CHECK(cudaGraphDestroy(empty_graph));
    if (nodes != 0) {
        std::cerr << "rmsnorm empty inverse call submitted device work\n";
        ++failed;
    }
    failed += verify_exact("rmsnorm D=0 preserves inverse",
                           from_device<std::uint32_t>(inverse_storage.data(), 6),
                           std::vector<std::uint32_t>(6, 0x5a5a5a5aU));
    failed += inverse_storage.verify_guards("rmsnorm inverse empty");
    return failed;
}

void add_rmsnorm_cases(ninfer::test::OpSuite& suite) {
    suite.add("rmsnorm/contract", validate_contract);
    suite.add("rmsnorm/inverse/contract", validate_inverse_contract, {{"suite", "concurrency"}});
    const auto add_inverse = [&](const Shape& shape, bool offset, bool graph = false, int width = 0,
                                 float scale = 4.F) {
        const auto id = "rmsnorm/inverse/d" + std::to_string(shape.d) + "/r" +
                        std::to_string(shape.rows) + "/t" + std::to_string(shape.tokens) +
                        "/scale" + std::to_string(scale);
        suite.add(id,
                  [=] {
                      return run_case("RMSNorm with inverse", shape, offset, 2501U, scale, 0, graph,
                                      width, false, kEps, true);
                  },
                  {{"execution", graph ? "graph" : "eager"}, {"suite", "concurrency"}});
    };
    add_inverse({131, 3, 2}, false);
    add_inverse({128, 1, 7}, false);
    add_inverse({128, 3, 6}, true, true, 3);
    add_inverse({2048, 1, 3}, false);
    add_inverse({5120, 1, 7}, false);
    add_inverse({4096, 1, 289}, false, true);
    add_inverse({20480, 1, 3}, true);
    add_inverse({20480, 1, 171}, false);
    add_inverse({131, 1, 2}, true, false, 0, 0.F);
    for (int d : {1024, 4096}) {
        for (int rows : {1, 17, 128, 288, 289, 4097})
            suite.add("rmsnorm/glm/d" + std::to_string(d) + "/m" + std::to_string(rows), [=] {
                return run_case("GLM RMSNorm", {d, 1, rows}, false, 2401U, 4.F, 0,
                                rows == 17 || rows == 289, 0, false, 1.e-5F);
            });
        for (int prefix : {1, 2, 4})
            suite.add("rmsnorm/glm/d" + std::to_string(d) + "/offset" + std::to_string(prefix * 2),
                      [=] {
                          return run_case("GLM RMSNorm alignment", {d, 1, 300}, false, 2402U, 4.F,
                                          prefix, true, 0, false, 1.e-5F);
                      });
        for (float scale : {0.F, 1.e-5F, 4096.F})
            suite.add("rmsnorm/glm/d" + std::to_string(d) + "/scale" + std::to_string(scale), [=] {
                return run_case("GLM RMSNorm range", {d, 1, 3}, false, 2403U, scale, 0, false, 0,
                                false, 1.e-5F);
            });
    }
    const auto add = [&](const char* label, const Shape& shape, bool offset, std::uint32_t seed,
                         float scale = 4.0F, std::size_t unaligned = 0, bool graph = false,
                         int width = 0, bool heterogeneous_quarters = false) {
        const auto id = "rmsnorm/" + std::string(label) + "/seed" + std::to_string(seed) +
                        "/scale" + std::to_string(scale) + (offset ? "/offset" : "/plain") +
                        "/width" + std::to_string(width) + "/" + std::to_string(shape.d) + "x" +
                        std::to_string(shape.rows) + "x" + std::to_string(shape.tokens);
        suite.add(id,
                  [=, label = std::string(label)] {
                      return run_case(label.c_str(), shape, offset, seed, scale, unaligned, graph,
                                      width, heterogeneous_quarters);
                  },
                  {{"execution", graph ? "graph" : "eager"}});
    };

    add("rmsnorm offset [5120,1]", {5120, 1}, true, 1101U);
    add("rmsnorm offset [5120,128]", {5120, 128}, true, 1102U);
    add("rmsnorm offset [5120,1024]", {5120, 1024}, true, 1107U);
    add("rmsnorm offset [2048,7]", {2048, 7}, true, 1103U);
    add("rmsnorm offset [256,24,7]", {256, 24, 7}, true, 1104U);
    add("rmsnorm offset [256,2,1]", {256, 2}, true, 1105U);
    add("rmsnorm offset [256,4,48]", {256, 4, 48}, true, 1106U);
    add("rmsnorm plain [2048,1]", {2048, 1}, false, 1201U);
    add("rmsnorm plain [2048,128]", {2048, 128}, false, 1202U);
    add("rmsnorm plain [128,32,7]", {128, 32, 7}, false, 1203U);
    add("rmsnorm plain [128,8,128]", {128, 8, 128}, false, 1204U);
    add("rmsnorm offset unaligned [128,32]", {128, 32}, true, 1301U, 4.0F, true);
    add("rmsnorm plain unaligned [128,8]", {128, 8}, false, 1302U, 4.0F, true);
    add("rmsnorm plain near-zero [128,32]", {128, 32}, false, 1303U, 1.0e-5F);
    for (int t : {1, 2, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 96, 127, 128}) {
        const bool graph = t == 1 || t == 2 || t == 7 || t == 8 || t == 15 || t == 16 || t == 31 ||
                           t == 32 || t == 63 || t == 64 || t == 65 || t == 96 || t == 127 ||
                           t == 128;
        for (bool offset : {false, true}) {
            const auto label =
                "rmsnorm hidden5120 T=" + std::to_string(t) + (offset ? " offset" : " plain");
            add(label.c_str(), {5120, 1, t}, offset, 1400U + t, 4.f, false, graph);
        }
        for (int heads : {4, 24}) {
            const auto label =
                "rmsnorm QK256 H=" + std::to_string(heads) + " T=" + std::to_string(t);
            add(label.c_str(), {256, heads, t}, true, 1600U + t, 4.f, false, graph);
        }
    }
    for (int t : {129, 256, 1024, 2048})
        for (bool offset : {false, true})
            add("rmsnorm hidden prefill", {5120, 1, t}, offset, 1700U + t, 4.f, false, true);
    for (int t : {129, 256, 1024, 2048})
        for (int heads : {4, 24})
            add("rmsnorm QK prefill", {256, heads, t}, true, 1750U + t, 4.f, false, true);
    for (int width : {2, 3, 8, 9, 15, 16})
        for (int batch : {1, 8}) {
            for (bool offset : {false, true})
                add("rmsnorm hidden W/B", {5120, 1, width * batch}, offset, 1800U + width, 4.f,
                    false, true, width);
            for (int heads : {4, 24})
                add("rmsnorm QK W/B", {256, heads, width * batch}, true, 1900U + width, 4.f, false,
                    true, width);
        }
    for (float scale : {0.0f, 1.e-5f, 4096.f})
        for (bool offset : {false, true}) {
            add("rmsnorm hidden scale/unaligned", {5120, 1, 17}, offset, 2001U, scale, true, true);
            add("rmsnorm QK scale/unaligned", {256, 4, 17}, offset, 2002U, scale, true, true);
        }
    for (int rows : {1, 8, 128, 1024}) {
        const auto label = "rmsnorm offset2560 rows=" + std::to_string(rows);
        add(label.c_str(), {2560, 1, rows}, true, 2300U + rows, 4.f, false, rows != 1);
    }
    for (int rows : {1, 8, 128, 1024, 4096}) {
        const auto label  = "rmsnorm offset10240 rows=" + std::to_string(rows);
        const bool replay = rows == 8 || rows == 128 || rows == 1024;
        add(label.c_str(), {10240, 1, rows}, true, 2100U + rows, 4.f, false, replay,
            rows == 128 ? 16 : 0);
    }
    for (float scale : {0.0f, 1.e-5f, 4096.f})
        add("rmsnorm offset10240 scale", {10240, 1, 3}, true, 2201U, scale);
    add("rmsnorm offset10240 unaligned", {10240, 1, 7}, true, 2202U, 4.f, true, true);
    add("rmsnorm plain10240", {10240, 1, 3}, false, 2203U);
    // Whole-vector MTP residual normalization; both public gain conventions share
    // the same width. Keep shape/stress coverage compact instead of sweeping T.
    add("whole20480 offset", {20480, 1, 1}, true, 2401U, 4.F, false, false, 0, true);
    add("whole20480 offset graph W/B", {20480, 1, 8}, true, 2402U, 4.F, 4, true, 4, true);
    add("whole20480 offset rows171", {20480, 1, 171}, true, 2403U, 4.F, false, true, 0, true);
    add("whole20480 plain rows257", {20480, 1, 257}, false, 2404U, 4.F, false, true, 0, true);
    add("whole20480 unaligned", {20480, 1, 3}, true, 2405U, 4.F, true, true, 0, true);
    for (float scale : {0.F, 1.e-5F, 4096.F})
        add("whole20480 scale", {20480, 1, 3}, true, 2406U, scale, false, false, 0, true);
}
