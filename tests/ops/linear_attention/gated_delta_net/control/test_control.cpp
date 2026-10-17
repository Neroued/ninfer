#include "support.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/decode_graph.h"
#include <algorithm>
#include <cmath>
#include <limits>
using namespace ninfer;
using namespace ninfer::test;
namespace control = ninfer::test::gdn_control;

namespace {

int run(int columns, bool edges, bool captured, CudaExecutionView execution) {
    const auto count = std::size_t(48) * columns;
    std::vector<float> a(count), b(count), a_log(48), dt_bias(48);
    fill_uniform(a, 101, -8.F, 8.F);
    fill_uniform(b, 103, -8.F, 8.F);
    fill_uniform(a_log, 105, -2.F, 1.F);
    fill_uniform(dt_bias, 107, -1.F, 1.F);
    if (edges) {
        constexpr std::array values{-30.F, -15.F, 0.F, 15.F, 30.F};
        for (std::size_t i = 0; i < count; ++i) {
            a[i] = values[i % 5];
            b[i] = values[(i / 5) % 5];
        }
    }
    DeviceBuffer device_a_log(ninfer::test::cuda_environment().context.device(), count * 2);
    DeviceBuffer device_dt_bias(ninfer::test::cuda_environment().context.device(), count * 2);
    DeviceBuffer dl(ninfer::test::cuda_environment().context.device(), 48 * 4);
    DeviceBuffer dd(ninfer::test::cuda_environment().context.device(), 48 * 4);
    GuardedDeviceBuffer g(count * 4), beta(count * 4);
    Tensor a_log_tensor(device_a_log.span(), DType::BF16, {48, columns}),
        tb(device_dt_bias.span(), DType::BF16, {48, columns}), tl(dl.span(), DType::FP32, {48}),
        dt_bias_tensor(dd.span(), DType::FP32, {48});
    ops::GdnControlOutputs outputs{Tensor(g.span(), DType::FP32, {48, columns}),
                                   Tensor(beta.span(), DType::FP32, {48, columns})};
    auto upload = [&] {
        round_to_bf16(a);
        round_to_bf16(b);
        const auto ab = control::bits(a), bb = control::bits(b);
        device_a_log.copy_from_host(ab.data(), count * 2);
        device_dt_bias.copy_from_host(bb.data(), count * 2);
        dl.copy_from_host(a_log.data(), 48 * 4);
        dd.copy_from_host(dt_bias.data(), 48 * 4);
    };
    upload();
    cuda_synchronize();
    const auto launch = [&] {
        ops::gdn_control(a_log_tensor, tb, tl, dt_bias_tensor, outputs, execution);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (captured) {
        launch();
        cuda_synchronize(execution.stream());
        definition.capture(execution.stream(), launch);
        graph.instantiate(definition);
    }
    int failed = 0;
    for (int phase = 0; phase < (captured ? 3 : 1); ++phase) {
        if (phase) {
            for (auto& v : a) v = -v;
            for (auto& v : b) v *= -0.5F;
            for (auto& v : a_log) v += 0.125F;
            for (auto& v : dt_bias) v = -v;
            upload();
        }
        CUDA_CHECK(cudaMemsetAsync(g.data(), 0xff, count * 4, execution.stream()));
        CUDA_CHECK(cudaMemsetAsync(beta.data(), 0xff, count * 4, execution.stream()));
        if (captured)
            graph.launch(execution.stream());
        else
            launch();
        cuda_synchronize(execution.stream());
        std::vector<double> rg(count), rb(count);
        for (std::size_t i = 0; i < count; ++i) {
            rg[i] = -std::exp(double(a_log[i % 48])) *
                    control::softplus(double(a[i]) + double(dt_bias[i % 48]));
            rb[i] = control::sigmoid(double(b[i]));
        }
        const auto actual_g = from_device<float>(g.data(), count),
                   actual_b = from_device<float>(beta.data(), count);
        const auto name =
            "gdn_control M=" + std::to_string(columns) + " phase=" + std::to_string(phase);
        failed += verify_pointwise((name + " g").c_str(),
                                   std::vector<double>(actual_g.begin(), actual_g.end()), rg,
                                   control::kDecay);
        failed += verify_pointwise((name + " beta").c_str(),
                                   std::vector<double>(actual_b.begin(), actual_b.end()), rb,
                                   control::kSigmoid);
        failed += g.verify_guards(name + " g");
        failed += beta.verify_guards(name + " beta");
        failed += verify_exact((name + " a immutable").c_str(),
                               from_device<std::uint16_t>(device_a_log, count), control::bits(a));
        failed += verify_exact((name + " b immutable").c_str(),
                               from_device<std::uint16_t>(device_dt_bias, count), control::bits(b));
        failed +=
            verify_exact((name + " a_log immutable").c_str(), from_device<float>(dl, 48), a_log);
        failed += verify_exact((name + " dt_bias immutable").c_str(), from_device<float>(dd, 48),
                               dt_bias);
    }
    auto reject = [&](auto call) {
        try {
            call();
        } catch (const std::invalid_argument&) { return 0; }
        std::cerr << "control invalid operand accepted\n";
        return 1;
    };
    const auto saved = outputs.g;
    outputs.g        = outputs.beta;
    failed += reject(launch);
    outputs.g          = saved;
    auto saved_a       = a_log_tensor;
    a_log_tensor.ne[0] = 47;
    failed += reject(launch);
    a_log_tensor       = saved_a;
    a_log_tensor.dtype = DType::FP32;
    failed += reject(launch);
    a_log_tensor       = saved_a;
    a_log_tensor.ne[1] = -1;
    failed += reject(launch);
    a_log_tensor      = saved_a;
    a_log_tensor.data = nullptr;
    failed += reject(launch);
    a_log_tensor = saved_a;
    a_log_tensor.nb[0] += 2;
    failed += reject(launch);
    a_log_tensor   = saved_a;
    outputs.g.data = a_log_tensor.data;
    failed += reject(launch);
    outputs.g = saved;
    return failed;
}
} // namespace

void add_control_cases(ninfer::test::OpSuite& suite) {
    for (int columns : {1, 7, 128, 4096}) {
        const bool graph = columns == 1 || columns == 128;
        suite.add(
            "gdn_control/t" + std::to_string(columns),
            [=] { return run(columns, false, graph, cuda_environment().context.execution_view()); },
            {{"execution", graph ? "graph" : "eager"}});
    }
    suite.add("gdn_control/edges",
              [] { return run(17, true, true, cuda_environment().context.execution_view()); },
              {{"execution", "graph"}});
    suite.add("gdn_control/empty", [] {
        auto input  = Tensor::metadata(DType::BF16, {48, 1});
        auto vector = Tensor::metadata(DType::FP32, {48});
        auto output = Tensor::metadata(DType::FP32, {48, 1});
        input.ne[1] = output.ne[1] = 0;
        ops::gdn_control(input, input, vector, vector, {output, output},
                         cuda_environment().context.execution_view());
        return 0;
    });
}
