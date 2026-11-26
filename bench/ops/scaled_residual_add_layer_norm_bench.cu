#include "ninfer/ops/scaled_residual_add_layer_norm.h"
#include "common/gpu_activity.cuh"
#include "op_measurement.cuh"

#include <optional>

using namespace ninfer;
using namespace ninfer::bench;

int main(int argc, char** argv) try {
    int d = 1024;
    std::vector<int> rows{257, 1021, 1024, 4093, 4096};
    std::string format = "both", mode = "both", alias = "separate";
    measurement::Options timing;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "Scaled residual: --d D --tokens M[,M...] --gamma bf16|fp32|both\n"
                         "                 --op add|norm|both --alias separate|inplace|both\n";
            measurement::help();
            return 0;
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--d")
            d = measurement::integer(value);
        else if (flag == "--tokens")
            rows = measurement::integers(value);
        else if (flag == "--gamma" && (value == "bf16" || value == "fp32" || value == "both"))
            format = value;
        else if (flag == "--op" && (value == "add" || value == "norm" || value == "both"))
            mode = value;
        else if (flag == "--alias" &&
                 (value == "separate" || value == "inplace" || value == "both"))
            alias = value;
        else if (!measurement::parse(timing, flag, value))
            throw std::invalid_argument("invalid option " + std::string(flag));
    }

    constexpr int condition_us = 200;
    measurement::hardware();
    std::cerr << "GPU activity prelude " << condition_us << " us outside timing\n";
    measurement::LogicalBenchmark benchmark(timing);
    auto weight = make_bf16(d, 103, .5f, 1.5f);
    auto bias   = make_bf16(d, 105, -.125f, .125f);
    Tensor w(weight.span(), DType::BF16, {d}), b(bias.span(), DType::BF16, {d});
    for (const auto& dtype : measurement::selections(format, "bf16", "fp32")) {
        const bool bf16 = dtype == "bf16";
        auto gamma      = bf16 ? make_bf16(d, 107) : make_f32(d, 107, -.5f, .5f);
        Tensor g(gamma.span(), bf16 ? DType::BF16 : DType::FP32, {d});
        for (int m : rows)
            for (const auto& placement : measurement::selections(alias, "separate", "inplace")) {
                const auto count = std::size_t(d) * m;
                auto residual    = make_bf16(count, 109);
                auto update      = make_bf16(count, 113);
                DeviceBuffer publication;
                std::optional<SavedBuffer> initial;
                if (placement == "inplace")
                    initial.emplace(residual);
                else
                    publication = DeviceBuffer(residual.device(), residual.size());
                Tensor r(residual.span(), DType::BF16, {d, m});
                Tensor u(update.span(), DType::BF16, {d, m});
                Tensor out(initial ? residual.span() : publication.span(), DType::BF16, {d, m});
                const auto bytes   = 3 * residual.size() + gamma.size();
                const auto restore = [&](CudaExecutionView e) {
                    if (initial) initial->restore(e);
                    condition_gpu(e, condition_us);
                };
                for (const auto& op : measurement::selections(mode, "add", "norm")) {
                    if (op == "add") {
                        benchmark.run(
                            {.op     = "scaled-residual-add",
                             .format = dtype,
                             .policy = placement,
                             .n      = d,
                             .tokens = m},
                            bytes,
                            [&](CudaExecutionView e) { ops::scaled_residual_add(r, u, g, out, e); },
                            restore);
                    } else {
                        DeviceBuffer normalized(residual.device(), residual.size());
                        Tensor y(normalized.span(), DType::BF16, {d, m});
                        benchmark.run(
                            {.op     = "scaled-residual-add-layer-norm",
                             .format = dtype,
                             .policy = placement,
                             .n      = d,
                             .tokens = m},
                            bytes + normalized.size() + weight.size() + bias.size(),
                            [&](CudaExecutionView e) {
                                ops::scaled_residual_add_layer_norm(r, u, g, w, b, 1.e-5f, out, y,
                                                                    e);
                            },
                            restore);
                    }
                }
            }
    }
    return 0;
} catch (const std::exception& error) {

    std::cerr << "ninfer_scaled_residual_add_layer_norm_bench: " << error.what() << '\n';
    return 2;
}
