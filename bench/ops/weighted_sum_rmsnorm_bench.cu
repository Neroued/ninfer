#include "ninfer/ops/weighted_sum_rmsnorm.h"
#include "common/gpu_activity.cuh"
#include "op_measurement.cuh"

using namespace ninfer;
using namespace ninfer::bench;

int main(int argc, char** argv) try {
    int d = 2304, k = 4;
    std::vector<int> rows{1, 2, 3, 4, 5, 6, 7, 8, 1021, 1024, 4093, 4096};
    std::string format = "both";
    measurement::Options timing;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "Weighted RMSNorm: --d D --k K --tokens M[,M...] --alpha bf16|fp32|both\n";
            measurement::help();
            return 0;
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--d")
            d = measurement::integer(value);
        else if (flag == "--k")
            k = measurement::integer(value);
        else if (flag == "--tokens")
            rows = measurement::integers(value);
        else if (flag == "--alpha" && (value == "bf16" || value == "fp32" || value == "both"))
            format = value;
        else if (!measurement::parse(timing, flag, value))
            throw std::invalid_argument("invalid option " + std::string(flag));
    }

    constexpr int condition_us = 200;
    measurement::hardware();
    std::cerr << "GPU activity prelude " << condition_us << " us outside timing\n";
    measurement::LogicalBenchmark benchmark(timing);
    auto weight = make_bf16(d, 103, .5f, 1.5f);
    Tensor w(weight.span(), DType::BF16, {d});
    for (const auto& dtype : measurement::selections(format, "bf16", "fp32")) {
        const bool bf16 = dtype == "bf16";
        auto alphas     = bf16 ? make_bf16(k, 107) : make_f32(k, 107, -.5f, .5f);
        Tensor a(alphas.span(), bf16 ? DType::BF16 : DType::FP32, {k});
        for (int m : rows) {
            const auto count = std::size_t(d) * m;
            auto inputs      = make_bf16(count * k, 101);
            DeviceBuffer output(inputs.device(), count * 2);
            Tensor x(inputs.span(), DType::BF16, {d, k, m});
            Tensor y(output.span(), DType::BF16, {d, m});
            const auto bytes = inputs.size() + output.size() + weight.size() + alphas.size();
            benchmark.run(
                {.op = "weighted-sum-rmsnorm", .format = dtype, .n = d, .k = k, .tokens = m}, bytes,
                [&](CudaExecutionView e) { ops::weighted_sum_rmsnorm(x, a, w, 1.e-5f, y, e); },
                [](CudaExecutionView e) { condition_gpu(e, condition_us); });
        }
    }
    return 0;
} catch (const std::exception& error) {

    std::cerr << "ninfer_weighted_sum_rmsnorm_bench: " << error.what() << '\n';
    return 2;
}
