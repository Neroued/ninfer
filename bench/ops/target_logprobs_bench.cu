#include "ninfer/ops/target_logprobs.h"
#include "op_measurement.cuh"

using namespace ninfer;
using namespace ninfer::bench;

int main(int argc, char** argv) try {
    measurement::Options timing;
    std::vector<int> columns{1, 8, 32, 128, 512};
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "Target logprobs: --columns C[,C...]\n";
            measurement::help();
            return 0;
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--columns")
            columns = measurement::integers(value);
        else if (!measurement::parse(timing, flag, value))
            throw std::invalid_argument("unknown option");
    }
    measurement::hardware();
    measurement::LogicalBenchmark benchmark(timing);
    constexpr int physical = 248320, valid = 248077;
    for (int c : columns) {
        auto logits = make_bf16(std::size_t(physical) * c, 101, -8.f, 8.f);
        std::vector<int> targets(c);
        for (int i = 0; i < c; ++i) targets[i] = (i * 7919) % valid;
        auto ids = measurement::make_i32(targets), out = make_zeros(std::size_t(c) * sizeof(float));
        Tensor l(logits.span(), DType::BF16, {physical, c}), t(ids.span(), DType::I32, {c}),
            y(out.span(), DType::FP32, {c});
        benchmark.run(
            {.op = "target-logprobs", .format = "bf16", .n = physical, .k = valid, .tokens = c},
            std::size_t(c) * (valid * 2 + 8),
            [&](CudaExecutionView cuda) { ops::target_logprobs(l, t, valid, y, cuda); },
            [](CudaExecutionView) {});
    }
    return 0;
} catch (const std::exception& error) {

    std::cerr << error.what() << '\n';
    return 1;
}
