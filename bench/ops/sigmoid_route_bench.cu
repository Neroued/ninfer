#include "ninfer/ops/moe.h"
#include "op_measurement.cuh"
#include "direct_bf16_weight.cuh"
#include "common/roofline.h"

using namespace ninfer;
using namespace ninfer::bench;

namespace {
struct Options {
    measurement::Options timing;
    std::vector<int> tokens{1, 6, 16, 32, 128, 512, 2048, 4096};
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "Corrected sigmoid routing: H=4096, E=288, TopK=8\n"
                         "  --tokens T[,T...] --sweep START:END[:STEP]\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--tokens")
            o.tokens = measurement::integers(value);
        else if (flag == "--sweep")
            o.tokens = measurement::sweep(value);
        else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option");
    }
    return o;
}

void run(const Options& options) {
    constexpr int h = 4096, e = 288, k = 8;
    const int maximum = *std::max_element(options.tokens.begin(), options.tokens.end());
    auto control      = make_direct_bf16_weight(e, h, 2631);
    auto input        = make_bf16(std::size_t(h) * maximum, 2633U);
    auto ids          = make_buffer(std::size_t(k) * maximum * 4);
    auto coefficients = make_buffer(std::size_t(k) * maximum * 4);
    auto bias         = make_buffer(e * 4);
    std::vector<float> host_bias(e);
    for (int j = 0; j < e; ++j)
        host_bias[j] = (int((std::uint32_t(j + 2635) * 2654435761U) % 31) - 15) * .01f;
    bias.copy_from_host(host_bias.data(), e * 4);
    const ops::SigmoidRouteProfile profile{{h, e, k}, 2.5f, 1e-20f};
    const auto capacity = ops::topk_sigmoid_route_workspace_capacity_bytes(profile, 1, maximum);
    auto scratch        = make_buffer(capacity);
    MemoryArena arena(scratch.span());
    L2FlushBuffer flush(options.timing.flush_bytes);
    measurement::Csv csv(options.timing.csv_out,
                         "logits_io_bytes,logical_bytes,logical_gb_s,useful_tflops,bandwidth_"
                         "reference_us,compute_reference_us");
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    for (int t : options.tokens) {
        Tensor x(input.span(), DType::BF16, {h, t}), b(bias.span(), DType::FP32, {e});
        ops::ExpertRoutes routes{Tensor(ids.span(), DType::I32, {k, t}),
                                 Tensor(coefficients.span(), DType::FP32, {k, t})};
        measurement::Row row;
        row.op               = "topk_sigmoid_route";
        row.format           = "bf16";
        row.policy           = "a16";
        row.n                = e;
        row.k                = h;
        row.tokens           = t;
        row.workspace_bytes  = capacity;
        const auto logits_io = 8ULL * e * t;
        const auto bytes =
            control.model_weight_bytes() + e * 4 + 2ULL * h * t + logits_io + 8ULL * k * t;
        const double flops = 2.0 * h * e * t;
        for (const auto& execution :
             measurement::selections(options.timing.execution, "graph", "eager"))
            for (const auto& cache :
                 measurement::selections(options.timing.cache, "cold", "warm")) {
                arena.reset_peak();
                const auto result = measurement::measure(
                    [&](auto cuda) {
                        ops::topk_sigmoid_route(x, profile, control.weight, b, routes, arena, cuda);
                    },
                    [](auto) {}, flush, device_context().execution_view(), options.timing,
                    execution, cache);
                row.workspace_peak = arena.peak_used();
                csv.write(row, result, execution, cache,
                          std::to_string(logits_io) + ',' + std::to_string(bytes) + ',' +
                              format_metric(bytes / (result.gpu.median_us * 1000), 3, "") + ',' +
                              format_metric(flops / (result.gpu.median_us * 1e6), 3, "") + ',' +
                              format_metric(bytes / (kRoofline.memory_gbs * 1000), 3, "") + ',' +
                              format_metric(flops / (kRoofline.bf16_f32_tflops * 1e6), 3, ""));
            }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        measurement::hardware();
        run(options);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
