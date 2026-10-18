#include "ninfer/ops/linear_swiglu.h"
#include "op_measurement.cuh"
#include "quantized_weight.cuh"
#include "direct_bf16_weight.cuh"
#include "common/roofline.h"

using namespace ninfer;
using namespace ninfer::bench;

namespace {
struct Options {
    measurement::Options timing;
    std::vector<int> tokens{1, 6, 16, 32, 128, 512, 2048, 4096};
    std::string format = "nvfp4";
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "Clamped Linear SwiGLU benchmark\n"
                         "  --format nvfp4|bf16 --tokens T[,T...] --sweep START:END[:STEP]\n"
                         "  NVFP4: N=12288, K=4096, AllowA4; BF16: N=2048, K=4096, A16Only.\n"
                         "  Clamp limit=10. Operand traffic excludes private staging.\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--format")
            o.format = value;
        else if (flag == "--tokens")
            o.tokens = measurement::integers(value);
        else if (flag == "--sweep")
            o.tokens = measurement::sweep(value);
        else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option");
    }
    if (o.format != "bf16" && o.format != "nvfp4")
        throw std::invalid_argument("unknown weight format");
    return o;
}

template <class Fixture>
void run(const Options& options, Fixture& fixture) {
    const auto& weight = fixture.weight;
    const auto policy =
        options.format == "bf16" ? ops::LinearPolicy::A16Only : ops::LinearPolicy::AllowA4;
    const double peak =
        options.format == "bf16" ? kRoofline.bf16_f32_tflops : kRoofline.nvfp4_f32_tflops;
    const int maximum   = *std::max_element(options.tokens.begin(), options.tokens.end());
    const auto capacity = ops::linear_swiglu_workspace_capacity_bytes(weight.qtype, weight.n,
                                                                      weight.k, policy, 1, maximum);
    auto input          = make_bf16(std::size_t(weight.k) * maximum, 103U);
    auto output         = make_buffer(std::size_t(weight.n / 2) * maximum * 2);
    auto scratch        = make_buffer(std::max<std::size_t>(256, capacity));
    MemoryArena arena(scratch.span());
    L2FlushBuffer flush(options.timing.flush_bytes);
    measurement::Csv csv(
        options.timing.csv_out,
        "operand_bytes,operand_gb_s,useful_tflops,bandwidth_reference_us,compute_reference_us");
    for (int t : options.tokens) {
        Tensor x(input.span(), DType::BF16, {weight.k, t}),
            y(output.span(), DType::BF16, {weight.n / 2, t});
        measurement::Row row;
        row.op              = "linear_swiglu_clamped";
        row.format          = options.format;
        row.policy          = options.format == "bf16" ? "a16" : "a4";
        row.n               = weight.n;
        row.k               = weight.k;
        row.tokens          = t;
        row.workspace_bytes = capacity;
        const auto bytes    = fixture.model_weight_bytes() + 2ULL * (weight.k + weight.n / 2) * t;
        const double flops  = 2.0 * weight.n * weight.k * t;
        for (const auto& execution :
             measurement::selections(options.timing.execution, "graph", "eager"))
            for (const auto& cache :
                 measurement::selections(options.timing.cache, "cold", "warm")) {
                arena.reset_peak();
                const auto result = measurement::measure(
                    [&](auto cuda) { ops::linear_swiglu(x, weight, y, 10.f, policy, arena, cuda); },
                    [](auto) {}, flush, device_context().execution_view(), options.timing,
                    execution, cache);
                row.workspace_peak = arena.peak_used();
                csv.write(row, result, execution, cache,
                          std::to_string(bytes) + ',' +
                              format_metric(bytes / (result.gpu.median_us * 1000), 3, "") + ',' +
                              format_metric(flops / (result.gpu.median_us * 1e6), 3, "") + ',' +
                              format_metric(bytes / (kRoofline.memory_gbs * 1000), 3, "") + ',' +
                              format_metric(flops / (peak * 1e6), 3, ""));
            }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        measurement::hardware();
        if (options.format == "bf16") {
            auto weight = make_direct_bf16_weight(2048, 4096, 191U);
            run(options, weight);
        } else {
            auto weight = make_nvfp4_weight(12288, 4096, 191U);
            run(options, weight);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
