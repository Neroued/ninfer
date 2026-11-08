#include "ninfer/ops/linear_bias.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/linear_swiglu.h"
#include "op_measurement.cuh"
#include "direct_bf16_weight.cuh"
#include "quantized_weight.cuh"
#include "common/roofline.h"
#include "ops/linear/bias/nvfp4_mul/dispatch.h"
#include "ops/linear/bias/mxfp8/dispatch.h"
#include "ops/linear/split/dispatch.h"

using namespace ninfer;
using namespace ninfer::bench;

namespace {
struct Options {
    measurement::Options timing;
    std::vector<int> tokens{1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    int n = 1024, k = 1176;
    std::string form = "projection", format = "bf16", policy;
};

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "Linear bias: --n N --k K --tokens T[,T...] --sweep A:B[:S]\n"
                         "  --form projection|split|swiglu --format bf16|mxfp8|nvfp4_mul\n"
                         "  --policy a16|a8|a4 (default: MXFP8 a8; other formats a16)\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--form")
            options.form = value;
        else if (flag == "--format")
            options.format = value;
        else if (flag == "--policy")
            options.policy = value;
        else if (flag == "--n")
            options.n = measurement::integer(value);
        else if (flag == "--k")
            options.k = measurement::integer(value);
        else if (flag == "--tokens")
            options.tokens = measurement::integers(value);
        else if (flag == "--sweep")
            options.tokens = measurement::sweep(value);
        else if (!measurement::parse(options.timing, flag, value))
            throw std::invalid_argument("unknown option");
    }
    if (options.form != "projection" && options.form != "split" && options.form != "swiglu")
        throw std::invalid_argument("unknown form");
    if (options.format != "bf16" && options.format != "mxfp8" && options.format != "nvfp4_mul")
        throw std::invalid_argument("unknown format");
    if (options.policy.empty()) options.policy = options.format == "mxfp8" ? "a8" : "a16";
    if (options.policy != "a16" && options.policy != "a8" && options.policy != "a4")
        throw std::invalid_argument("unknown policy");
    if (options.tokens.empty()) throw std::invalid_argument("no token extents");
    return options;
}

template <class Fixture>
void run(const Options& options, Fixture& fixture) {
    const auto& weight = fixture.weight;
    const auto& timing = options.timing;
    const auto policy  = options.policy == "a16"  ? ops::LinearPolicy::A16Only
                         : options.policy == "a8" ? ops::LinearPolicy::AllowA8
                                                  : ops::LinearPolicy::AllowA4;
    const int maximum  = *std::max_element(options.tokens.begin(), options.tokens.end());
    const int n = weight.n, k = weight.k;
    const int rows     = options.form == "swiglu" ? n / 2 : n;
    const int segments = options.form == "split" ? 3 : 1;
    if (n % segments) throw std::invalid_argument("split rows must divide into three segments");
    const std::array segment_rows{n / 3, n / 3, n / 3};
    const auto capacity =
        options.form == "projection"
            ? ops::linear_bias_workspace_capacity_bytes(weight.qtype, n, k, policy, 1, maximum)
        : options.form == "split"
            ? ops::linear_split_bias_workspace_capacity_bytes(weight.qtype, n, k, segment_rows,
                                                              policy, 1, maximum)
            : ops::linear_swiglu_bias_workspace_capacity_bytes(weight.qtype, n, k, policy, 1,
                                                               maximum);
    auto input   = make_bf16(std::size_t(k) * maximum, 17);
    auto scratch = make_buffer(std::max<std::size_t>(capacity, 256));
    MemoryArena arena(scratch.span());
    std::vector<DeviceBuffer> outputs;
    for (int s = 0; s < segments; ++s)
        outputs.push_back(make_buffer(std::size_t(rows / segments) * maximum * 2));
    auto bias = make_bf16(n, 19);
    Tensor b(bias.span(), DType::BF16, {n});
    L2FlushBuffer flush(timing.flush_bytes);
    measurement::Csv csv(timing.csv_out,
                         "operand_bytes,workspace_traffic_bytes,effective_bytes,effective_gb_s,"
                         "useful_tflops,bandwidth_reference_us,compute_reference_us,tensor_profile,"
                         "tensor_peak_tflops,roofline_reference_us");
    for (int t : options.tokens) {
        Tensor x(input.span(), DType::BF16, {k, t});
        std::vector<Tensor> y;
        for (auto& output : outputs)
            y.emplace_back(output.span(), DType::BF16,
                           std::initializer_list<int>{rows / segments, t});
        measurement::Row row;
        row.op                   = options.form == "projection" ? "linear_bias"
                                   : options.form == "split"    ? "linear_split_bias"
                                                                : "linear_swiglu_bias";
        row.format               = options.format;
        row.policy               = options.policy;
        row.n                    = n;
        row.k                    = k;
        row.tokens               = t;
        row.workspace_bytes      = capacity;
        const auto operand_bytes = fixture.model_weight_bytes() + 2ULL * n + 2ULL * (rows + k) * t;
        const double flops       = 2.0 * n * k * t;
        for (const auto& execution : measurement::selections(timing.execution, "graph", "eager"))
            for (const auto& cache : measurement::selections(timing.cache, "cold", "warm")) {
                arena.reset_peak();
                const auto result = measurement::measure(
                    [&](auto cuda) {
                        if (options.form == "projection")
                            ops::linear_bias(x, weight, b, y[0], policy, arena, cuda);
                        else if (options.form == "split")
                            ops::linear_split(x, weight, b, y, policy, arena, cuda);
                        else
                            ops::linear_swiglu(x, weight, b, y[0], 10.F, policy, arena, cuda);
                    },
                    [](auto) {}, flush, device_context().execution_view(), timing, execution,
                    cache);
                row.workspace_peak = arena.peak_used();
                const bool a8      = weight.qtype == QType::MXFP8 && row.workspace_peak != 0;
                // The registered native K1024 workspaces have no internal alignment gap.
                // Peak usage includes initialized code rows beyond the logical token extent.
                const auto workspace_bytes = a8 ? 2ULL * row.workspace_peak : 0ULL;
                const auto bytes           = operand_bytes + workspace_bytes;
                bool tensor =
                    weight.qtype == QType::MXFP8 ? true
                    : weight.qtype == QType::NVFP4_MUL
                        ? options.form != "projection" ||
                              ops::detail::select_nvfp4_mul_bias_route(n, k, t, policy).tensor_core
                        : t > 1;
                if (weight.qtype == QType::MXFP8 && options.form == "projection") {
                    tensor = ops::detail::select_mxfp8_bias_route(n, k, t, policy).tensor_core;
                } else if (weight.qtype == QType::MXFP8 && options.form == "split") {
                    const auto& shape = ops::detail::linear_split::select_split_bias_shape(
                        weight.qtype, n, k, policy, segment_rows);
                    if (shape.uses_tensor_cores) tensor = shape.uses_tensor_cores(policy, t);
                }
                const double peak     = !tensor ? std::numeric_limits<double>::quiet_NaN()
                                        : a8    ? kRoofline.mxfp8_f32_tflops
                                                : kRoofline.bf16_f32_tflops;
                const double compute  = flops / (peak * 1e6);
                const char* profile   = !tensor ? "SIMT_FP32" : a8 ? "MXFP8_F32ACC" : "BF16_F32ACC";
                const double memory   = cache == "cold"
                                            ? bytes / (kRoofline.sustained_read_gbs * 1000)
                                            : std::numeric_limits<double>::quiet_NaN();
                const double roofline = cache == "cold" && tensor
                                            ? std::max(memory, compute)
                                            : std::numeric_limits<double>::quiet_NaN();
                csv.write(row, result, execution, cache,
                          std::to_string(operand_bytes) + ',' + std::to_string(workspace_bytes) +
                              ',' + std::to_string(bytes) + ',' +
                              format_metric(bytes / (result.gpu.median_us * 1000), 3, "") + ',' +
                              format_metric(flops / (result.gpu.median_us * 1e6), 3, "") + ',' +
                              format_metric(memory, 3, "") + ',' + format_metric(compute, 3, "") +
                              ',' + profile + ',' + format_metric(peak, 3, "") + ',' +
                              format_metric(roofline, 3, ""));
            }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        measurement::hardware();
        if (options.format == "bf16") {
            auto weight = make_direct_bf16_weight(options.n, options.k);
            run(options, weight);
        } else if (options.format == "mxfp8") {
            auto weight = make_mxfp8_weight(options.n, options.k);
            run(options, weight);
        } else {
            auto weight = make_nvfp4_mul_weight(options.n, options.k);
            run(options, weight);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
