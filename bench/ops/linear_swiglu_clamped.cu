#include "ninfer/ops/linear_swiglu.h"
#include "op_measurement.cuh"
#include "quantized_weight.cuh"
#include "direct_bf16_weight.cuh"
#include "common/roofline.h"
#include "ops/linear/swiglu/mxfp8/dispatch.h"

using namespace ninfer;
using namespace ninfer::bench;

namespace {
struct Options {
    measurement::Options timing;
    std::vector<int> tokens{1, 6, 16, 32, 128, 512, 2048, 4096};
    std::string format = "nvfp4";
    std::string policy;
    bool clamped = true;
    int n = 0, k = 4096;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout
                << "Linear SwiGLU benchmark\n"
                   "  --product ordinary|clamped (default clamped)\n"
                   "  --format nvfp4|bf16|q8|mxfp8|fp8_block_fp32|nvfp4_mul\n"
                   "  --policy a16|a8|a4 --tokens T[,T...] --sweep START:END[:STEP] --n N --k "
                   "K\n"
                   "  Defaults: NVFP4 N=12288, K=4096, AllowA4; BF16: N=2048, K=4096, A16Only. Q8: "
                   "N=2048, K=4096, A16Only.\n"
                   "  MXFP8 defaults to AllowA8. NVFP4_MUL and block FP8 default to A16Only.\n"
                   "  Clamped product uses limit=10. Effective traffic includes activation "
                   "packing.\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--n")
            o.n = measurement::integer(value);
        else if (flag == "--k")
            o.k = measurement::integer(value);
        else if (flag == "--format")
            o.format = value;
        else if (flag == "--policy")
            o.policy = value;
        else if (flag == "--product") {
            if (value != "ordinary" && value != "clamped")
                throw std::invalid_argument("product must be ordinary or clamped");
            o.clamped = value == "clamped";
        } else if (flag == "--tokens")
            o.tokens = measurement::integers(value);
        else if (flag == "--sweep")
            o.tokens = measurement::sweep(value);
        else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option");
    }
    if (o.format != "bf16" && o.format != "nvfp4" && o.format != "q8" && o.format != "mxfp8" &&
        o.format != "fp8_block_fp32" && o.format != "nvfp4_mul")
        throw std::invalid_argument("unknown weight format");
    if (o.n == 0)
        o.n = o.format == "nvfp4" || o.format == "mxfp8" ? 12288
              : o.format == "nvfp4_mul"                  ? 20480
                                                         : 2048;
    if (o.policy.empty())
        o.policy = o.format == "nvfp4" ? "a4" : o.format == "mxfp8" ? "a8" : "a16";
    if (o.policy != "a16" && o.policy != "a8" && o.policy != "a4")
        throw std::invalid_argument("unknown compute policy");
    if (o.tokens.empty()) throw std::invalid_argument("no token extents");
    return o;
}

template <class Fixture>
void run(const Options& options, Fixture& fixture) {
    const auto& weight  = fixture.weight;
    const auto policy   = options.policy == "a16"  ? ops::LinearPolicy::A16Only
                          : options.policy == "a8" ? ops::LinearPolicy::AllowA8
                                                   : ops::LinearPolicy::AllowA4;
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
        "operand_bytes,workspace_traffic_bytes,effective_bytes,effective_gb_s,useful_tflops,"
        "bandwidth_reference_us,compute_reference_us,"
        "tensor_profile,tensor_peak_tflops,roofline_reference_us");
    for (int t : options.tokens) {
        Tensor x(input.span(), DType::BF16, {weight.k, t}),
            y(output.span(), DType::BF16, {weight.n / 2, t});
        measurement::Row row;
        row.op              = options.clamped ? "linear_swiglu_clamped" : "linear_swiglu";
        row.format          = options.format;
        row.policy          = options.policy;
        row.n               = weight.n;
        row.k               = weight.k;
        row.tokens          = t;
        row.workspace_bytes = capacity;
        const auto operand_bytes =
            fixture.model_weight_bytes() + 2ULL * (weight.k + weight.n / 2) * t;
        const double flops = 2.0 * weight.n * weight.k * t;
        for (const auto& execution :
             measurement::selections(options.timing.execution, "graph", "eager"))
            for (const auto& cache :
                 measurement::selections(options.timing.cache, "cold", "warm")) {
                arena.reset_peak();
                const auto result = measurement::measure(
                    [&](auto cuda) {
                        if (options.clamped)
                            ops::linear_swiglu(x, weight, y, 10.f, policy, arena, cuda);
                        else
                            ops::linear_swiglu(x, weight, y, policy, arena, cuda);
                    },
                    [](auto) {}, flush, device_context().execution_view(), options.timing,
                    execution, cache);
                row.workspace_peak   = arena.peak_used();
                const bool native_a8 = weight.qtype == QType::MXFP8 && row.workspace_peak != 0;
                const bool native_a4 = weight.qtype == QType::NVFP4 && row.workspace_peak != 0;
                // Registered native K4096 workspaces have no internal alignment gap.
                // Peak usage includes any initialized code rows beyond the logical extent.
                const auto workspace_bytes = native_a8   ? 2ULL * row.workspace_peak
                                             : native_a4 ? 2ULL * t * (weight.k / 2 + weight.k / 16)
                                                         : 0ULL;
                const auto bytes           = operand_bytes + workspace_bytes;
                const bool tensor =
                    weight.qtype == QType::MXFP8
                        ? ops::detail::select_mxfp8_swiglu_route(weight.n, weight.k, t, policy)
                              .tensor_core
                    : weight.qtype == QType::FP8_E4M3FN_BLOCK128X128_FP32 ? t > 2
                    : weight.qtype == QType::NVFP4_MUL ? weight.n == 8192 || t > 2
                    : weight.qtype == QType::NVFP4
                        ? t >= 3
                        : t >= 2 || (weight.qtype == QType::BF16 &&
                                     (weight.n == 20480 || weight.n == 12288));
                const double peak          = !tensor     ? std::numeric_limits<double>::quiet_NaN()
                                             : native_a4 ? kRoofline.nvfp4_f32_tflops
                                             : native_a8 ? kRoofline.mxfp8_f32_tflops
                                                         : kRoofline.bf16_f32_tflops;
                const auto compute_floor   = flops / (peak * 1e6);
                const char* tensor_profile = !tensor     ? "SIMT_FP32"
                                             : native_a4 ? "NVFP4_F32ACC"
                                             : native_a8 ? "MXFP8_F32ACC"
                                                         : "BF16_F32ACC";
                const double read_floor    = cache == "cold"
                                                 ? bytes / (kRoofline.sustained_read_gbs * 1000)
                                                 : std::numeric_limits<double>::quiet_NaN();
                const double roofline      = cache == "cold" && tensor
                                                 ? std::max(read_floor, compute_floor)
                                                 : std::numeric_limits<double>::quiet_NaN();
                csv.write(row, result, execution, cache,
                          std::to_string(operand_bytes) + ',' + std::to_string(workspace_bytes) +
                              ',' + std::to_string(bytes) + ',' +
                              format_metric(bytes / (result.gpu.median_us * 1000), 3, "") + ',' +
                              format_metric(flops / (result.gpu.median_us * 1e6), 3, "") + ',' +
                              format_metric(read_floor, 3, "") + ',' +
                              format_metric(compute_floor, 3, "") + ',' + tensor_profile + ',' +
                              format_metric(peak, 3, "") + ',' + format_metric(roofline, 3, ""));
            }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        measurement::hardware();
        if (options.format == "bf16") {
            auto weight = make_direct_bf16_weight(options.n, options.k, 191U);
            run(options, weight);
        } else if (options.format == "q8") {
            auto weight =
                make_row_split_weight(QType::Q8_G32_FP16, options.n, options.k, options.k, 191U);
            run(options, weight);
        } else if (options.format == "mxfp8") {
            auto weight = make_mxfp8_weight(options.n, options.k, 191U);
            run(options, weight);
        } else if (options.format == "fp8_block_fp32") {
            auto weight = make_fp8_block_weight(options.n, options.k, 191U);
            run(options, weight);
        } else if (options.format == "nvfp4_mul") {
            auto weight = make_nvfp4_mul_weight(options.n, options.k, 191U);
            run(options, weight);
        } else {
            auto weight = make_nvfp4_weight(options.n, options.k, 191U);
            run(options, weight);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
