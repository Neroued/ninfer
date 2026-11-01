#include "ninfer/ops/moe.h"
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
            std::cout << "Complete clamped SwiGLU benchmark\n"
                         "  --format nvfp4|bf16|q8 --tokens T[,T...] --sweep START:END[:STEP]\n"
                         "  H=4096. NVFP4: I=6144, AllowA4; BF16/Q8: I=1024, A16Only.\n"
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
    if (o.format != "bf16" && o.format != "nvfp4" && o.format != "q8")
        throw std::invalid_argument("unknown weight format");
    return o;
}

template <class Fixture>
void run(const Options& options, Fixture& gu, Fixture& down) {
    const auto& weight = gu.weight;
    const auto policy =
        options.format != "nvfp4" ? ops::LinearPolicy::A16Only : ops::LinearPolicy::AllowA4;
    const int maximum = *std::max_element(options.tokens.begin(), options.tokens.end());
    ops::SwiGLUWeights weights{{gu.weight, policy}, {down.weight, policy}};
    auto profile        = ops::swiglu_profile(weights);
    profile.clamp_limit = 10.f;
    const auto capacity = ops::swiglu_workspace_capacity_bytes(profile, 1, maximum);
    auto input          = make_bf16(std::size_t(weight.k) * maximum, 103U);
    auto output         = make_buffer(std::size_t(weight.k) * maximum * 2);
    auto scratch        = make_buffer(std::max<std::size_t>(256, capacity));
    MemoryArena arena(scratch.span());
    L2FlushBuffer flush(options.timing.flush_bytes);
    measurement::Csv csv(
        options.timing.csv_out,
        "operand_bytes,operand_gb_s,useful_tflops,bandwidth_reference_us,compute_reference_us,"
        "tensor_profile,tensor_peak_tflops,selected_compute_reference_us,scratch_logical_bytes,"
        "interval_capacity_bytes");
    for (int t : options.tokens) {
        Tensor x(input.span(), DType::BF16, {weight.k, t}),
            y(output.span(), DType::BF16, {weight.k, t});
        measurement::Row row;
        row.op              = "swiglu";
        row.format          = options.format;
        row.policy          = options.format != "nvfp4" ? "a16" : "a4";
        row.n               = weight.k;
        row.k               = weight.n / 2;
        row.tokens          = t;
        row.workspace_bytes = ops::swiglu_workspace_capacity_bytes(profile, t, t);
        const auto bytes =
            gu.model_weight_bytes() + down.model_weight_bytes() + 4ULL * weight.k * t;
        const double flops    = 3.0 * weight.n * weight.k * t;
        const double gu_flops = 2.0 * weight.n * weight.k * t, down_flops = flops - gu_flops;
        const double absent = std::numeric_limits<double>::quiet_NaN();
        double peak = absent, compute = absent, selected_compute = absent;
        const char* tensor_profile = "SIMT_FP32";
        if ((options.format == "q8" && t >= 2) || (options.format == "bf16" && t >= 17)) {
            tensor_profile = options.format == "q8" && t >= 49 ? "BF16_F32ACC_COMPENSATED_GU_DOWN"
                                                               : "BF16_F32ACC_COMPENSATED_DOWN";
            peak           = kRoofline.bf16_f32_tflops;
            compute        = flops / (peak * 1e6);
            selected_compute =
                ((options.format == "q8" && t >= 49 ? 2 : 1) * gu_flops + 2 * down_flops) /
                (peak * 1e6);
        } else if (options.format == "bf16" && t >= 2) {
            tensor_profile = "BF16_F32ACC+TF32_F32ACC";
        } else if (options.format == "nvfp4" && t >= 3 && t <= 16) {
            tensor_profile   = "BF16_F32ACC+NVFP4_F32ACC";
            selected_compute = gu_flops / (kRoofline.bf16_f32_tflops * 1e6) +
                               down_flops / (kRoofline.nvfp4_f32_tflops * 1e6);
        } else if (options.format == "nvfp4" && t >= 17) {
            tensor_profile = "NVFP4_F32ACC";
            peak           = kRoofline.nvfp4_f32_tflops;
            compute = selected_compute = flops / (peak * 1e6);
        }
        const double scratch_bytes = options.format != "nvfp4" ? 2.0 * row.workspace_bytes : absent;
        for (const auto& execution :
             measurement::selections(options.timing.execution, "graph", "eager"))
            for (const auto& cache :
                 measurement::selections(options.timing.cache, "cold", "warm")) {
                arena.reset_peak();
                const auto result = measurement::measure(
                    [&](auto cuda) { ops::swiglu(x, profile, weights, y, arena, cuda); },
                    [](auto) {}, flush, device_context().execution_view(), options.timing,
                    execution, cache);
                row.workspace_peak = arena.peak_used();
                if (arena.used() != 0 || row.workspace_peak > row.workspace_bytes)
                    throw std::runtime_error("SwiGLU workspace contract violated");
                csv.write(row, result, execution, cache,
                          std::to_string(bytes) + ',' +
                              format_metric(bytes / (result.gpu.median_us * 1000), 3, "") + ',' +
                              format_metric(flops / (result.gpu.median_us * 1e6), 3, "") + ',' +
                              format_metric(cache == "cold"
                                                ? bytes / (kRoofline.sustained_read_gbs * 1000)
                                                : absent,
                                            3, "") +
                              ',' + format_metric(compute, 3, "") + ',' + tensor_profile + ',' +
                              format_metric(peak, 3, "") + ',' +
                              format_metric(selected_compute, 3, "") + ',' +
                              format_metric(scratch_bytes, 0, "") + ',' + std::to_string(capacity));
            }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        measurement::hardware();
        if (options.format == "bf16") {
            auto gu   = make_direct_bf16_weight(2048, 4096, 191U);
            auto down = make_direct_bf16_weight(4096, 1024, 193U);
            run(options, gu, down);
        } else if (options.format == "q8") {
            auto gu   = make_row_split_weight(QType::Q8_G32_FP16, 2048, 4096, 4096, 191U);
            auto down = make_row_split_weight(QType::Q8_G32_FP16, 4096, 1024, 1024, 193U);
            run(options, gu, down);
        } else {
            auto gu   = make_nvfp4_weight(12288, 4096, 191U);
            auto down = make_nvfp4_weight(4096, 6144, 193U);
            run(options, gu, down);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
