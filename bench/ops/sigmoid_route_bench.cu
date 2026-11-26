#include "ninfer/ops/moe.h"
#include "op_measurement.cuh"
#include "direct_bf16_weight.cuh"
#include "common/roofline.h"

using namespace ninfer;
using namespace ninfer::bench;

namespace {
#if NINFER_TARGET_RTX5090
constexpr ops::SigmoidRouteProfile kProfile{{2304, 128, 8}, 8.0f, 1e-20f};
constexpr std::string_view kProfileName = "h2304-e128-k8";
#elif NINFER_TARGET_SPARK
constexpr ops::SigmoidRouteProfile kProfile{{4096, 288, 8}, 2.5f, 1e-20f};
constexpr std::string_view kProfileName = "h4096-e288-k8";
#endif

struct Options {
    measurement::Options timing;
    std::vector<int> tokens{1, 6, 16, 32, 128, 512, 2048, 4096};
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "Corrected sigmoid routing (" << kProfileName
                      << ")\n"
                         "  --tokens T[,T...] --sweep START:END[:STEP] --graph-timer body\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (flag == "--tokens")
            o.tokens = measurement::integers(value);
        else if (flag == "--sweep")
            o.tokens = measurement::sweep(value);
        else if (flag == "--graph-timer") {
            if (value != "body") throw std::invalid_argument("graph timer must be body");
        } else if (!measurement::parse(o.timing, flag, value))
            throw std::invalid_argument("unknown option");
    }
    return o;
}

void run(const Options& options) {
    const auto [h, e, k] = kProfile.route;
    const int maximum    = *std::max_element(options.tokens.begin(), options.tokens.end());
    auto control         = make_direct_bf16_weight(e, h, 2631);
    auto input           = make_bf16(std::size_t(h) * maximum, 2633U);
    auto ids             = make_buffer(std::size_t(k) * maximum * 4);
    auto coefficients    = make_buffer(std::size_t(k) * maximum * 4);
    auto bias            = make_buffer(e * 4);
    std::vector<float> host_bias(e);
    for (int j = 0; j < e; ++j)
        host_bias[j] = (int((std::uint32_t(j + 2635) * 2654435761U) % 31) - 15) * .01f;
    bias.copy_from_host(host_bias.data(), e * 4);
    const auto minimum = *std::min_element(options.tokens.begin(), options.tokens.end());
    const auto capacity =
        ops::topk_sigmoid_route_workspace_capacity_bytes(kProfile, minimum, maximum);
    auto scratch = make_buffer(capacity);
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
        row.op              = "topk_sigmoid_route";
        row.format          = "bf16";
        row.policy          = "a16";
        row.n               = e;
        row.k               = h;
        row.tokens          = t;
        row.workspace_bytes = ops::topk_sigmoid_route_workspace_capacity_bytes(kProfile, t, t);
        const auto bytes    = control.model_weight_bytes() + e * 4 + 2ULL * h * t + 8ULL * k * t;
        const double flops  = 2.0 * h * e * t;
        for (const auto& execution :
             measurement::selections(options.timing.execution, "graph", "eager"))
            for (const auto& cache :
                 measurement::selections(options.timing.cache, "cold", "warm")) {
                arena.reset_peak();
                const auto result = measurement::measure(
                    [&](auto cuda) {
                        ops::topk_sigmoid_route(x, kProfile, control.weight, b, routes, {}, arena,
                                                cuda);
                    },
                    [](auto) {}, flush, device_context().execution_view(), options.timing,
                    execution, cache);
                row.workspace_peak = arena.peak_used();
                csv.write(row, result, execution, cache,
                          std::to_string(2 * row.workspace_peak) + ',' + std::to_string(bytes) +
                              ',' + format_metric(bytes / (result.gpu.median_us * 1000), 3, "") +
                              ',' + format_metric(flops / (result.gpu.median_us * 1e6), 3, "") +
                              ',' + format_metric(bytes / (kRoofline.memory_gbs * 1000), 3, "") +
                              ',' +
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
