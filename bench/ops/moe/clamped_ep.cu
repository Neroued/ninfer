#include "ops/moe/clamped_ep_fixture.h"
#include "op_measurement.cuh"
#include "common/roofline.h"

#include <set>

using namespace ninfer;
using namespace ninfer::bench;

namespace {
namespace ep = ninfer::test::clamped_ep;

struct Options {
    measurement::Options timing;
    std::vector<int> tokens{1, 6, 16, 32, 128, 512, 2048, 4096};
    std::string format = "nvfp4", distribution = "mixed", placement = "permuted";
    int hits = -1, side = 0;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout
                << "Clamped routed SwiGLU: H=4096, I=2048, E=288, TopK=8, 144 local experts.\n"
                   "  --format nvfp4|fp8 --tokens T[,T...]\n"
                   "  --distribution mixed|zero|all|skew|sweep --hits 0..8\n"
                   "  --placement permuted|contiguous|interleaved --side 0|1\n"
                   "Semantic traffic counts unique weights and operand I/O; private staging is "
                   "excluded.\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (measurement::parse(o.timing, flag, value)) continue;
        if (flag == "--format")
            o.format = value;
        else if (flag == "--tokens")
            o.tokens = measurement::integers(value);
        else if (flag == "--distribution")
            o.distribution = value;
        else if (flag == "--placement")
            o.placement = value;
        else if (flag == "--side")
            o.side = measurement::integer(value, 0, 1);
        else if (flag == "--hits")
            o.hits = measurement::integer(value, 0, 8);
        else
            throw std::invalid_argument("unknown option");
    }
    if ((o.format != "nvfp4" && o.format != "fp8") ||
        (o.distribution != "mixed" && o.distribution != "zero" && o.distribution != "all" &&
         o.distribution != "skew" && o.distribution != "sweep") ||
        (o.placement != "permuted" && o.placement != "contiguous" && o.placement != "interleaved"))
        throw std::invalid_argument("invalid workload");
    return o;
}

void run(const Options& o) {
    const int maximum = *std::max_element(o.tokens.begin(), o.tokens.end());
    const auto cuda   = device_context().execution_view();
    const auto place  = o.placement == "permuted"     ? ep::Placement::Permuted
                        : o.placement == "contiguous" ? ep::Placement::Contiguous
                                                      : ep::Placement::Interleaved;
    ep::Weights weights(cuda.device(), place, o.side, o.format == "fp8");
    weights.release_host_oracle();
    const auto profile  = weights.profile();
    const auto capacity = ops::routed_swiglu_workspace_capacity_bytes(
        profile, weights.subset().kind, DType::BF16, ops::MoeEpilogue::Write, 1, maximum);
    auto input        = make_bf16(std::size_t(ep::H) * maximum, 2711, -.7F, .7F);
    auto output       = make_buffer(std::size_t(ep::H) * maximum * 2);
    auto ids          = make_buffer(std::size_t(ep::K) * maximum * 4);
    auto coefficients = make_buffer(std::size_t(ep::K) * maximum * 4);
    auto scratch      = make_buffer(std::max<std::size_t>(capacity, 256));
    MemoryArena workspace(scratch.span());
    L2FlushBuffer flush(o.timing.flush_bytes);
    measurement::Csv csv(
        o.timing.csv_out,
        "distribution,hits_per_token,placement,side,local_experts,local_hits,active_tokens,unique_"
        "local_experts,"
        "bank_backing_bytes,unique_weight_bytes,per_hit_weight_bytes,semantic_bytes,useful_flops,"
        "semantic_gb_s,useful_tflops,bandwidth_reference_us,compute_reference_us");
    const auto distributions = o.distribution == "sweep"
                                   ? std::vector<std::string>{"zero", "all", "mixed", "skew"}
                                   : std::vector<std::string>{o.distribution};
    for (const auto& distribution : distributions)
        for (int t : o.tokens) {
            const auto selected = ep::make_routes(weights, t, distribution.c_str(), o.hits);
            ids.copy_from_host(selected.ids.data(), selected.ids.size() * 4);
            coefficients.copy_from_host(selected.coefficients.data(),
                                        selected.coefficients.size() * 4);
            Tensor x(input.span(), DType::BF16, {ep::H, t}),
                y(output.span(), DType::BF16, {ep::H, t});
            ops::ExpertRoutes routes{Tensor(ids.span(), DType::I32, {ep::K, t}),
                                     Tensor(coefficients.span(), DType::FP32, {ep::K, t})};
            const auto unique =
                std::count_if(selected.per_expert.begin(), selected.per_expert.end(),
                              [](int n) { return n > 0; });
            const std::set<int> route_experts(selected.ids.begin(), selected.ids.end());
            const auto unique_bytes   = unique * weights.encoded_expert_bytes();
            const std::uint64_t bytes = unique_bytes + 2ULL * ep::H * (t + selected.active_tokens) +
                                        4ULL * (ep::K * t + selected.hits) +
                                        4ULL * route_experts.size() +
                                        unique * (o.format == "fp8" ? 16 : 24);
            const double flops = 6.0 * ep::H * ep::I * selected.hits;
            const double peak =
                o.format == "nvfp4" ? kRoofline.nvfp4_f32_tflops : kRoofline.fp8_f32_tflops;
            measurement::Row row;
            row.op              = "routed_swiglu_clamped";
            row.format          = o.format;
            row.policy          = o.format == "nvfp4" ? "a4" : "a8";
            row.n               = ep::H;
            row.k               = ep::I;
            row.tokens          = t;
            row.workspace_bytes = capacity;
            for (const auto& execution :
                 measurement::selections(o.timing.execution, "graph", "eager"))
                for (const auto& cache : measurement::selections(o.timing.cache, "cold", "warm")) {
                    workspace.reset_peak();
                    const auto result = measurement::measure(
                        [&](auto view) {
                            ops::routed_swiglu(x, profile, weights.experts(), routes,
                                               weights.subset(), ops::MoeEpilogue::Write, y,
                                               workspace, view);
                        },
                        [](auto) {}, flush, cuda, o.timing, execution, cache);
                    row.workspace_peak = workspace.peak_used();
                    csv.write(
                        row, result, execution, cache,
                        distribution + ',' + std::to_string(o.hits) + ',' + o.placement + ',' +
                            std::to_string(o.side) + ',' + std::to_string(ep::Local) + ',' +
                            std::to_string(selected.hits) + ',' +
                            std::to_string(selected.active_tokens) + ',' + std::to_string(unique) +
                            ',' + std::to_string(weights.backing_bytes()) + ',' +
                            std::to_string(unique_bytes) + ',' +
                            std::to_string(selected.hits * weights.encoded_expert_bytes()) + ',' +
                            std::to_string(bytes) + ',' + std::to_string(flops) + ',' +
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
        run(options);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
