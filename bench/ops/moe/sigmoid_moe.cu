#include "ops/moe/clamped_ep_fixture.h"
#include "op_measurement.cuh"
#include "common/roofline.h"
#include "direct_bf16_weight.cuh"

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
                << "Complete sigmoid MoE: H=4096, I=2048, E=288, TopK=8, 144 local experts, BF16 "
                   "shared I=1024.\n"
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
    ep::Weights experts(cuda.device(), place, o.side, o.format == "fp8");
    experts.release_host_oracle();
    auto router  = make_direct_bf16_weight(ep::E, ep::H, 2911, .02f);
    auto gate_up = make_direct_bf16_weight(2048, ep::H, 2913, .02f);
    auto down    = make_direct_bf16_weight(ep::H, 1024, 2915, .02f);
    const ops::SwiGLUWeights shared{{gate_up.weight, ops::LinearPolicy::A16Only},
                                    {down.weight, ops::LinearPolicy::A16Only}};
    auto shared_profile        = ops::swiglu_profile(shared);
    shared_profile.clamp_limit = 10;
    const ops::SigmoidMoeProfile profile{
        {{ep::H, ep::E, ep::K}, 2.5f, 1e-20f}, experts.profile(), shared_profile};
    const auto capacity = ops::topk_sigmoid_swiglu_moe_workspace_capacity_bytes(
        profile, experts.subset().kind, 1, maximum);
    auto input        = make_bf16(std::size_t(ep::H) * maximum, 2711, -.7f, .7f);
    auto output       = make_buffer(std::size_t(ep::H) * maximum * 2);
    auto ids          = make_buffer(std::size_t(ep::K) * maximum * 4);
    auto coefficients = make_buffer(std::size_t(ep::K) * maximum * 4);
    auto bias         = make_buffer(ep::E * 4);
    auto scratch      = make_buffer(capacity);
    MemoryArena workspace(scratch.span());
    L2FlushBuffer flush(o.timing.flush_bytes);
    measurement::Csv csv(
        o.timing.csv_out,
        "distribution,hits_per_token,placement,side,local_hits,active_tokens,unique_local_experts,"
        "bank_backing_bytes,"
        "unique_weight_bytes,semantic_bytes,router_flops,shared_flops,routed_flops,semantic_gb_s,"
        "useful_tflops,bandwidth_reference_us,compute_reference_us");
    const auto distributions = o.distribution == "sweep"
                                   ? std::vector<std::string>{"zero", "all", "mixed", "skew"}
                                   : std::vector<std::string>{o.distribution};
    for (const auto& distribution : distributions)
        for (int t : o.tokens) {
            std::vector<float> correction(ep::E);
            if (o.hits >= 0) {
                for (int j = 0; j < ep::K; ++j)
                    correction[j < o.hits ? experts.local_id(j) : experts.remote_id(j)] = 2;
            } else if (distribution == "zero" || distribution == "all") {
                for (int j = 0; j < ep::Local; ++j)
                    correction[distribution == "all" ? experts.local_id(j) : experts.remote_id(j)] =
                        2;
            } else if (distribution == "skew") {
                for (int j = 0; j < ep::K; ++j) correction[experts.local_id(j)] = 2;
            } else {
                for (int e = 0; e < ep::E; ++e) correction[e] = (e % 17 - 8) * .003f;
            }
            bias.copy_from_host(correction.data(), ep::E * 4);
            Tensor x(input.span(), DType::BF16, {ep::H, t}),
                y(output.span(), DType::BF16, {ep::H, t});
            Tensor correction_tensor(bias.span(), DType::FP32, {ep::E});
            const ops::SigmoidMoeWeights weights{router.weight, correction_tensor,
                                                 experts.experts(), shared};
            // Characterize the actual global routes before timing the complete public Op.
            ops::ExpertRoutes routes{Tensor(ids.span(), DType::I32, {ep::K, t}),
                                     Tensor(coefficients.span(), DType::FP32, {ep::K, t})};
            ops::topk_sigmoid_route(x, profile.routing, router.weight, correction_tensor, routes,
                                    workspace, cuda);
            CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
            std::vector<int> selected(std::size_t(ep::K) * t);
            ids.copy_to_host(selected.data(), selected.size() * 4);
            std::vector<int> histogram(ep::E);
            std::uint64_t hits = 0, active = 0;
            for (int token = 0; token < t; ++token) {
                bool local = false;
                for (int j = 0; j < ep::K; ++j) {
                    const int e = selected[std::size_t(token) * ep::K + j];
                    if (experts.active(e)) {
                        ++histogram[e];
                        ++hits;
                        local = true;
                    }
                }
                active += local;
            }
            const auto unique =
                std::count_if(histogram.begin(), histogram.end(), [](int n) { return n > 0; });
            const auto weight_bytes = unique * experts.encoded_expert_bytes() +
                                      router.model_weight_bytes() + gate_up.model_weight_bytes() +
                                      down.model_weight_bytes();
            const auto bytes = weight_bytes + 4ULL * ep::H * t + ep::E * 4 + 8ULL * ep::K * t;
            const double router_flops = 2.0 * ep::H * ep::E * t;
            const double shared_flops = 6.0 * ep::H * 1024 * t;
            const double routed_flops = 6.0 * ep::H * ep::I * hits;
            const double flops        = router_flops + shared_flops + routed_flops;
            const double routed_peak =
                o.format == "nvfp4" ? kRoofline.nvfp4_f32_tflops : kRoofline.fp8_f32_tflops;
            const double compute_reference =
                (router_flops + shared_flops) / (kRoofline.bf16_f32_tflops * 1e6) +
                routed_flops / (routed_peak * 1e6);
            measurement::Row row;
            row.op              = "topk_sigmoid_swiglu_moe";
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
                            ops::topk_sigmoid_swiglu_moe(x, profile, weights, experts.subset(), y,
                                                         workspace, view);
                        },
                        [](auto) {}, flush, cuda, o.timing, execution, cache);
                    row.workspace_peak = workspace.peak_used();
                    csv.write(
                        row, result, execution, cache,
                        distribution + ',' + std::to_string(o.hits) + ',' + o.placement + ',' +
                            std::to_string(o.side) + ',' + std::to_string(hits) + ',' +
                            std::to_string(active) + ',' + std::to_string(unique) + ',' +
                            std::to_string(experts.backing_bytes()) + ',' +
                            std::to_string(weight_bytes) + ',' + std::to_string(bytes) + ',' +
                            std::to_string(router_flops) + ',' + std::to_string(shared_flops) +
                            ',' + std::to_string(routed_flops) + ',' +
                            format_metric(bytes / (result.gpu.median_us * 1000), 3, "") + ',' +
                            format_metric(flops / (result.gpu.median_us * 1e6), 3, "") + ',' +
                            format_metric(bytes / (kRoofline.memory_gbs * 1000), 3, "") + ',' +
                            format_metric(compute_reference, 3, ""));
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
