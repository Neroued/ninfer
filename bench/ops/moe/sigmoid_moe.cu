#include "ops/moe/clamped_ep_fixture.h"
#include "op_measurement.cuh"
#include "common/roofline.h"
#include "direct_bf16_weight.cuh"
#include "quantized_weight.cuh"

using namespace ninfer;
using namespace ninfer::bench;

namespace {
namespace ep = ninfer::test::clamped_ep;

struct Options {
    measurement::Options timing;
    std::vector<int> tokens{1, 6, 16, 32, 128, 512, 2048, 4096};
    std::string format = "nvfp4", distribution = "mixed", placement = "permuted";
    std::string shared_format = "bf16", subset = "local";
    int hits = -1, side = 0;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout
                << "Complete sigmoid MoE: H=4096, I=2048, E=288, TopK=8, 144 local experts, "
                   "shared I=1024.\n"
                   "  --format nvfp4|fp8 --tokens T[,T...]\n"
                   "  --shared-format bf16|q8 --subset local|none\n"
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
        else if (flag == "--shared-format")
            o.shared_format = value;
        else if (flag == "--subset")
            o.subset = value;
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
        (o.shared_format != "bf16" && o.shared_format != "q8") ||
        (o.subset != "local" && o.subset != "none") ||
        (o.distribution != "mixed" && o.distribution != "zero" && o.distribution != "all" &&
         o.distribution != "skew" && o.distribution != "sweep") ||
        (o.placement != "permuted" && o.placement != "contiguous" && o.placement != "interleaved"))
        throw std::invalid_argument("invalid workload");
    return o;
}

template <class Fixture>
void run(const Options& o, Fixture& gate_up, Fixture& down) {
    const int maximum = *std::max_element(o.tokens.begin(), o.tokens.end());
    const auto cuda   = device_context().execution_view();
    const auto place  = o.placement == "permuted"     ? ep::Placement::Permuted
                        : o.placement == "contiguous" ? ep::Placement::Contiguous
                                                      : ep::Placement::Interleaved;
    ep::Weights experts(cuda.device(), place, o.side, o.format == "fp8");
    experts.release_host_oracle();
    auto router = make_direct_bf16_weight(ep::E, ep::H, 2911, .02f);
    const ops::SwiGLUWeights shared{{gate_up.weight, ops::LinearPolicy::A16Only},
                                    {down.weight, ops::LinearPolicy::A16Only}};
    auto shared_profile        = ops::swiglu_profile(shared);
    shared_profile.clamp_limit = 10;
    const ops::SigmoidMoeProfile profile{
        {{ep::H, ep::E, ep::K}, 2.5f, 1e-20f}, experts.profile(), shared_profile};
    const auto subset =
        o.subset == "none" ? ops::ExpertSubset{ops::ExpertSubsetKind::None, {}} : experts.subset();
    const auto capacity =
        ops::topk_sigmoid_swiglu_moe_workspace_capacity_bytes(profile, subset.kind, 1, maximum);
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
        "useful_tflops,bandwidth_reference_us,compute_reference_us,shared_format,subset,"
        "arithmetic_profile,selected_compute_reference_us,known_staging_minimum_bytes,"
        "interval_capacity_bytes,router_tensor_peak_tflops,shared_tensor_peak_tflops,"
        "routed_tensor_peak_tflops,router_tensor_pct,shared_useful_tensor_pct,routed_tensor_pct,"
        "selected_tensor_busy_reference_pct");
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
            std::vector<int> selected(std::size_t(ep::K) * t);
            if (subset.kind != ops::ExpertSubsetKind::None) {
                ops::topk_sigmoid_route(x, profile.routing, router.weight, correction_tensor,
                                        routes, workspace, cuda);
                CUDA_CHECK(cudaStreamSynchronize(cuda.stream()));
                ids.copy_to_host(selected.data(), selected.size() * 4);
            }
            std::vector<int> histogram(ep::E);
            std::uint64_t hits = 0, active = 0;
            for (int token = 0; token < t; ++token) {
                bool local = false;
                for (int j = 0; j < ep::K; ++j) {
                    const int e = selected[std::size_t(token) * ep::K + j];
                    if (subset.kind != ops::ExpertSubsetKind::None && experts.active(e)) {
                        ++histogram[e];
                        ++hits;
                        local = true;
                    }
                }
                active += local;
            }
            const auto unique =
                std::count_if(histogram.begin(), histogram.end(), [](int n) { return n > 0; });
            const bool routed       = subset.kind != ops::ExpertSubsetKind::None;
            const auto weight_bytes = unique * experts.encoded_expert_bytes() +
                                      (routed ? router.model_weight_bytes() : 0) +
                                      gate_up.model_weight_bytes() + down.model_weight_bytes();
            const auto bytes =
                weight_bytes + 4ULL * ep::H * t + (routed ? ep::E * 4 + 8ULL * ep::K * t : 0);
            const double router_flops = routed ? 2.0 * ep::H * ep::E * t : 0;
            const double shared_flops = 6.0 * ep::H * 1024 * t;
            const double routed_flops = 6.0 * ep::H * ep::I * hits;
            const double flops        = router_flops + shared_flops + routed_flops;
            const double routed_peak =
                o.format == "nvfp4" ? kRoofline.nvfp4_f32_tflops : kRoofline.fp8_f32_tflops;
            const double absent   = std::numeric_limits<double>::quiet_NaN();
            const bool shared_tc  = o.shared_format == "q8" ? t >= 2 : t >= 17;
            const bool routed_tc  = hits == 0 || (o.format == "nvfp4" ? t >= 2 : t >= 3);
            const bool router_tc  = !routed || t >= 2;
            const bool calibrated = shared_tc && routed_tc && router_tc;
            const double compute_reference =
                calibrated ? (router_flops + shared_flops) / (kRoofline.bf16_f32_tflops * 1e6) +
                                 routed_flops / (routed_peak * 1e6)
                           : absent;
            const double selected_compute =
                calibrated ? (router_flops +
                              shared_flops * (o.shared_format == "q8" && t >= 49 ? 2 : 4.0 / 3)) /
                                     (kRoofline.bf16_f32_tflops * 1e6) +
                                 routed_flops / (routed_peak * 1e6)
                           : absent;
            const std::string arithmetic =
                std::string("shared:") +
                (shared_tc ? (o.shared_format == "q8" && t >= 49 ? "BF16_F32ACC_COMPENSATED_GU_DOWN"
                                                                 : "BF16_F32ACC_COMPENSATED_DOWN")
                           : (t == 1 ? "SIMT_FP32" : "BF16_F32ACC+TF32_F32ACC")) +
                (routed ? std::string(";router:") + (router_tc ? "BF16_F32ACC" : "SIMT_FP32") +
                              ";routed:" +
                              (hits == 0             ? "empty"
                               : !routed_tc          ? "SIMT_FP32"
                               : o.format == "nvfp4" ? "NVFP4_F32ACC"
                                                     : "FP8_F32ACC")
                        : "");
            // Explicit logical write+read minima, separate from operand traffic and DRAM.
            const std::uint64_t staging =
                8ULL * 1024 * t +
                (routed ? (16ULL * ep::K + 8ULL * ep::E) * t + 8ULL * ep::H * active : 0);
            measurement::Row row;
            row.op     = "topk_sigmoid_swiglu_moe";
            row.format = o.format;
            row.policy = o.format == "nvfp4" ? "a4" : "a8";
            row.n      = ep::H;
            row.k      = ep::I;
            row.tokens = t;
            row.workspace_bytes =
                ops::topk_sigmoid_swiglu_moe_workspace_capacity_bytes(profile, subset.kind, t, t);
            for (const auto& execution :
                 measurement::selections(o.timing.execution, "graph", "eager"))
                for (const auto& cache : measurement::selections(o.timing.cache, "cold", "warm")) {
                    workspace.reset_peak();
                    const auto result = measurement::measure(
                        [&](auto view) {
                            ops::topk_sigmoid_swiglu_moe(x, profile, weights, subset, y, workspace,
                                                         view);
                        },
                        [](auto) {}, flush, cuda, o.timing, execution, cache);
                    row.workspace_peak = workspace.peak_used();
                    if (workspace.used() != 0 || row.workspace_peak > row.workspace_bytes)
                        throw std::runtime_error("MoE workspace contract violated");
                    const auto fraction = [&](double work, double peak) {
                        return format_metric(work / (result.gpu.median_us * peak * 1e4), 3, "");
                    };
                    const double router_reference =
                        routed && router_tc ? kRoofline.bf16_f32_tflops : absent;
                    const double shared_reference = shared_tc ? kRoofline.bf16_f32_tflops : absent;
                    const double routed_reference = hits && routed_tc ? routed_peak : absent;
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
                            format_metric(cache == "cold"
                                              ? bytes / (kRoofline.sustained_read_gbs * 1000)
                                              : absent,
                                          3, "") +
                            ',' + format_metric(compute_reference, 3, "") + ',' + o.shared_format +
                            ',' + o.subset + ',' + arithmetic + ',' +
                            format_metric(selected_compute, 3, "") + ',' + std::to_string(staging) +
                            ',' + std::to_string(capacity) + ',' +
                            format_metric(router_reference, 3, "") + ',' +
                            format_metric(shared_reference, 3, "") + ',' +
                            format_metric(routed_reference, 3, "") + ',' +
                            fraction(router_flops, router_reference) + ',' +
                            fraction(shared_flops, shared_reference) + ',' +
                            fraction(routed_flops, routed_reference) + ',' +
                            format_metric(selected_compute / result.gpu.median_us * 100, 3, ""));
                }
        }
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto options = parse(argc, argv);
        measurement::hardware();
        if (options.shared_format == "q8") {
            auto gu   = make_row_split_weight(QType::Q8_G32_FP16, 2048, 4096, 4096, 2913U);
            auto down = make_row_split_weight(QType::Q8_G32_FP16, 4096, 1024, 1024, 2915U);
            run(options, gu, down);
        } else {
            auto gu   = make_direct_bf16_weight(2048, ep::H, 2913, .02f);
            auto down = make_direct_bf16_weight(ep::H, 1024, 2915, .02f);
            run(options, gu, down);
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
