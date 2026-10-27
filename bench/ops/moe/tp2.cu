#include "ops/moe/tp2_fixture.h"
#include "op_measurement.cuh"
#include "common/roofline.h"

#include <numeric>
#include <sstream>

using namespace ninfer;
using namespace ninfer::bench;
namespace tp = ninfer::test::moe_tp2;

namespace {
struct Options {
    measurement::Options timing;
    std::vector<int> tokens;
    std::string profile = "m1", distribution = "all", entry = "complete";
    int rank = 0;

    Options() {
        timing.cache = "cold";
        for (int t = 1; t <= 256; ++t) tokens.push_back(t);
        for (int t : {512, 1024, 2048, 4096}) tokens.push_back(t);
    }
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout
                << "TP2 MoE, all distinct expert payloads and verified router distributions.\n"
                   "  --profile-id m1|m2|m3 --entry complete|routed|both --rank 0|1\n"
                   "  --distribution reuse|pool|tail|spread|all\n"
                   "  --tokens T[,T...] --sweep FIRST:LAST[:STEP]\n"
                   "Default extents: every T=1..256, then 512,1024,2048,4096.\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (measurement::parse(o.timing, flag, value)) continue;
        if (flag == "--profile-id")
            o.profile = value;
        else if (flag == "--entry")
            o.entry = value;
        else if (flag == "--distribution")
            o.distribution = value;
        else if (flag == "--rank")
            o.rank = measurement::integer(value, 0, 1);
        else if (flag == "--tokens")
            o.tokens = measurement::integers(value);
        else if (flag == "--sweep")
            o.tokens = measurement::sweep(value);
        else
            throw std::invalid_argument("unknown TP2 benchmark option");
    }
    if ((o.profile != "m1" && o.profile != "m2" && o.profile != "m3") ||
        (o.entry != "complete" && o.entry != "routed" && o.entry != "both") ||
        (o.distribution != "all" && o.distribution != "reuse" && o.distribution != "pool" &&
         o.distribution != "tail" && o.distribution != "spread"))
        throw std::invalid_argument("invalid TP2 benchmark workload");
    return o;
}

void run(const Options& o) {
    const bool clamped = o.profile != "m3";
    const bool fp8     = o.profile == "m2";
    const tp::Shape shape =
        clamped ? tp::Shape{4096, 1024, 288,
                            8,    true, fp8 ? QType::FP8_E4M3FN_BLOCK128X128_BF16 : QType::NVFP4}
                : tp::Shape{2560, 320, 512, 10, false, QType::NVFP4};
    const auto cuda   = device_context().execution_view();
    const int maximum = *std::max_element(o.tokens.begin(), o.tokens.end());
    tp::Fixture fixture(cuda, shape, o.rank, maximum);
    const auto profile           = fixture.routed_profile();
    const auto full_profile      = clamped ? fixture.sigmoid_profile() : ops::SigmoidMoeProfile{};
    const auto softmax_profile   = clamped ? ops::MoeProfile{} : fixture.softmax_profile();
    const auto weights           = clamped ? fixture.sigmoid_weights() : ops::SigmoidMoeWeights{};
    const auto softmax_weights   = clamped ? ops::MoeWeights{} : fixture.softmax_weights();
    const auto experts           = fixture.experts();
    const auto complete_capacity = [&](int first, int last) {
        return clamped ? ops::topk_sigmoid_swiglu_moe_workspace_capacity_bytes(
                             full_profile, ops::ExpertSubsetKind::All, first, last)
                       : ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(
                             softmax_profile, ops::ExpertSubsetKind::All, ops::MoeEpilogue::Write,
                             first, last);
    };
    const auto full_capacity   = complete_capacity(1, maximum);
    const auto routed_capacity = ops::routed_swiglu_workspace_capacity_bytes(
        profile, ops::ExpertSubsetKind::All, DType::BF16, ops::MoeEpilogue::Write, 1, maximum);
    auto output  = make_buffer(std::size_t(shape.hidden) * maximum * 2);
    auto scratch = make_buffer(std::max(full_capacity, routed_capacity));
    MemoryArena workspace(scratch.span());
    L2FlushBuffer flush(o.timing.flush_bytes);
    measurement::Csv csv(o.timing.csv_out,
                         "profile,distribution,rank,unique_experts,expert_counts,encoded_expert_"
                         "bytes,effective_bytes,"
                         "effective_gb_s,projection_flops,router_flops,shared_flops,routed_flops,"
                         "bandwidth_reference_us,"
                         "compute_reference_us,roofline_reference_us,roofline_gap,arithmetic,"
                         "interval_capacity_bytes,"
                         "known_workspace_traffic_minimum_bytes,bandwidth_limited,bandwidth_pass");
    const auto distributions = o.distribution == "all"
                                   ? std::vector<std::string>{"reuse", "pool", "tail", "spread"}
                                   : std::vector<std::string>{o.distribution};
    const auto entries       = o.entry == "both" ? std::vector<std::string>{"complete", "routed"}
                                                 : std::vector<std::string>{o.entry};
    std::cerr << "target=" << kRoofline.target << " profile=" << o.profile
              << " sustained_read_gbs=" << kRoofline.sustained_read_gbs << '\n';
    for (const auto& distribution : distributions)
        for (int tokens : o.tokens) {
            fixture.input(tokens, distribution);
            const auto counts = fixture.check_routes(tokens, workspace, cuda);
            const int unique =
                std::count_if(counts.begin(), counts.end(), [](int c) { return c > 0; });
            std::ostringstream histogram;
            for (std::size_t e = 0; e < counts.size(); ++e) {
                if (e) histogram << ';';
                histogram << counts[e];
            }
            for (const auto& entry : entries) {
                const bool complete = entry == "complete";
                if (!complete) fixture.publish_routes(tokens);
                const auto x      = fixture.x(tokens);
                const auto routes = fixture.routes(tokens);
                Tensor y(output.span(), DType::BF16, {shape.hidden, tokens});
                const std::size_t bytes =
                    unique * fixture.encoded_expert_bytes() + 4ULL * shape.hidden * tokens +
                    (complete ? fixture.router_bytes() + fixture.shared_bytes() +
                                    (clamped ? shape.experts * 4ULL : 0)
                              : 8ULL * shape.top_k * tokens);
                const double router_flops =
                    complete ? 2.0 * shape.hidden * (shape.experts + !clamped) * tokens : 0;
                const double shared_flops =
                    complete ? 6.0 * shape.hidden * shape.intermediate * tokens : 0;
                const double routed_flops =
                    6.0 * shape.hidden * shape.intermediate * shape.top_k * tokens;
                const double memory_reference = bytes / (kRoofline.sustained_read_gbs * 1000);
                const double compute_reference =
                    (tokens <= (clamped ? (fp8 ? 2 : 1) : 3) || (complete && !clamped && tokens < 8))
                        ? std::numeric_limits<double>::quiet_NaN()
                        : (router_flops + shared_flops * (clamped ? (tokens <= 48 ? 4.0 / 3 : 2)
                                                                  : (tokens < 128 ? 4.0 / 3 : 1))) /
                                  (kRoofline.bf16_f32_tflops * 1e6) +
                              routed_flops /
                                  ((fp8 ? kRoofline.fp8_f32_tflops : kRoofline.nvfp4_f32_tflops) *
                                   1e6);
                const double roofline = std::isfinite(compute_reference)
                                            ? std::max(memory_reference, compute_reference)
                                            : memory_reference;
                const bool memory_limited =
                    !std::isfinite(compute_reference) || memory_reference >= compute_reference;
                const auto call = [&](auto execution) {
                    if (complete) {
                        if (clamped)
                            ops::topk_sigmoid_swiglu_moe(x, full_profile, weights, {}, y, workspace,
                                                         execution);
                        else
                            ops::topk_softmax_swiglu_moe(x, softmax_profile, softmax_weights, {},
                                                         ops::MoeEpilogue::Write, y, workspace,
                                                         execution);
                    } else
                        ops::routed_swiglu(x, profile, experts, routes, {}, ops::MoeEpilogue::Write,
                                           y, workspace, execution);
                };
                measurement::Row row;
                row.op              = complete
                                          ? (clamped ? "topk_sigmoid_swiglu_moe" : "topk_softmax_swiglu_moe")
                                          : "routed_swiglu";
                row.format          = fp8 ? "fp8" : "nvfp4";
                row.policy          = fp8 ? "a8" : "a4";
                row.n               = shape.hidden;
                row.k               = shape.intermediate;
                row.tokens          = tokens;
                row.workspace_bytes = complete
                                          ? complete_capacity(tokens, tokens)
                                          : ops::routed_swiglu_workspace_capacity_bytes(
                                                profile, ops::ExpertSubsetKind::All, DType::BF16,
                                                ops::MoeEpilogue::Write, tokens, tokens);
                // Quantized input/product publication and reduction scratch are separate
                // from the unique public operands. This is a minimum, not measured DRAM.
                const bool direct = tokens <= (clamped ? (fp8 ? 2 : 1) : 3);
                const auto elements =
                    std::uint64_t(tokens) * (shape.hidden + shape.intermediate * shape.top_k);
                const int group        = fp8 ? 128 : 16;
                const auto scale_words = std::uint64_t(tokens) *
                                         ((shape.hidden + group - 1) / group +
                                          shape.top_k * ((shape.intermediate + group - 1) / group));
                const std::uint64_t staging =
                    (complete ? (16ULL * shape.top_k + 8ULL * shape.hidden) * tokens : 0) +
                    (direct ? 8ULL * shape.intermediate * shape.top_k * tokens
                            : 2 * (elements / (fp8 ? 1 : 2) + scale_words * (fp8 ? 4 : 1)));
                for (const auto& execution :
                     measurement::selections(o.timing.execution, "graph", "eager"))
                    for (const auto& cache :
                         measurement::selections(o.timing.cache, "cold", "warm")) {
                        workspace.reset_peak();
                        const auto result = measurement::measure(
                            call, [](auto) {}, flush, cuda, o.timing, execution, cache);
                        row.workspace_peak     = workspace.peak_used();
                        const double bandwidth = bytes / (result.gpu.median_us * 1000);
                        std::ostringstream extra;
                        extra << o.profile << ',' << distribution << ',' << o.rank << ',' << unique
                              << ',' << histogram.str() << ',' << fixture.encoded_expert_bytes()
                              << ',' << bytes << ',' << bandwidth << ','
                              << router_flops + shared_flops + routed_flops << ',' << router_flops
                              << ',' << shared_flops << ',' << routed_flops << ','
                              << memory_reference << ',' << format_metric(compute_reference, 3, "")
                              << ',' << roofline << ',' << result.gpu.median_us / roofline << ','
                              << (direct
                                      ? (complete && clamped && tokens > 1 ? "BF16_F32ACC+SIMT_FP32"
                                                                           : "SIMT_FP32")
                                  : complete && !clamped && tokens < 8
                                      ? (fp8 ? "SIMT_FP32+FP8_F32ACC" : "SIMT_FP32+NVFP4_F32ACC")
                                  : complete ? (fp8 ? "BF16_F32ACC+FP8_F32ACC"
                                                    : "BF16_F32ACC+NVFP4_F32ACC")
                                             : (fp8 ? "FP8_F32ACC" : "NVFP4_F32ACC"))
                              << ',' << (complete ? full_capacity : routed_capacity) << ','
                              << staging << ',' << memory_limited << ','
                              << (cache == "cold" && memory_limited ? bandwidth >= 235 : 0);
                        csv.write(row, result, execution, cache, extra.str());
                    }
            }
        }
}
} // namespace

int main(int argc, char** argv) {
    try {
        run(parse(argc, argv));
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
