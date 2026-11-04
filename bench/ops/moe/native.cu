#include "ops/moe/native_fixture.h"
#include "op_measurement.cuh"
#include "common/roofline.h"

#include <numeric>
#include <sstream>

using namespace ninfer;
using namespace ninfer::bench;
namespace native = ninfer::test::native_moe;

namespace {
struct Options {
    measurement::Options timing;
    std::vector<int> tokens;
    native::Profile profile            = native::Profile::MainTp;
    native::ScalePattern scale_pattern = native::ScalePattern::Typical;
    std::string distribution = "all", entry = "both", locality = "mixed";
    int rank = 0;

    Options() {
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
                << "Native FP4Local MoE, complete public entries.\n"
                   "  --profile-id main-tp|main-ep|mtp-tp|mtp-ep --entry routed|complete|both\n"
                   "  --distribution reuse|pool|tail|spread|all --locality local|mixed|remote\n"
                   "  --scale-pattern typical|stress (CSF; default typical)\n"
                   "  --rank 0|1 --tokens T[,T...] --sweep FIRST:LAST[:STEP]\n"
                   "Defaults: every T=1..256, then 512,1024,2048,4096; warm and cold.\n"
                   "Bytes count distinct encoded weights and public operands, not measured DRAM.\n"
                   "Compute bounds use allowed arithmetic; they do not identify the selected "
                   "kernel.\n";
            measurement::help();
            std::exit(0);
        }
        const auto value = measurement::take(argc, argv, i);
        if (measurement::parse(o.timing, flag, value)) continue;
        if (flag == "--profile-id")
            o.profile = native::parse_profile(value);
        else if (flag == "--scale-pattern") {
            if (value == "typical")
                o.scale_pattern = native::ScalePattern::Typical;
            else if (value == "stress")
                o.scale_pattern = native::ScalePattern::Stress;
            else
                throw std::invalid_argument("invalid CSF scale pattern");
        } else if (flag == "--entry")
            o.entry = value;
        else if (flag == "--distribution")
            o.distribution = value;
        else if (flag == "--locality")
            o.locality = value;
        else if (flag == "--rank")
            o.rank = measurement::integer(value, 0, 1);
        else if (flag == "--tokens")
            o.tokens = measurement::integers(value);
        else if (flag == "--sweep")
            o.tokens = measurement::sweep(value);
        else
            throw std::invalid_argument("unknown native MoE benchmark option");
    }
    if ((o.entry != "routed" && o.entry != "complete" && o.entry != "both") ||
        (o.locality != "local" && o.locality != "mixed" && o.locality != "remote") ||
        (o.distribution != "all" && o.distribution != "reuse" && o.distribution != "pool" &&
         o.distribution != "tail" && o.distribution != "spread"))
        throw std::invalid_argument("invalid native MoE benchmark workload");
    return o;
}

void run(const Options& o) {
    const auto cuda    = device_context().execution_view();
    const auto maximum = *std::max_element(o.tokens.begin(), o.tokens.end());
    native::Fixture fixture(cuda, o.profile, maximum, o.rank, o.scale_pattern);
    const auto routed          = fixture.routed_profile();
    const auto complete        = fixture.complete_profile();
    const auto experts         = fixture.expert_weights();
    const auto weights         = fixture.complete_weights();
    const auto subset          = fixture.subset();
    const auto routed_capacity = [&](int first, int last) {
        return ops::routed_swiglu_workspace_capacity_bytes(routed, subset.kind, DType::BF16,
                                                           ops::MoeEpilogue::Write, first, last);
    };
    const auto complete_capacity = [&](int first, int last) {
        return ops::topk_sigmoid_swiglu_moe_workspace_capacity_bytes(complete, subset.kind, first,
                                                                     last);
    };
    const auto full_bytes   = o.entry == "routed" ? 0 : complete_capacity(1, maximum);
    const auto routed_bytes = o.entry == "complete" ? 0 : routed_capacity(1, maximum);
    DeviceBuffer output(cuda.device(), std::size_t(native::Fixture::hidden) * maximum * 2);
    DeviceBuffer scratch(cuda.device(), std::max(full_bytes, routed_bytes));
    MemoryArena workspace(scratch.span());
    fixture.release_host_weights();
    L2FlushBuffer flush(o.timing.flush_bytes);
    measurement::Csv csv(
        o.timing.csv_out,
        "profile,scale_pattern,distribution,locality,rank,local_assignments,unique_local_experts,"
        "expert_counts,"
        "routed_weight_bytes,shared_weight_bytes,router_weight_bytes,index_calibration_bytes,"
        "unique_operand_bytes,operand_gb_s,routed_flops,shared_flops,router_flops,"
        "routed_memory_bound_us,shared_memory_bound_us,router_memory_bound_us,"
        "memory_bound_us,routed_allowed_compute_bound_us,shared_allowed_compute_bound_us,"
        "router_allowed_compute_bound_us,allowed_compute_bound_us,roofline_bound_us,"
        "latency_above_bound_us,allowed_arithmetic,interval_workspace_bytes,bandwidth_limited,"
        "cold_bandwidth_235gbs");
    const auto distributions = o.distribution == "all"
                                   ? std::vector<std::string>{"reuse", "pool", "tail", "spread"}
                                   : std::vector<std::string>{o.distribution};
    const auto entries       = o.entry == "both" ? std::vector<std::string>{"routed", "complete"}
                                                 : std::vector<std::string>{o.entry};
    std::cerr << "target=" << kRoofline.target << " profile=" << native::profile_name(o.profile)
              << " sustained_read_gbs=" << kRoofline.sustained_read_gbs << '\n';
    for (const auto& distribution : distributions)
        for (int tokens : o.tokens) {
            fixture.input(tokens, distribution, 0, true, o.locality);
            fixture.publish_routes();
            const auto counts = fixture.local_counts();
            const int active  = std::accumulate(counts.begin(), counts.end(), 0);
            const int unique =
                std::count_if(counts.begin(), counts.end(), [](int count) { return count > 0; });
            std::ostringstream histogram;
            for (int e = 0; e < int(counts.size()); ++e) {
                if (e) histogram << ';';
                histogram << counts[e];
            }
            const auto routed_weight = fixture.active_weight_bytes();
            const std::uint64_t index_bytes =
                std::uint64_t(unique) *
                ((fixture.main() ? 8 : 0) + (fixture.ep() ? (fixture.main() ? 8 : 16) : 0));
            for (const auto& entry : entries) {
                const bool full                   = entry == "complete";
                const std::uint64_t shared_weight = full ? fixture.shared_bytes() : 0;
                const std::uint64_t router_weight = full ? fixture.router_bytes() : 0;
                const std::uint64_t public_bytes =
                    std::uint64_t(tokens) *
                    (4 * native::Fixture::hidden + (full ? 0 : 8 * native::Fixture::top_k));
                const auto bytes =
                    routed_weight + shared_weight + router_weight + index_bytes + public_bytes;
                const double routed_flops =
                    6.0 * active * native::Fixture::hidden * fixture.intermediate;
                const double shared_flops = full ? 6.0 * tokens * native::Fixture::hidden *
                                                       native::Fixture::shared_intermediate
                                                 : 0;
                const double router_flops =
                    full ? 2.0 * tokens * native::Fixture::hidden * native::Fixture::experts_count
                         : 0;
                const double routed_compute =
                    routed_flops / (1e6 * (fixture.main() ? kRoofline.nvfp4_f32_tflops
                                                          : kRoofline.bf16_f32_tflops));
                const double shared_compute =
                    shared_flops / (1e6 * (fixture.main() ? kRoofline.bf16_f32_tflops
                                                          : kRoofline.mxfp8_f32_tflops));
                const double router_compute = router_flops / (1e6 * kRoofline.bf16_f32_tflops);
                const double compute        = routed_compute + shared_compute + router_compute;
                const double memory         = bytes / (1000 * kRoofline.sustained_read_gbs);
                const double routed_memory  = (routed_weight + index_bytes + public_bytes) /
                                             (1000 * kRoofline.sustained_read_gbs);
                const double shared_memory = shared_weight / (1000 * kRoofline.sustained_read_gbs);
                const double router_memory = router_weight / (1000 * kRoofline.sustained_read_gbs);
                const double bound         = std::max(routed_memory, routed_compute) +
                                     std::max(shared_memory, shared_compute) +
                                     std::max(router_memory, router_compute);
                const bool memory_limited = routed_memory >= routed_compute &&
                                            shared_memory >= shared_compute &&
                                            router_memory >= router_compute;
                const auto x      = fixture.x(tokens);
                const auto routes = fixture.routes(tokens);
                auto y = Tensor(output.span(), DType::BF16, {native::Fixture::hidden, tokens});
                const auto launch = [&](CudaExecutionView execution) {
                    if (full)
                        ops::topk_sigmoid_swiglu_moe(x, complete, weights, subset, y, workspace,
                                                     execution);
                    else
                        ops::routed_swiglu(x, routed, experts, routes, subset,
                                           ops::MoeEpilogue::Write, y, workspace, execution);
                };
                measurement::Row row;
                row.op     = full ? "topk_sigmoid_swiglu_moe" : "routed_swiglu";
                row.format = fixture.main() ? "nvfp4_mul_csf32" : "nvfp4_mul_raw";
                row.policy = fixture.main() ? "a4" : full ? "a16+a8" : "a16";
                row.n      = native::Fixture::hidden;
                row.k      = fixture.intermediate;
                row.tokens = tokens;
                row.workspace_bytes =
                    full ? complete_capacity(tokens, tokens) : routed_capacity(tokens, tokens);
                for (const auto& execution :
                     measurement::selections(o.timing.execution, "graph", "eager"))
                    for (const auto& cache :
                         measurement::selections(o.timing.cache, "cold", "warm")) {
                        workspace.reset_peak();
                        const auto result = measurement::measure(
                            launch, [](auto) {}, flush, cuda, o.timing, execution, cache);
                        row.workspace_peak     = workspace.peak_used();
                        const double bandwidth = bytes / (1000 * result.gpu.median_us);
                        std::ostringstream extra;
                        extra << native::profile_name(o.profile) << ','
                              << (fixture.main() ? (o.scale_pattern == native::ScalePattern::Typical
                                                        ? "typical"
                                                        : "stress")
                                                 : "raw")
                              << ',' << distribution << ',' << o.locality << ',' << o.rank << ','
                              << active << ',' << unique << ',' << histogram.str() << ','
                              << routed_weight << ',' << shared_weight << ',' << router_weight
                              << ',' << index_bytes << ',' << bytes << ',' << bandwidth << ','
                              << routed_flops << ',' << shared_flops << ',' << router_flops << ','
                              << routed_memory << ',' << shared_memory << ',' << router_memory
                              << ',' << memory << ',' << routed_compute << ',' << shared_compute
                              << ',' << router_compute << ',' << compute << ',' << bound << ','
                              << result.gpu.median_us - bound << ','
                              << (fixture.main() ? (full ? "NVFP4+BF16" : "NVFP4")
                                                 : (full ? "BF16+MXFP8" : "BF16"))
                              << ',' << (full ? full_bytes : routed_bytes) << ',' << memory_limited
                              << ',';
                        if (cache == "cold" && memory_limited) extra << (bandwidth >= 235);
                        csv.write(row, result, execution, cache, extra.str());
                    }
            }
        }
}
} // namespace

int main(int argc, char** argv) {
    try {
        run(parse(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
