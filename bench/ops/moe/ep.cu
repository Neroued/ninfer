#include "ops/moe/ep_fixture.h"
#include "op_measurement.cuh"

#include <numeric>

using namespace ninfer;

namespace {
namespace ep = ninfer::test::moe_ep;
namespace bm = ninfer::bench::measurement;
using namespace ninfer::bench;

struct Options {
    bm::Options timing;
    std::vector<int> tokens{1, 2, 4, 6, 8, 16, 32, 64, 128, 512, 1024, 2048, 4096};
    std::string distribution = "mixed", placement = "contiguous", side = "0", format = "nvfp4";
    std::string operation   = "routed";
    int shared_intermediate = 320;
};

Options parse(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help") {
            std::cout << "512 logical experts, 256 physical payloads, global TopK10.\n"
                         "--format nvfp4|fp8 --tokens T[,T...]\n"
                         "--operation routed|routed-shared|complete|decomposed\n"
                         "--shared-intermediate 320|640 --distribution zero|all|mixed|skew|sweep\n"
                         "--placement contiguous|interleaved|both --side 0|1|both\n";
            bm::help();
            std::exit(0);
        }
        const auto v = bm::take(argc, argv, i);
        if (bm::parse(o.timing, flag, v)) continue;
        if (flag == "--format")
            o.format = v;
        else if (flag == "--tokens")
            o.tokens = bm::integers(v);
        else if (flag == "--operation")
            o.operation = v;
        else if (flag == "--shared-intermediate")
            o.shared_intermediate = bm::integer(v);
        else if (flag == "--distribution")
            o.distribution = v;
        else if (flag == "--placement")
            o.placement = v;
        else if (flag == "--side")
            o.side = v;
        else
            throw std::invalid_argument("unknown EP option");
    }
    if ((o.format != "nvfp4" && o.format != "fp8") ||
        (o.operation != "routed" && o.operation != "routed-shared" && o.operation != "complete" &&
         o.operation != "decomposed") ||
        (o.shared_intermediate != 320 && o.shared_intermediate != 640) ||
        (o.distribution != "zero" && o.distribution != "all" && o.distribution != "mixed" &&
         o.distribution != "skew" && o.distribution != "sweep") ||
        (o.placement != "contiguous" && o.placement != "interleaved" && o.placement != "both") ||
        (o.side != "0" && o.side != "1" && o.side != "both"))
        throw std::invalid_argument("invalid EP options");
    return o;
}

Weight dense(const DeviceBuffer& buffer, int n, int k) {
    Weight w;
    w.device  = buffer.device();
    w.payload = w.qdata = buffer.data();
    w.payload_bytes     = buffer.size();
    w.qtype             = QType::BF16;
    w.layout            = QuantLayout::Contiguous;
    w.ndim              = 2;
    w.n = w.shape[0] = w.padded_shape[0] = n;
    w.k = w.shape[1] = w.padded_shape[1] = k;
    return w;
}

struct ControlShared {
    DeviceBuffer control, gu, down;
    ops::MoeWeights weights;

    ControlShared(int intermediate)
        : control(make_buffer((ep::E + 1) * ep::H * 2)),
          gu(make_bf16(2 * intermediate * ep::H, 991, -.012F, .012F)),
          down(make_bf16(ep::H * intermediate, 1011, -.025F, .025F)) {
        std::vector<std::uint16_t> values((ep::E + 1) * ep::H);
        for (int e = 0; e <= ep::E; ++e) values[e * ep::H + e] = f32_to_bf16(2);
        control.copy_from_host(values.data(), control.size());
        weights.control = dense(control, ep::E + 1, ep::H);
        weights.shared = {{dense(gu, 2 * intermediate, ep::H)}, {dense(down, ep::H, intermediate)}};
    }
};
} // namespace

int main(int argc, char** argv) try {
    const auto o = parse(argc, argv);
    bm::hardware();
    auto& device = device_context();
    L2FlushBuffer flush(o.timing.flush_bytes);
    ControlShared dense_weights(o.shared_intermediate);
    bm::Csv csv(o.timing.csv_out,
                "placement,side,distribution,shared_intermediate,local_payloads,local_assignments,"
                "active_tokens,unique_experts,bank_backing_bytes,unique_weight_bytes,"
                "logical_bytes,useful_flops,useful_TFLOPs,logical_GBps,memory_lower_bound_us");
    for (const auto& placement : bm::selections(o.placement, "contiguous", "interleaved")) {
        for (const auto& side : bm::selections(o.side, "0", "1")) {
            ep::Weights weights(device.device(),
                                placement == "contiguous" ? ep::Placement::Contiguous
                                                          : ep::Placement::Interleaved,
                                std::stoi(side), o.format == "fp8");
            weights.release_host_oracle();
            auto full   = dense_weights.weights;
            full.routed = weights.experts();
            const ops::MoeProfile profile{weights.profile(), ops::swiglu_profile(full.shared)};
            const auto distributions =
                o.distribution == "sweep" ? std::vector<std::string>{"zero", "all", "mixed", "skew"}
                                          : std::vector<std::string>{o.distribution};
            for (const auto& distribution : distributions) {
                for (int m : o.tokens) {
                    auto routes = ep::make_routes(weights, m, distribution.c_str());
                    auto input  = make_bf16(std::size_t(ep::H) * m, 871, -.7F, .7F);
                    std::vector<std::uint16_t> values(std::size_t(ep::H) * m);
                    input.copy_to_host(values.data(), input.size());
                    std::vector<float> beta(m, .5F);
                    for (int t = 0; t < m; ++t) {
                        std::fill_n(values.begin() + std::size_t(t) * ep::H, ep::E,
                                    f32_to_bf16(-.75F));
                        values[std::size_t(t) * ep::H + ep::E] = 0;
                        double denominator                     = 0;
                        for (int j = 0; j < ep::K; ++j) denominator += std::exp(-j / 16.);
                        for (int j = 0; j < ep::K; ++j) {
                            values[std::size_t(t) * ep::H + routes.ids[t * ep::K + j]] =
                                f32_to_bf16(.75F - j / 32.F);
                            routes.coefficients[t * ep::K + j] = std::exp(-j / 16.) / denominator;
                        }
                    }
                    input.copy_from_host(values.data(), input.size());
                    auto ids    = bm::make_i32(routes.ids),
                         coeff  = make_buffer(routes.coefficients.size() * 4),
                         shared = make_buffer(m * 4);
                    coeff.copy_from_host(routes.coefficients.data(), coeff.size());
                    shared.copy_from_host(beta.data(), shared.size());
                    auto output = make_buffer(std::size_t(ep::H) * m * 2);
                    Tensor x(input.span(), DType::BF16, {ep::H, m}),
                        y(output.span(), DType::BF16, {ep::H, m}),
                        b(shared.span(), DType::FP32, {m});
                    ops::ExpertRoutes selected{Tensor(ids.span(), DType::I32, {ep::K, m}),
                                               Tensor(coeff.span(), DType::FP32, {ep::K, m})};
                    const bool has_shared = o.operation != "routed";
                    auto bytes            = has_shared
                                                ? ops::routed_swiglu_workspace_capacity_bytes(
                                           profile.routed, profile.shared, weights.subset().kind,
                                           DType::BF16, ops::MoeEpilogue::Write, m, m)
                                                : ops::routed_swiglu_workspace_capacity_bytes(
                                           profile.routed, weights.subset().kind, DType::BF16,
                                           ops::MoeEpilogue::Write, m, m);
                    if (o.operation == "complete")
                        bytes = ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(
                            profile, weights.subset().kind, ops::MoeEpilogue::Write, m, m);
                    if (o.operation == "decomposed")
                        bytes = std::max(bytes, ops::topk_softmax_route_workspace_capacity_bytes(
                                                    {ep::H, ep::E, ep::K}, m, m));
                    auto scratch = make_buffer(bytes);
                    MemoryArena arena(scratch.span());
                    const auto launch = [&](CudaExecutionView cuda) {
                        if (o.operation == "complete") {
                            ops::topk_softmax_swiglu_moe(x, profile, full, weights.subset(),
                                                         ops::MoeEpilogue::Write, y, arena, cuda);
                        } else {
                            if (o.operation == "decomposed")
                                ops::topk_softmax_route(x, {ep::H, ep::E, ep::K}, full.control,
                                                        selected, b, arena, cuda);
                            if (has_shared)
                                ops::routed_swiglu(x, profile.routed, full.routed, selected,
                                                   weights.subset(), {full.shared, b},
                                                   ops::MoeEpilogue::Write, y, arena, cuda);
                            else
                                ops::routed_swiglu(x, profile.routed, full.routed, selected,
                                                   weights.subset(), ops::MoeEpilogue::Write, y,
                                                   arena, cuda);
                        }
                    };
                    const auto unique =
                        std::count_if(routes.per_expert.begin(), routes.per_expert.end(),
                                      [](int n) { return n > 0; });
                    const std::uint64_t weight_bytes =
                        unique * weights.encoded_expert_bytes() +
                        (has_shared ? 6ULL * ep::H * o.shared_intermediate : 0) +
                        (o.operation == "complete" || o.operation == "decomposed"
                             ? 2ULL * ep::H * (ep::E + 1)
                             : 0);
                    const std::uint64_t logical =
                        weight_bytes + 4ULL * ep::H * m + 8ULL * m * ep::K + 4ULL * ep::E;
                    const std::uint64_t flops =
                        routes.hits * (6ULL * ep::H * ep::I) +
                        (has_shared ? 6ULL * ep::H * o.shared_intermediate * m : 0) +
                        (o.operation == "complete" || o.operation == "decomposed"
                             ? 2ULL * ep::H * (ep::E + 1) * m
                             : 0);
                    bm::Row row;
                    row.op              = o.operation + "-ep";
                    row.format          = o.format;
                    row.policy          = o.format == "fp8" ? "a8" : "a4";
                    row.n               = ep::H;
                    row.k               = ep::I;
                    row.tokens          = m;
                    row.valid           = distribution;
                    row.workspace_bytes = bytes;
                    for (const auto& execution :
                         bm::selections(o.timing.execution, "graph", "eager")) {
                        for (const auto& cache : bm::selections(o.timing.cache, "cold", "warm")) {
                            const auto result = bm::measure(
                                launch, [](auto) {}, flush, device.execution_view(), o.timing,
                                execution, cache);
                            row.workspace_peak = arena.peak_used();
                            std::ostringstream extra;
                            extra << placement << ',' << side << ',' << distribution << ','
                                  << o.shared_intermediate << ',' << ep::Local << ',' << routes.hits
                                  << ',' << routes.active_tokens << ',' << unique << ','
                                  << weights.backing_bytes() << ',' << weight_bytes << ','
                                  << logical << ',' << flops << ','
                                  << flops / result.gpu.median_us / 1e6 << ','
                                  << logical / result.gpu.median_us / 1000 << ','
                                  << logical / (kRoofline.memory_gbs * 1000);
                            csv.write(row, result, execution, cache, extra.str());
                        }
                    }
                }
            }
        }
    }
} catch (const std::exception& e) {

    std::cerr << e.what() << '\n';
    return 1;
}
