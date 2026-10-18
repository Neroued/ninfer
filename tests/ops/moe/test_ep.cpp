#include "ops/moe/ep_fixture.h"
#include "ops/op_tester.h"
#include "ops/suite.h"
#include <array>
#include <numeric>

namespace ninfer::test::moe_ep {
namespace {
constexpr ReductionCriterion kA8{.08, 1e-7, .12};
constexpr ReductionCriterion kA4{.25, 1e-7, .35};

struct Case {
    int tokens;
    std::string distribution;
    int hits      = -1;
    bool graph    = false;
    bool complete = false;
};

struct ControlShared {
    DeviceBuffer control, gu, down;
    ops::MoeWeights weights;

    explicit ControlShared(Device device)
        : control(device, (E + 1) * H * 2), gu(device, 2 * 320 * H * 2), down(device, H * 320 * 2) {
        std::vector<float> c((E + 1) * H), g(2 * 320 * H), d(H * 320);
        for (int e = 0; e <= E; ++e) c[e * H + e] = 2;
        for (int i = 0; i < 320; ++i) {
            g[i * H + E + 1 + i]         = 1;
            g[(320 + i) * H + E + 1 + i] = -.5F;
            d[i * 320 + i]               = .125F;
        }
        auto publish = [](DeviceBuffer& buffer, const std::vector<float>& values, int n, int k) {
            std::vector<std::uint16_t> bits(values.size());
            std::transform(values.begin(), values.end(), bits.begin(), f32_to_bf16);
            buffer.copy_from_host(bits.data(), buffer.size());
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
        };
        weights.control = publish(control, c, E + 1, H);
        weights.shared  = {{publish(gu, g, 640, H)}, {publish(down, d, H, 320)}};
    }
};

int run(Weights& w, CudaExecutionView execution, const Case& c,
        const std::array<std::vector<float>, 3>& patterns, const ControlShared& fused) {
    const int m    = c.tokens;
    const bool fp8 = w.profile().ffn.gate_up_type == QType::FP8_E4M3FN_BLOCK128X128_BF16;
    auto routes    = make_routes(w, m, c.distribution.c_str(), c.hits);
    std::vector<float> input(std::size_t(m) * H);
    for (int t = 0; t < m; ++t)
        std::copy(patterns[t % 3].begin(), patterns[t % 3].end(),
                  input.begin() + std::size_t(t) * H);
    auto dx = to_device_bf16(input), di = to_device(routes.ids),
         da = to_device(routes.coefficients);
    Tensor x(dx.span(), DType::BF16, {H, m});
    ops::ExpertRoutes selected{Tensor(di.span(), DType::I32, {K, m}),
                               Tensor(da.span(), DType::FP32, {K, m})};
    auto weights   = fused.weights;
    weights.routed = w.experts();
    const ops::MoeProfile profile{w.profile(), ops::swiglu_profile(weights.shared)};
    const auto capacity = [&](int first, int last) {
        return c.complete ? ops::topk_softmax_swiglu_moe_workspace_capacity_bytes(
                                profile, w.subset().kind, ops::MoeEpilogue::Write, first, last)
                          : ops::routed_swiglu_workspace_capacity_bytes(
                                w.profile(), w.subset().kind, DType::BF16, ops::MoeEpilogue::Write,
                                first, last);
    };
    const auto bytes = capacity(m, m), bound = capacity(1, m);
    if (bytes > bound) throw std::runtime_error("EP interval capacity smaller than exact capacity");
    GuardedDeviceBuffer scratch(bytes), result(input.size() * 2);
    MemoryArena arena(scratch.span());
    Tensor y(result.span(), DType::BF16, {H, m});
    auto invoke = [&] {
        if (c.complete) {
            ops::topk_softmax_swiglu_moe(x, profile, weights, w.subset(), ops::MoeEpilogue::Write,
                                         y, arena, execution);
        } else {
            ops::routed_swiglu(x, w.profile(), w.experts(), selected, w.subset(),
                               ops::MoeEpilogue::Write, y, arena, execution);
        }
    };
    cudaGraph_t graph    = nullptr;
    cudaGraphExec_t exec = nullptr;
    if (c.graph) {
        cuda_synchronize();
        cuda_check(cudaStreamBeginCapture(execution.stream(), cudaStreamCaptureModeThreadLocal),
                   "capture");
        invoke();
        cuda_check(cudaStreamEndCapture(execution.stream(), &graph), "end capture");
        cuda_check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");
    }
    std::array<std::array<int, K>, 3> global_ids;
    std::array<std::array<double, K>, 3> global_coefficients;
    for (int p = 0; p < 3; ++p) {
        std::vector<int> order(E);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return patterns[p][a] > patterns[p][b]; });
        double denominator = 0;
        for (int j = 0; j < K; ++j)
            denominator += std::exp(2. * (patterns[p][order[j]] - patterns[p][order[0]]));
        for (int j = 0; j < K; ++j) {
            global_ids[p][j] = order[j];
            global_coefficients[p][j] =
                std::exp(2. * (patterns[p][order[j]] - patterns[p][order[0]])) / denominator;
        }
    }
    int failures = 0;
    for (int replay = 0; replay < (c.graph ? 3 : 1); ++replay) {
        // Change route, mask and global-ID index contents under the captured envelope.
        w.publish(replay == 1 ? 1 : 0, replay == 2);
        routes = make_routes(w, m, c.distribution.c_str(), c.hits);
        if (c.distribution == "jobs2" || c.distribution == "jobs3") {
            const int limit = c.distribution == "jobs2" ? 256 : 257;
            for (int t = 0; t < m; ++t)
                for (int j = 0; j < K; ++j)
                    routes.ids[std::size_t(t) * K + j] =
                        j == 0 ? (t < limit ? w.local_id(0) : w.remote_id(0))
                               : w.local_id(1 + (t * 13 + j * 97) % 255);
        }
        if (replay == 1) {
            for (int t = 0; t < m; ++t) {
                auto first = routes.ids.begin() + std::size_t(t) * K;
                std::rotate(first, first + 3, first + K);
            }
        }
        // Replay two tokens with no, partial, then full logical-expert overlap.
        if (fp8 && m == 2 && c.graph) {
            if (replay == 1)
                for (int j = 0; j < 3; ++j) routes.ids[K + j] = routes.ids[7 + j];
            if (replay == 2) std::copy_n(routes.ids.begin(), K, routes.ids.begin() + K);
        }
        di.copy_from_host(routes.ids.data(), di.size());
        da.copy_from_host(routes.coefficients.data(), da.size());
        result.fill(0x7f);
        if (c.graph)
            cuda_check(cudaGraphLaunch(exec, execution.stream()), "replay");
        else
            invoke();
        cuda_synchronize(execution.stream());
        std::vector<double> reference(input.size());
        for (int t = 0; t < m; ++t) {
            int hits = 0;
            for (int j = 0; j < K; ++j) {
                const auto a = std::size_t(t) * K + j;
                const int e  = c.complete ? global_ids[t % 3][j] : routes.ids[a];
                if (!w.active(e)) continue;
                ++hits;
                const auto& v = w.oracle(t % 3, e, patterns[t % 3]);
                for (int h = 0; h < H; ++h)
                    reference[std::size_t(t) * H + h] +=
                        (c.complete ? global_coefficients[t % 3][j]
                                    : double(routes.coefficients[a])) *
                        v[h];
            }
            if (!hits && !c.complete) {
                const auto words = from_device<std::uint16_t>(
                    static_cast<const std::uint16_t*>(result.data()) + std::size_t(t) * H, H);
                if (std::any_of(words.begin(), words.end(), [](auto v) { return v != 0; }))
                    ++failures;
            }
        }
        if (c.complete) {
            for (int t = 0; t < m; ++t) {
                const auto& pattern = patterns[t % 3];
                const double beta   = 1 / (1 + std::exp(-2. * pattern[E]));
                for (int i = 0; i < 320; ++i) {
                    const double g = pattern[E + 1 + i], u = -.5 * g;
                    reference[std::size_t(t) * H + i] += beta * .125 * g / (1 + std::exp(-g)) * u;
                }
            }
        }
        const std::string label = "EP256 m=" + std::to_string(m) + " " + c.distribution +
                                  " hits=" + std::to_string(c.hits) +
                                  " replay=" + std::to_string(replay);
        failures += verify_reduction(label, from_device_bf16(result.data(), input.size()),
                                     reference, fp8 ? kA8 : kA4);
        failures +=
            scratch.verify_guards(label) + result.verify_guards(label) + w.verify_controls();
        if (arena.used() != 0 || arena.peak_used() != bytes) {
            std::cerr << label << " capacity/arena mismatch\n";
            ++failures;
        }
        if (from_device<int>(di, routes.ids.size()) != routes.ids ||
            from_device<float>(da, routes.coefficients.size()) != routes.coefficients ||
            from_device_bf16(dx, input.size()) != std::vector<double>(input.begin(), input.end()))
            ++failures;
    }
    if (exec) cuda_check(cudaGraphExecDestroy(exec), "destroy exec");
    if (graph) cuda_check(cudaGraphDestroy(graph), "destroy graph");
    w.publish();
    return failures;
}
} // namespace
} // namespace ninfer::test::moe_ep

int main(int argc, char** argv) {
    using namespace ninfer;
    using namespace ninfer::test;
    using namespace ninfer::test::moe_ep;
    OpSuite suite;
    for (bool fp8 : {false, true}) {
        const std::string format = fp8 ? "fp8" : "nvfp4";
        for (auto p : {Placement::Contiguous, Placement::Interleaved})
            for (int side : {0, 1}) {
                std::vector<OpCase> catalogue;
                std::vector<Case> cases;
                auto add = [&](int m, std::string dist, int hits = -1, bool graph = false,
                               bool complete = false) {
                    const std::string id =
                        "ep/" + format + (complete ? "/complete/" : "/routed/") +
                        std::string(p == Placement::Contiguous ? "contiguous" : "interleaved") +
                        "/side" + std::to_string(side) + "/m" + std::to_string(m) + "/" + dist +
                        "/h" + std::to_string(hits) + (graph ? "/graph" : "/eager");
                    catalogue.push_back(
                        {id, {{"format", format}, {"execution", graph ? "graph" : "eager"}}});
                    cases.push_back({m, std::move(dist), hits, graph, complete});
                };
                for (int m : {1, 2, 16, 65, 128, 512}) add(m, "mixed", -1, true);
                if (p == Placement::Contiguous && side == 0) {
                    for (int hits : {0, 1, 5, 10}) add(1, "mixed", hits);
                    for (int m : {3,   7,   8,   9,   15,  17,  31,   32,   33,   63,   64,   127,
                                  129, 255, 256, 257, 511, 513, 1023, 1024, 1025, 2048, 4096, 4097})
                        add(m, "all");
                    if (fp8)
                        for (int m : {3071, 3072, 3073}) add(m, "all");
                    for (int m : {8, 128, 4096}) {
                        add(m, "zero");
                        add(m, "skew");
                    }
                    add(fp8 ? 3072 : 1024, "jobs2", -1, true);
                    add(fp8 ? 3072 : 1024, "jobs3", -1, true);
                }
                for (int m : {1, 8, 128, 1024}) add(m, "mixed", -1, true, true);
                suite.add(std::move(catalogue), [=](auto chosen) {
                    auto& env = cuda_environment();
                    Weights w(env.context.device(), p, side, fp8);
                    if (w.encoded_expert_bytes() != (fp8 ? 4915800 : 2764800) ||
                        w.backing_bytes() != (fp8 ? 1258487808 : 707788800))
                        throw std::runtime_error("EP payload accounting mismatch");
                    std::array<std::vector<float>, 3> patterns;
                    for (int i = 0; i < 3; ++i) {
                        patterns[i].resize(H);
                        fill_uniform(patterns[i], 871 + i, -.7F, .7F);
                        if (i == 2) {
                            std::mt19937 g(873);
                            std::normal_distribution<float> normal(0, 1);
                            for (auto& v : patterns[i]) v = std::clamp(normal(g), -3.25F, 3.25F);
                        }
                        std::fill_n(patterns[i].begin(), fp8 ? 128 : 32, 0);
                        patterns[i][H - 1] = i == 2 ? 3.5F : .875F;
                        round_to_bf16(patterns[i]);
                    }
                    ControlShared fused(env.context.device());
                    int failures = 0;
                    for (auto i : chosen)
                        failures += run(w, env.context.execution_view(), cases[i], patterns, fused);
                    return failures + w.verify_payloads();
                });
            }
    }
    return suite.main(argc, argv);
}
