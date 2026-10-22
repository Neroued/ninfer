#include "ops/moe/clamped_ep_fixture.h"
#include "ops/op_tester.h"
#include "ops/suite.h"
#include "ops/linear_common.h"
#include "core/platform.h"
#include "core/decode_graph.h"
#include <array>

namespace ninfer::test::clamped_ep {
namespace {
struct Case {
    int tokens;
    std::string distribution;
    int hits   = -1;
    bool graph = false;
    bool fp32  = false;
    bool all   = false;
};

int run(Weights& w, CudaExecutionView execution, const Case& c,
        const std::array<std::vector<float>, 3>& patterns) {
    const int m                  = c.tokens;
    const auto dtype             = c.fp32 ? DType::FP32 : DType::BF16;
    const std::size_t cell_bytes = c.fp32 ? 4 : 2;
    const bool fp8    = w.profile().ffn.gate_up_type == QType::FP8_E4M3FN_BLOCK128X128_BF16;
    const auto subset = c.all ? ops::ExpertSubset{ops::ExpertSubsetKind::All, {}} : w.subset();
    auto routes       = make_routes(w, m, c.distribution.c_str(), c.hits);
    std::vector<float> input(std::size_t(m) * H);
    for (int t = 0; t < m; ++t)
        std::copy(patterns[t % 3].begin(), patterns[t % 3].end(),
                  input.begin() + std::size_t(t) * H);
    auto dx = to_device_bf16(input), di = to_device(routes.ids),
         da = to_device(routes.coefficients);
    Tensor x(dx.span(), DType::BF16, {H, m});
    ops::ExpertRoutes selected{Tensor(di.span(), DType::I32, {K, m}),
                               Tensor(da.span(), DType::FP32, {K, m})};
    const auto bytes = ops::routed_swiglu_workspace_capacity_bytes(w.profile(), subset.kind, dtype,
                                                                   ops::MoeEpilogue::Write, m, m);
    const auto bound = ops::routed_swiglu_workspace_capacity_bytes(w.profile(), subset.kind, dtype,
                                                                   ops::MoeEpilogue::Write, 1, m);
    if (bytes > bound) throw std::runtime_error("EP interval capacity smaller than exact capacity");
    GuardedDeviceBuffer scratch(bytes + 256), result(input.size() * cell_bytes);
    MemoryArena arena(scratch.span());
    (void)arena.alloc_bytes(256);
    cuda_check(cudaMemset(scratch.data(), 0x5a, 256), "EP prefix");
    Tensor y(result.span(), dtype, {H, m});
    auto invoke = [&] {
        ops::routed_swiglu(x, w.profile(), w.experts(), selected, subset, ops::MoeEpilogue::Write,
                           y, arena, execution);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    if (c.graph) {
        cuda_synchronize();
        definition.capture(execution.stream(), invoke);
        graph.instantiate(definition);
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
                               : w.local_id(1 + (t * 13 + j * 97) % (Local - 1));
        }
        if (c.distribution == "load-bands" || c.distribution == "broad-loads") {
            // Distinct global routes cross load bands while addresses and the Graph stay fixed.
            static constexpr int bands[3][K]{{16, 17, 32, 33, 64, 65, 128, 129},
                                             {32, 33, 64, 65, 128, 129, 512, 513},
                                             {0, 1, 16, 17, 256, 257, 512, 513}};
            static constexpr int broad[3][K]{{64, 128, 128, 128, 128, 128, 128, 512},
                                             {65, 129, 129, 129, 129, 129, 129, 513},
                                             {64, 64, 64, 64, 513, 513, 513, 513}};
            const auto& counts = c.distribution == "load-bands" ? bands[replay] : broad[replay];
            for (int t = 0; t < m; ++t)
                for (int j = 0; j < K; ++j)
                    routes.ids[std::size_t(t) * K + j] =
                        t < counts[j] ? w.local_id(j) : w.remote_id(j);
        }
        if (c.distribution == "slice-tail") {
            static constexpr int counts[K]{16, 17, 128, 129, 256, 257, 512, 513};
            for (int t = 0; t < m; ++t)
                for (int j = 0; j < K; ++j)
                    routes.ids[std::size_t(t) * K + j] =
                        c.all || t < counts[j] || (t == m - 1 && j == 0) ? w.local_id(j)
                                                                         : w.remote_id(j);
        }
        if (replay == 1) {
            for (int t = 0; t < m; ++t) {
                auto first = routes.ids.begin() + std::size_t(t) * K;
                std::rotate(first, first + 3, first + K);
            }
        }
        if (fp8 && c.distribution == "pair") {
            for (int j = 0; j < K; ++j) {
                routes.ids[j]     = w.local_id(j);
                routes.ids[K + j] = w.local_id(K + j);
            }
            if (replay == 1)
                for (int j = 0; j < 3; ++j) routes.ids[K + j] = routes.ids[j];
            if (replay == 2) std::copy_n(routes.ids.begin(), K, routes.ids.begin() + K);
        }
        di.copy_from_host(routes.ids.data(), di.size());
        da.copy_from_host(routes.coefficients.data(), da.size());
        result.fill(0x7f);
        if (c.graph)
            graph.launch(execution.stream());
        else
            invoke();
        cuda_synchronize(execution.stream());
        const auto actual_bits =
            from_device<std::uint8_t>(result.data(), input.size() * cell_bytes);
        std::vector<double> reference(input.size());
        for (int t = 0; t < m; ++t) {
            int hits = 0;
            for (int j = 0; j < K; ++j) {
                const auto a = std::size_t(t) * K + j;
                const int e  = routes.ids[a];
                if (!c.all && !w.active(e)) continue;
                ++hits;
                const auto& v = w.oracle(t % 3, e, patterns[t % 3]);
                for (int h = 0; h < H; ++h)
                    reference[std::size_t(t) * H + h] += double(routes.coefficients[a]) * v[h];
            }
            if (!hits) {
                const auto begin = actual_bits.begin() + std::size_t(t) * H * cell_bytes;
                if (std::any_of(begin, begin + H * cell_bytes, [](auto v) { return v != 0; }))
                    ++failures;
            }
        }
        const std::string label = "EP144 m=" + std::to_string(m) + " " + c.distribution +
                                  " hits=" + std::to_string(c.hits) +
                                  " replay=" + std::to_string(replay);
        std::vector<double> got;
        if (c.fp32) {
            const auto values = from_device<float>(result.data(), input.size());
            got.assign(values.begin(), values.end());
        } else
            got = from_device_bf16(result.data(), input.size());
        failures += verify_reduction(label, got, reference, fp8 ? kA8 : kA4);
        failures +=
            scratch.verify_guards(label) + result.verify_guards(label) + w.verify_controls();
        if (arena.used() != 256 || arena.peak_used() > bytes + 256) {
            std::cerr << label << " capacity/arena mismatch\n";
            ++failures;
        }
        if (from_device<int>(di, routes.ids.size()) != routes.ids ||
            from_device<float>(da, routes.coefficients.size()) != routes.coefficients ||
            from_device_bf16(dx, input.size()) != std::vector<double>(input.begin(), input.end()))
            ++failures;
    }
    std::array<std::byte, 256> prefix;
    prefix.fill(std::byte{0x5a});
    failures += linear_common::verify_preserved(scratch.data(), prefix, "EP prefix");
    w.publish();
    return failures;
}

int metadata(bool fp8) {
    auto& env = cuda_environment();
    Weights w(env.context.device(), Placement::Permuted, 0, fp8);
    const auto p        = w.profile();
    int failures        = 0;
    const auto rejected = [&](auto&& call) {
        try {
            call();
            ++failures;
        } catch (const std::invalid_argument&) {};
    };
    for (int field = 0; field < 10; ++field) {
        auto bad = p;
        if (field == 0) bad.ffn.clamp_limit.reset();
        if (field == 1) bad.ffn.clamp_limit = 0;
        if (field == 2) bad.ffn.clamp_limit = -1;
        if (field == 3) bad.ffn.clamp_limit = std::numeric_limits<float>::infinity();
        if (field == 4) bad.ffn.clamp_limit = std::numeric_limits<float>::quiet_NaN();
        if (field == 5) --bad.experts;
        if (field == 6) --bad.top_k;
        if (field == 7) bad.ffn.intermediate -= 128;
        if (field == 8)
            bad.ffn.gate_up_policy = fp8 ? ops::LinearPolicy::AllowA4 : ops::LinearPolicy::AllowA8;
        if (field == 9) bad.ffn.down_layout = QuantLayout::Contiguous;
        rejected([&] {
            (void)ops::routed_swiglu_workspace_capacity_bytes(bad, w.subset().kind, DType::BF16,
                                                              ops::MoeEpilogue::Write, 1, 4097);
        });
    }
    for (auto [first, last] : std::array<std::pair<int, int>, 13>{{{1, 17},
                                                                   {16, 17},
                                                                   {64, 65},
                                                                   {127, 129},
                                                                   {255, 257},
                                                                   {511, 513},
                                                                   {767, 769},
                                                                   {1023, 1025},
                                                                   {1535, 1537},
                                                                   {2047, 2049},
                                                                   {3071, 3073},
                                                                   {4095, 4097},
                                                                   {8191, 8193}}}) {
        for (auto subset : {w.subset().kind, ops::ExpertSubsetKind::All}) {
            const auto bound = ops::routed_swiglu_workspace_capacity_bytes(
                p, subset, DType::BF16, ops::MoeEpilogue::Write, first, last);
            for (int t = first; t <= last; ++t)
                if (ops::routed_swiglu_workspace_capacity_bytes(
                        p, subset, DType::BF16, ops::MoeEpilogue::Write, t, t) > bound)
                    ++failures;
        }
    }
    rejected([&] {
        (void)ops::routed_swiglu_workspace_capacity_bytes(p, w.subset().kind, DType::BF16,
                                                          ops::MoeEpilogue::AddResidual, 1, 1);
    });
    const auto cap = ops::routed_swiglu_workspace_capacity_bytes(p, w.subset().kind, DType::BF16,
                                                                 ops::MoeEpilogue::Write, 1, 1);
    GuardedDeviceBuffer xdata(H * 2), output(H * 2), ids(K * 4), coeff(K * 4), scratch(cap);
    std::vector<std::byte> poison(H * 2, std::byte{0xa6});
    output.copy_from_host(poison.data(), poison.size());
    Tensor x(xdata.span(), DType::BF16, {H, 1}), y(output.span(), DType::BF16, {H, 1});
    ops::ExpertRoutes routes{Tensor(ids.span(), DType::I32, {K, 1}),
                             Tensor(coeff.span(), DType::FP32, {K, 1})};
    MemoryArena arena(scratch.span());
    for (int field = 0; field < 8; ++field) {
        auto bad = w.experts();
        if (field == 0) {
            if (fp8)
                bad.activation_divisors.gate_up = 1;
            else
                bad.activation_divisors.gate_up.reset();
        }
        if (field == 1) bad.activation_divisors.down = 0;
        if (field == 2) bad.activation_divisors.gate_up = std::numeric_limits<float>::quiet_NaN();
        if (field == 3) --bad.bank.count;
        if (field == 4) bad.bank.down.offsets = nullptr;
        if (field == 5)
            bad.bank.gate_up.divisors =
                fp8 ? reinterpret_cast<const float*>(bad.bank.gate_up.codes) : nullptr;
        if (field == 6) bad.bank.down.indexed_storage.bytes = 0;
        if (field == 7) bad.bank.device = Device::cpu();
        rejected([&] {
            ops::routed_swiglu(x, p, bad, routes, w.subset(), ops::MoeEpilogue::Write, y, arena,
                               env.context.execution_view());
        });
    }
    rejected([&] {
        ops::routed_swiglu(x, p, w.experts(), routes, w.subset(), ops::MoeEpilogue::Write, x, arena,
                           env.context.execution_view());
    });
    MemoryArena short_arena(scratch.span().subspan(0, cap - 256));
    rejected([&] {
        ops::routed_swiglu(x, p, w.experts(), routes, w.subset(), ops::MoeEpilogue::Write, y,
                           short_arena, env.context.execution_view());
    });
    failures += linear_common::verify_preserved(output.data(), poison, "EP rejected output");
    for (auto type : {DType::BF16, DType::FP32}) {
        const int bytes = type == DType::BF16 ? 2 : 4;
        GuardedDeviceBuffer zero_output(std::size_t(H) * 17 * bytes),
            input17(std::size_t(H) * 17 * 2);
        zero_output.fill(0xa6);
        Tensor zx(input17.span(), DType::BF16, {H, 17}), zy(zero_output.span(), type, {H, 17});
        if (ops::routed_swiglu_workspace_capacity_bytes(p, ops::ExpertSubsetKind::None, type,
                                                        ops::MoeEpilogue::Write, 1, 4097) != 0)
            ++failures;
        ops::routed_swiglu(zx, p, ops::CudaSwiGLUExperts{}, ops::ExpertRoutes{},
                           ops::ExpertSubset{ops::ExpertSubsetKind::None}, ops::MoeEpilogue::Write,
                           zy, arena, env.context.execution_view());
        cuda_synchronize();
        const auto got = from_device<std::uint8_t>(zero_output.data(), zero_output.bytes());
        if (std::any_of(got.begin(), got.end(), [](auto v) { return v != 0; })) ++failures;
        failures += zero_output.verify_guards("EP None");
    }
    return failures + output.verify_guards("EP metadata") + scratch.verify_guards("EP metadata");
}
} // namespace
} // namespace ninfer::test::clamped_ep

int main(int argc, char** argv) {
    using namespace ninfer;
    using namespace ninfer::test;
    using namespace ninfer::test::clamped_ep;
    OpSuite suite;
#if NINFER_TARGET_SPARK
    suite.add("clamped-ep/nvfp4/metadata", [] { return metadata(false); },
              {{"format", "nvfp4"}, {"suite", "metadata"}});
    suite.add("clamped-ep/fp8/metadata", [] { return metadata(true); },
              {{"format", "fp8"}, {"suite", "metadata"}});
    for (bool fp8 : {false, true}) {
        const std::string format = fp8 ? "fp8" : "nvfp4";
        for (auto placement : {Placement::Permuted, Placement::Contiguous, Placement::Interleaved})
            for (int side : {0, 1}) {
                const bool primary = placement == Placement::Permuted && side == 0;
                std::vector<Case> cases;
                std::vector<OpCase> catalogue;
                const auto add = [&](int t, const char* dist, int hits = -1, bool graph = false,
                                     bool fp32 = false, bool all = false) {
                    const std::string id =
                        "clamped-ep/" + format + "/p" + std::to_string(int(placement)) + "/side" +
                        std::to_string(side) + "/T" + std::to_string(t) + "/" + dist + "/h" +
                        std::to_string(hits) + (graph ? "/graph" : "/eager");
                    cases.push_back({t, dist, hits, graph, fp32, all});
                    catalogue.push_back(
                        {id + (fp32 ? "/fp32" : "") + (all ? "/all" : ""),
                         {{"format", format}, {"execution", graph ? "graph" : "eager"}}});
                };
                for (int hits = 0; hits <= K; ++hits) add(1, "mixed", hits);
                if (primary) {
                    if (!fp8) {
                        add(128, "load-bands", -1, true);
                        add(1024, "load-bands", -1, true);
                        add(4096, "broad-loads", -1, true);
                        add(8193, "slice-tail", -1, true);
                        add(8193, "slice-tail", -1, true, false, true);
                    }
                    for (int t : {2, 3, 6, 8, 16, 17, 32, 33, 64, 65, 96, 127, 128})
                        add(t, "mixed");
                    for (int t : {129, 191,  192,  193,  255,  256,  257,  384,  511,  512,
                                  513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097})
                        add(t, "mixed");
                    for (int t : {1, 4, 6, 8, 16, 17, 64, 65, 128, 129, 256, 257, 512, 513, 1024,
                                  2048, 4096, 4097})
                        add(t, "mixed", -1, true);
                    for (int t : {4, 6, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096})
                        for (const char* d : {"zero", "all", "skew"}) add(t, d);
                    for (int hits = 0; hits <= K; ++hits) add(17, "mixed", hits);
                    if (fp8) {
                        for (int hits = 0; hits <= K; ++hits) add(2, "mixed", hits);
                        add(2, "pair", -1, true);
                    }
                    add(1024, "jobs2", -1, true);
                    add(1024, "jobs3", -1, true);
                    for (int t : {1, 4, 8, 17, 65, 128, 129, 512, 4097})
                        add(t, "mixed", -1, false, true);
                    for (int t : {17, 4097}) add(t, "skew", -1, true, true);
                } else {
                    for (int t : {4, 6, 8, 17, 128, 129, 512, 4097}) add(t, "mixed");
                    for (int t : {2, 17, 129, 4097}) add(t, "skew", -1, true);
                }
                suite.add(std::move(catalogue), [=](auto chosen) {
                    auto& env = cuda_environment();
                    std::cerr << "Qualify EP144 placement=" << int(placement) << " side=" << side
                              << " format=" << format << " selected=" << chosen.size() << "\n";
                    Weights w(env.context.device(), placement, side, fp8);
                    const std::size_t encoded =
                        3ULL * H * I * (fp8 ? 1 : 0) +
                        (!fp8 ? 3ULL * H * I * 9 / 16 : 3ULL * H * I / 8192);
                    if (w.encoded_expert_bytes() != encoded || w.backing_bytes() != Local * encoded)
                        throw std::runtime_error("EP144 payload accounting");
                    std::array<std::vector<float>, 3> patterns;
                    for (int i = 0; i < 3; ++i) {
                        patterns[i].resize(H);
                        fill_uniform(patterns[i], 871 + i, -.7F, .7F);
                        if (i == 2) {
                            std::mt19937 gen(873);
                            std::normal_distribution<float> norm(0, 1);
                            for (auto& v : patterns[i]) v = std::clamp(norm(gen), -3.25F, 3.25F);
                        }
                        std::fill_n(patterns[i].begin(), fp8 ? 128 : 32, 0);
                        patterns[i][H - 1] = i == 2 ? 3.5F : .875F;
                        round_to_bf16(patterns[i]);
                    }
                    int failures = 0;
                    for (auto i : chosen)
                        failures += run(w, env.context.execution_view(), cases[i], patterns);
                    return failures + w.verify_payloads();
                });
            }
        for (int recipe = 0; recipe < 4; ++recipe)
            suite.add("clamped-ep/" + format + "/clamp" + std::to_string(recipe),
                      [recipe, fp8] {
                          auto& env = cuda_environment();
                          Weights w(env.context.device(), Placement::Permuted, 0, fp8);
                          w.clamp_probe(recipe);
                          std::array<std::vector<float>, 3> patterns;
                          for (auto& v : patterns) {
                              v.resize(H);
                              v[H - 1] = 1;
                          }
                          int failures = 0;
                          for (int t : {1, 4, 6, 8, 16, 17, 64, 65, 128, 129, 256, 257, 512, 513,
                                        1024, 2048, 4096, 4097})
                              failures += run(w, env.context.execution_view(),
                                              {t, "skew", -1, false}, patterns);
                          failures += run(w, env.context.execution_view(), {129, "skew", -1, true},
                                          patterns);
                          return failures + w.verify_payloads();
                      },
                      {{"format", format}});
    }
#endif
    return suite.main(argc, argv);
}
