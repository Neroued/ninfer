#include "core/decode_graph.h"
#include "core/platform.h"
#include "ops/moe/clamped_ep_fixture.h"
#include "ops/moe/sigmoid_route_oracle.h"
#include "ops/moe/swiglu_oracle.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear/weights.h"
#include "ops/linear_common.h"
#include <array>
#include <map>
#include "ops/float_codec.h"

namespace ninfer::test::sigmoid_moe {
namespace {
namespace ep    = clamped_ep;
namespace ip    = input_projection;
constexpr int H = 4096, I = 1024, E = 288, K = 8;

struct Case {
    int m, hits = 4;
    bool graph = false, none = false, all = false;
};

void put(quantized_weight::PackedWeight& w, int row, int col, float value) {
    if (w.weight.qtype == QType::BF16) {
        const auto b = f32_to_bf16(value);
        std::memcpy(w.payload.data() + (std::size_t(row) * w.weight.k + col) * 2, &b, 2);
    } else {
        const float scale                              = value == 0 ? 1.f : std::abs(value) / 64;
        w.payload[std::size_t(row) * w.weight.k + col] = static_cast<std::uint8_t>(value < 0   ? -64
                                                                                   : value > 0 ? 64
                                                                                               : 0);
        const auto b                                   = float_codec::f32_to_f16_bits(scale);
        std::memcpy(w.payload.data() + w.scale_plane_offset +
                        (std::size_t(row) * (w.weight.k / 32) + col / 32) * 2,
                    &b, 2);
    }
}

quantized_weight::PackedWeight control(int recipe) {
    auto w = linear_weights::make_bf16_weight(E, H, 2811);
    std::fill(w.payload.begin(), w.payload.end(), 0);
    if (recipe != 4)
        for (int e = 0; e < E; ++e) put(w, e, H - 1, (e % 17 - 8) * .125f);
    return w;
}

quantized_weight::PackedWeight shared(bool down, int recipe, bool q8 = false) {
    auto w =
        q8 ? linear_weights::make_q8_g32_fp16_weight(down ? H : 2 * I, down ? I : H,
                                                     down ? 2813 : 2815)
           : linear_weights::make_bf16_weight(down ? H : 2 * I, down ? I : H, down ? 2813 : 2815);
    if (recipe < 0) return w;
    std::fill(w.payload.begin(), w.payload.end(), 0);
    if (recipe == 4) {
        const int tail = q8 ? 32 : 1;
        if (!down) {
            put(w, 0, H - 1, 1);
            put(w, tail, H - 1, 1);
            put(w, I, H - 1, 1);
            put(w, I + tail, H - 1, 1.f / 32);
        } else
            for (int r = 0; r < H; ++r) {
                // Keep noncancelling norm anchors when the routed contraction may quantize
                // its private activation. The T=1 subset below still checks the join seam.
                put(w, r, 0, r % 13 ? (r % 2 ? 2.5f : -2.5f) : (q8 ? 2.5f : 0));
                put(w, r, tail, 1.f / 256);
            }
    } else if (recipe == 5) {
        if (!down) {
            put(w, I - 1, H - 1, 20);
            put(w, 2 * I - 1, H - 1, -20);
        } else
            for (int r = 0; r < H; ++r) put(w, r, I - 1, r % 2 ? -1 : 1);
    }
    // Recipes0..3 and6 intentionally have zero shared, so it cannot hide routed errors.
    return w;
}

struct Fixture {
    ep::Weights bank;
    ip::DevicePackedWeight router, gu, dn;
    std::vector<std::uint16_t> bits;
    std::array<std::vector<float>, 3> patterns;
    swiglu::Oracle shared_ref;
    std::map<std::pair<int, int>, std::vector<double>> shared_cache;
    bool fp8;
    int recipe;
    bool q8_shared;

    Fixture(ep::Placement placement, int side, bool eight, int rec = -1, bool q8 = false)
        : bank(cuda_environment().context.device(), placement, side, eight), router(control(rec)),
          gu(shared(false, rec, q8)), dn(shared(true, rec, q8)), bits(3 * H),
          shared_ref(gu.host, dn.host, bits, rec == 5 ? 3 : 10), fp8(eight), recipe(rec),
          q8_shared(q8) {
        if (rec >= 0 && rec <= 4) bank.clamp_probe(rec);
        for (int p = 0; p < 3; ++p) {
            auto& v = patterns[p];
            v.resize(H);
            if (rec < 0 || rec == 6) {
                fill_uniform(v, 2821 + p, -.7f, .7f);
                if (p == 2) {
                    std::mt19937 g(2823);
                    std::normal_distribution<float> n(0, 1);
                    for (auto& x : v) x = std::clamp(n(g), -3.25f, 3.25f);
                }
                std::fill_n(v.begin(), eight ? 128 : 32, 0);
                v[H - 1] = p == 2 ? 3.5f : .875f;
            } else
                v[H - 1] = 1;
            round_to_bf16(v);
            for (int k = 0; k < H; ++k) bits[p * H + k] = f32_to_bf16(v[k]);
        }
    }

    ops::SigmoidMoeWeights weights() const {
        return {router.view(),
                {},
                bank.experts(),
                {{gu.view(), ops::LinearPolicy::A16Only}, {dn.view(), ops::LinearPolicy::A16Only}}};
    }

    ops::SigmoidMoeProfile profile() const {
        auto w        = weights();
        auto s        = ops::swiglu_profile(w.shared);
        s.clamp_limit = recipe == 5 ? 3 : 10;
        return {{{H, E, K}, 2.5f, 1e-20f}, bank.profile(), s};
    }

    const std::vector<double>& reference_shared(int pattern, int sign) {
        auto key = std::pair{pattern, sign};
        if (auto it = shared_cache.find(key); it != shared_cache.end()) return it->second;
        std::vector<double> r(H);
        for (int h = 0; h < H; ++h) r[h] = shared_ref.output(h, pattern, sign);
        return shared_cache.emplace(key, std::move(r)).first->second;
    }

    int verify() {
        return bank.verify_payloads() + router.verify_preserved("whole router") +
               gu.verify_preserved("whole shared GU") + dn.verify_preserved("whole shared down");
    }
};

int run(Fixture& f, const Case& c) {
    const auto p     = f.profile();
    const auto kind  = c.none ? ops::ExpertSubsetKind::None
                              : (c.all ? ops::ExpertSubsetKind::All : f.bank.subset().kind);
    const auto cap   = ops::topk_sigmoid_swiglu_moe_workspace_capacity_bytes(p, kind, c.m, c.m);
    const auto bound = ops::topk_sigmoid_swiglu_moe_workspace_capacity_bytes(p, kind, 1, c.m);
    if (cap > bound) throw std::runtime_error("whole interval capacity");
    GuardedDeviceBuffer input(std::size_t(H) * c.m * 2), output(std::size_t(H) * c.m * 2),
        bias(E * 4), scratch(cap + 256);
    MemoryArena ws(scratch.span());
    (void)ws.alloc_bytes(256);
    cuda_check(cudaMemset(scratch.data(), 0x5a, 256), "whole prefix");
    Tensor x(input.span(), DType::BF16, {H, c.m}), y(output.span(), DType::BF16, {H, c.m});
    auto w            = f.weights();
    w.correction_bias = Tensor(bias.span(), DType::FP32, {E});
    auto subset       = f.bank.subset();
    if (c.all) subset = {ops::ExpertSubsetKind::All, {}};
    if (c.none) {
        subset            = {ops::ExpertSubsetKind::None, {}};
        w.control         = {};
        w.correction_bias = {};
        w.routed          = {};
    }
    auto execution    = cuda_environment().context.execution_view();
    const auto invoke = [&] { ops::topk_sigmoid_swiglu_moe(x, p, w, subset, y, ws, execution); };
    DecodeGraphDefinition def;
    DecodeGraphExecutable graph;
    cuda_synchronize();
    if (c.graph) {
        def.capture(execution.stream(), invoke);
        graph.instantiate(def);
    }
    int failures = 0;
    for (int phase = 0; phase < (c.graph ? 3 : 1); ++phase) {
        f.bank.publish(phase == 1 ? 1 : 0, phase == 2);
        const int sign = phase == 1;
        std::vector<std::uint16_t> values(std::size_t(H) * c.m);
        for (int t = 0; t < c.m; ++t)
            for (int k = 0; k < H; ++k)
                values[std::size_t(t) * H + k] = f.bits[(t % 3) * H + k] ^ (sign ? 0x8000 : 0);
        std::vector<float> corrections(E);
        for (int j = 0; j < K; ++j) {
            const int ordinal = (f.recipe == 4 ? 0 : (c.m * 13 + phase * 17)) + j;
            corrections[j < c.hits ? f.bank.local_id(ordinal) : f.bank.remote_id(ordinal)] = 2;
        }
        input.copy_from_host(values.data(), values.size() * 2);
        bias.copy_from_host(corrections.data(), E * 4);
        output.fill(0x7f);
        cuda_synchronize();
        if (c.graph)
            graph.launch(execution.stream());
        else
            invoke();
        cuda_synchronize(execution.stream());
        auto actual = from_device_bf16(output.data(), std::size_t(H) * c.m);
        std::array<std::vector<double>, 3> refs;
        for (int pattern = 0; pattern < std::min(c.m, 3); ++pattern) {
            refs[pattern] = f.reference_shared(pattern, sign);
            if (c.none) continue;
            auto selected = sigmoid_route::reference(
                f.router.host, std::span(values).subspan(std::size_t(pattern) * H, H), corrections,
                p.routing.scaling_factor, p.routing.normalization_epsilon);
            auto v = f.patterns[pattern];
            if (sign)
                for (auto& a : v) a = -a;
            for (int j = 0; j < K; ++j) {
                if (!c.all && !f.bank.active(selected.ids[j])) continue;
                const auto& contribution = f.bank.oracle(pattern + 3 * sign, selected.ids[j], v);
                for (int h = 0; h < H; ++h)
                    refs[pattern][h] += selected.coefficients[j] * contribution[h];
            }
        }
        std::vector<double> reference(actual.size());
        for (int t = 0; t < c.m; ++t)
            std::copy(refs[t % 3].begin(), refs[t % 3].end(),
                      reference.begin() + std::size_t(t) * H);
        const auto label = "whole/" + std::string(f.fp8 ? "fp8" : "nvfp4") + "/M" +
                           std::to_string(c.m) + "/hits" + std::to_string(c.hits) + "/recipe" +
                           std::to_string(f.recipe) + "/phase" + std::to_string(phase);
        if (f.q8_shared) {
            constexpr ReductionCriterion a16{1.0 / 256, 1.0 / 256, 2.0 / 256};
            GuardedDeviceBuffer shared_storage(std::size_t(H) * c.m * 2);
            shared_storage.fill(0x7f);
            Tensor shared_output(shared_storage.span(), DType::BF16, {H, c.m});
            ops::swiglu(x, p.shared, w.shared, shared_output, ws, execution);
            cuda_synchronize(execution.stream());
            const auto shared_actual =
                from_device_bf16(shared_storage.data(), std::size_t(H) * c.m);
            std::vector<double> shared_reference(shared_actual.size());
            for (int t = 0; t < c.m; ++t) {
                const auto& values = f.reference_shared(t % 3, sign);
                std::copy(values.begin(), values.end(),
                          shared_reference.begin() + std::size_t(t) * H);
            }
            failures +=
                verify_reduction(label + "/shared-a16", shared_actual, shared_reference, a16) +
                shared_storage.verify_guards(label + "/shared-a16");
            if (c.none || (c.hits == 0 && !c.all))
                failures +=
                    verify_reduction(label + "/shared-only-a16", actual, shared_reference, a16);
        }
        failures += verify_reduction(label, actual, reference, f.fp8 ? ep::kA8 : ep::kA4) +
                    input.verify_guards(label) + output.verify_guards(label) +
                    scratch.verify_guards(label) + f.bank.verify_controls();
        if (f.q8_shared && f.recipe == 4 && c.m == 1) {
            std::vector<double> cancelling_actual, cancelling_reference;
            for (int h = 0; h < H; ++h)
                if (h % 13) {
                    cancelling_actual.push_back(actual[h]);
                    cancelling_reference.push_back(reference[h]);
                }
            failures += verify_reduction(label + "/join-cancellation", cancelling_actual,
                                         cancelling_reference, f.fp8 ? ep::kA8 : ep::kA4);
        }
        if (ws.used() != 256 || ws.peak_used() > cap + 256) {
            std::cerr << label << " workspace mismatch\n";
            ++failures;
        }
        failures +=
            linear_common::verify_preserved(input.data(), std::as_bytes(std::span(values)), label);
        failures += linear_common::verify_preserved(bias.data(),
                                                    std::as_bytes(std::span(corrections)), label);
    }
    std::array<std::byte, 256> prefix;
    prefix.fill(std::byte{0x5a});
    failures += linear_common::verify_preserved(scratch.data(), prefix, "whole prefix");
    f.bank.publish();
    return failures;
}

int metadata(bool fp8, bool q8 = false) {
    Fixture f(ep::Placement::Permuted, 0, fp8, -1, q8);
    auto p            = f.profile();
    int failures      = 0;
    const auto reject = [&](auto&& fn) {
        try {
            fn();
            ++failures;
        } catch (const std::invalid_argument&) {}
    };
    for (int field = 0; field < 12; ++field) {
        auto bad = p;
        if (field == 0) --bad.shared.hidden;
        if (field == 1) --bad.routed.experts;
        if (field == 2) --bad.routing.route.top_k;
        if (field == 3) bad.shared.clamp_limit.reset();
        if (field == 4) bad.shared.clamp_limit = 0;
        if (field == 5) bad.routed.ffn.clamp_limit = -1;
        if (field == 6) bad.routing.scaling_factor = 0;
        if (field == 7) bad.routing.normalization_epsilon = 0;
        if (field == 8) bad.shared.gate_up_policy = ops::LinearPolicy::AllowA8;
        if (field == 9) --bad.shared.intermediate;
        if (field == 10) bad.shared.down_type = QType::NVFP4;
        if (field == 11) bad.routed.ffn.down_layout = QuantLayout::Contiguous;
        reject([&] {
            (void)ops::topk_sigmoid_swiglu_moe_workspace_capacity_bytes(bad, f.bank.subset().kind,
                                                                        1, 4097);
        });
    }
    for (auto [first, last] : std::array<std::pair<int, int>, 9>{{{1, 17},
                                                                  {16, 17},
                                                                  {64, 65},
                                                                  {127, 129},
                                                                  {255, 257},
                                                                  {511, 513},
                                                                  {1023, 1025},
                                                                  {2047, 2049},
                                                                  {4095, 4097}}}) {
        auto b = ops::topk_sigmoid_swiglu_moe_workspace_capacity_bytes(p, f.bank.subset().kind,
                                                                       first, last);
        for (int t = first; t <= last; ++t)
            if (ops::topk_sigmoid_swiglu_moe_workspace_capacity_bytes(p, f.bank.subset().kind, t,
                                                                      t) > b)
                ++failures;
    }
    for (auto interval : std::array<std::pair<int, int>, 3>{{{0, 1}, {2, 1}, {-1, 8}}})
        reject([&] {
            (void)ops::topk_sigmoid_swiglu_moe_workspace_capacity_bytes(
                p, f.bank.subset().kind, interval.first, interval.second);
        });
    auto cap = ops::topk_sigmoid_swiglu_moe_workspace_capacity_bytes(p, f.bank.subset().kind, 1, 1);
    GuardedDeviceBuffer input(H * 2), out(H * 2), bias(E * 4), scratch(cap);
    Tensor x(input.span(), DType::BF16, {H, 1}), y(out.span(), DType::BF16, {H, 1});
    auto w            = f.weights();
    w.correction_bias = Tensor(bias.span(), DType::FP32, {E});
    MemoryArena ws(scratch.span());
    std::vector<std::byte> poison(H * 2, std::byte{0xa6});
    out.copy_from_host(poison.data(), poison.size());
    for (int field = 0; field < 15; ++field) {
        auto bad = w;
        if (field == 0) bad.shared.gate_up.policy = ops::LinearPolicy::AllowA8;
        if (field == 1) bad.shared.down.weight.payload_bytes -= 16;
        if (field == 2)
            bad.shared.gate_up.weight.scales = q8 ? nullptr : bad.shared.gate_up.weight.qdata;
        if (field == 3) bad.shared.down.weight.qhigh = bad.shared.down.weight.qdata;
        if (field == 4) bad.shared.gate_up.weight.group = 16;
        if (field == 5) bad.shared.gate_up.weight.group_size = 16;
        if (field == 6) bad.shared.gate_up.weight.high_plane_bytes = 16;
        if (field == 7) bad.shared.gate_up.weight.device = Device::cpu();
        if (field == 8) bad.shared.down.weight.qdata = out.data();
        if (field == 9) bad.correction_bias = Tensor(out.span(), DType::FP32, {E});
        if (field == 10) bad.control.qdata = out.data();
        if (field == 11) --bad.routed.bank.count;
        if (field == 12) bad.shared.gate_up.weight.padded_shape[0] += 16;
        if (field == 13)
            bad.shared.down.weight.qdata =
                static_cast<const std::byte*>(bad.shared.down.weight.qdata) + 2;
        if (field == 14) bad.routed.activation_divisors.down = 0;
        reject([&] {
            ops::topk_sigmoid_swiglu_moe(x, p, bad, f.bank.subset(), y, ws,
                                         cuda_environment().default_execution);
        });
    }
    reject([&] {
        ops::topk_sigmoid_swiglu_moe(x, p, w, f.bank.subset(), x, ws,
                                     cuda_environment().default_execution);
    });
    MemoryArena short_ws(scratch.span().subspan(0, cap - 256));
    reject([&] {
        ops::topk_sigmoid_swiglu_moe(x, p, w, f.bank.subset(), y, short_ws,
                                     cuda_environment().default_execution);
    });
    failures += linear_common::verify_preserved(out.data(), poison, "whole rejected output");
    return failures;
}
} // namespace
} // namespace ninfer::test::sigmoid_moe

int main(int argc, char** argv) {
    using namespace ninfer::test;
    using namespace ninfer::test::sigmoid_moe;
    OpSuite suite;
#if NINFER_TARGET_SPARK
    for (bool fp8 : {false, true}) {
        const std::string format = fp8 ? "fp8" : "nvfp4";
        suite.add("whole/" + format + "/metadata", [fp8] { return metadata(fp8); },
                  {{"format", format}});
        for (auto placement : {clamped_ep::Placement::Permuted, clamped_ep::Placement::Contiguous,
                               clamped_ep::Placement::Interleaved})
            for (int side : {0, 1}) {
                std::vector<Case> cases;
                const bool primary = placement == clamped_ep::Placement::Permuted && side == 0;
                if (primary) {
                    for (int t : {1, 2, 3, 6, 8, 16, 17, 32, 33, 64, 65, 96, 127, 128})
                        cases.push_back({t});
                    for (int t : {129, 191,  192,  193,  255,  256,  257,  384,  511,  512,
                                  513, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097})
                        cases.push_back({t});
                    for (int t : {1,   2,   4,   6,   8,   16,   17,   32,   64,   65,  128,
                                  129, 256, 257, 512, 513, 1024, 2048, 2049, 4096, 4097})
                        cases.push_back({t, 4, true});
                    if (!fp8) cases.push_back({8193, 4, true});
                    for (int hits = 0; hits <= 8; ++hits) {
                        // The ordinary mixed cases already cover four local hits.
                        if (hits == 4) continue;
                        for (int t : {1, 8, 128, 4097}) cases.push_back({t, hits});
                    }
                    for (int t : {1, 8, 17, 128, 512, 4097}) cases.push_back({t, 0, true, true});
                } else
                    for (int t : {1, 2, 4, 8, 17, 65, 129, 257, 512, 4097}) {
                        cases.push_back({t, 8});
                        cases.push_back({t, 4, true});
                    }
                std::vector<OpCase> catalogue;
                for (const auto& c : cases)
                    catalogue.push_back(
                        {"whole/" + format + "/place" + std::to_string(int(placement)) + "/side" +
                             std::to_string(side) + "/T" + std::to_string(c.m) + "/h" +
                             std::to_string(c.hits) + (c.none ? "/none/" : "/mask/") +
                             (c.graph ? "graph" : "eager"),
                         {{"format", format}, {"execution", c.graph ? "graph" : "eager"}}});
                suite.add(std::move(catalogue), [=](auto selected) {
                    std::cerr << "Qualify whole " << format << " placement=" << int(placement)
                              << " side=" << side << " selected=" << selected.size() << "\n";
                    Fixture f(placement, side, fp8);
                    int failed = 0;
                    for (auto i : selected) failed += run(f, cases[i]);
                    return failed + f.verify();
                });
            }
        suite.add("whole/" + format + "/all",
                  [fp8] {
                      Fixture f(clamped_ep::Placement::Permuted, 0, fp8);
                      int failed = 0;
                      // Top8 is constrained to resident IDs. All must ignore mask changes on
                      // replay.
                      for (int t : {1, 17, 4097}) failed += run(f, {t, 8, true, false, true});
                      return failed + f.verify();
                  },
                  {{"format", format}, {"execution", "graph"}});
        for (int recipe = 0; recipe <= 6; ++recipe)
            suite.add("whole/" + format + "/probe" + std::to_string(recipe),
                      [=] {
                          Fixture f(clamped_ep::Placement::Permuted, 0, fp8, recipe);
                          int failed = 0;
                          if (recipe == 4)
                              failed += run(f, {1, 8, true});
                          else {
                              for (int t : {1, 2, 4, 8, 16, 17, 65, 128, 129, 257, 512, 2048, 4097})
                                  failed += run(f, {t, recipe == 5 ? 0 : 8});
                              for (int t : {1, 2, 129})
                                  failed += run(f, {t, recipe == 5 ? 0 : 8, true});
                          }
                          return failed + f.verify();
                      },
                      {{"format", format}, {"execution", "graph"}});
    }
    for (bool fp8 : {false, true}) {
        const std::string format = fp8 ? "fp8" : "nvfp4";
        const std::string prefix = "whole/q8-shared/" + format;
        suite.add(prefix + "/metadata", [fp8] { return metadata(fp8, true); },
                  {{"suite", "q8-shared"}, {"format", format}});
        for (int side : {0, 1}) {
            std::vector<Case> cases;
            if (side == 0) {
                for (int m : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096})
                    cases.push_back({m});
                for (int m : {1, 8, 128, 512, 4096}) cases.push_back({m, 4, true});
                for (int hits : {0, 1, 8})
                    for (int m : {1, 128, 4096}) cases.push_back({m, hits});
                for (int m : {1, 17, 4097}) cases.push_back({m, 0, true, true});
                for (int m : {8, 4097}) cases.push_back({m, 8, true, false, true});
            } else {
                for (int m : {1, 8, 128, 512, 4097}) {
                    cases.push_back({m, 8});
                    cases.push_back({m, 4, true});
                }
            }
            std::vector<OpCase> catalogue;
            for (const auto& c : cases)
                catalogue.push_back({prefix + "/side" + std::to_string(side) + "/T" +
                                         std::to_string(c.m) + "/h" + std::to_string(c.hits) +
                                         (c.none  ? "/none/"
                                          : c.all ? "/all/"
                                                  : "/mask/") +
                                         (c.graph ? "graph" : "eager"),
                                     {{"suite", "q8-shared"},
                                      {"format", format},
                                      {"execution", c.graph ? "graph" : "eager"}}});
            suite.add(std::move(catalogue), [=](auto selected) {
                Fixture f(side ? ep::Placement::Interleaved : ep::Placement::Permuted, side, fp8,
                          -1, true);
                int failed = 0;
                for (auto i : selected) failed += run(f, cases[i]);
                return failed + f.verify();
            });
        }
        for (int recipe : {4, 5})
            suite.add(prefix + "/probe" + std::to_string(recipe),
                      [=] {
                          Fixture f(ep::Placement::Permuted, 0, fp8, recipe, true);
                          int failed = 0;
                          for (int m : {1, 128, 4097})
                              failed += run(f, {m, recipe == 5 ? 0 : 8, m != 4097});
                          return failed + f.verify();
                      },
                      {{"suite", "q8-shared"}, {"format", format}, {"execution", "graph"}});
    }

#endif
    return suite.main(argc, argv);
}
