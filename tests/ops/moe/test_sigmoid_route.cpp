#include "core/platform.h"
#include "core/decode_graph.h"
#include "ninfer/ops/moe.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear/weights.h"
#include "ops/linear_common.h"
#include "sigmoid_route_oracle.h"
#include <array>

namespace ninfer::test::sigmoid_route {
namespace {
constexpr int H = 4096, E = 288, K = 8;
constexpr ReductionCriterion kCoefficient{5e-6, 5e-7, 1e-5};

struct Call {
    int t;
    bool graph;
};

std::vector<std::uint16_t> activation(int tokens, bool repeated) {
    std::vector<std::uint16_t> v(std::size_t(H) * tokens);
    for (int t = 0; t < tokens; ++t)
        for (int j = 0; j < (t && !repeated ? 4 : H); ++j) {
            const int source = repeated ? 0 : t;
            const auto hash = quantized_weight::detail::mix64(std::uint64_t(source) * H + j + 2601);
            const int k     = source ? int(hash % H) : j;
            v[std::size_t(t) * H + k] =
                f32_to_bf16((int((hash >> 32) % 31) - 15) * (source ? .125f : .02f));
        }
    return v;
}

int run(std::span<const Call> calls, int recipe = 0, bool repeated = false) {
    const int max =
        std::max_element(calls.begin(), calls.end(), [](auto a, auto b) { return a.t < b.t; })->t;
    auto host = linear_weights::make_bf16_weight(E, H, 2603);
    auto bits = activation(max, repeated);
    std::vector<float> bias(E);
    for (int e = 0; e < E; ++e)
        bias[e] = (int(quantized_weight::detail::mix64(e + 2605) % 17) - 8) * .003f;
    if (recipe) {
        std::fill(host.payload.begin(), host.payload.end(), 0);
        std::fill(bits.begin(), bits.end(), 0);
        std::fill(bias.begin(), bias.end(), 0);
        const auto put = [&](int e, int k, float value) {
            const auto b = f32_to_bf16(value);
            std::memcpy(host.payload.data() + (std::size_t(e) * H + k) * 2, &b, 2);
        };
        for (int t = 0; t < max; ++t) {
            bits[std::size_t(t) * H]         = f32_to_bf16(1);
            bits[std::size_t(t) * H + H - 1] = f32_to_bf16(1);
        }
        if (recipe == 2) { // A BF16 logit exit would tie experts7/8 and select the wrong ID7.
            for (int e = 0; e < E; ++e) bias[e] = e < 7 ? 1.f : (e > 8 ? -1.f : 0.f);
            put(7, 0, .5f);
            put(8, 0, .5f);
            put(8, H - 1, 1.f / 4096);
        }
        if (recipe == 3) {
            for (int e = 0; e < E; ++e) put(e, 0, e % 2 ? 100.f : -100.f);
        }
        if (recipe == 4) {
            for (int e = 0; e < E; ++e) put(e, 0, (e % 13 - 6) * .25f);
            for (int e = 0; e < E; ++e) bias[e] = (e % 7) * .125f;
        }
    }
    input_projection::DevicePackedWeight weight(std::move(host));
    GuardedDeviceBuffer input(bits.size() * 2), device_bias(E * 4);
    const ops::SigmoidRouteProfile p{
        {H, E, K}, recipe == 4 ? 3.f : 2.5f, recipe == 4 ? .5f : 1e-20f};
    const auto interval = ops::topk_sigmoid_route_workspace_capacity_bytes(p, 1, max);
    DeviceContext context;
    int failures = 0;
    for (auto call : calls) {
        const auto label = "sigmoid-route/recipe" + std::to_string(recipe) + "/T" +
                           std::to_string(call.t) + (call.graph ? "/graph" : "/eager");
        const auto cap = ops::topk_sigmoid_route_workspace_capacity_bytes(p, call.t, call.t);
        GuardedDeviceBuffer ids(std::size_t(K) * call.t * 4),
            coefficients(std::size_t(K) * call.t * 4), scratch(cap + 256);
        MemoryArena ws(scratch.span());
        (void)ws.alloc_bytes(256);
        cuda_check(cudaMemset(scratch.data(), 0x5a, 256), "route prefix");
        ops::ExpertRoutes out{Tensor(ids.span(), DType::I32, {K, call.t}),
                              Tensor(coefficients.span(), DType::FP32, {K, call.t})};
        Tensor x(input.span(), DType::BF16, {H, call.t}), b(device_bias.span(), DType::FP32, {E});
        const auto cuda =
            call.graph ? context.execution_view() : cuda_environment().default_execution;
        const auto launch = [&] { ops::topk_sigmoid_route(x, p, weight.view(), b, out, ws, cuda); };
        DecodeGraphDefinition def;
        DecodeGraphExecutable graph;
        cuda_check(cudaDeviceSynchronize(), "route fixture ready");
        if (call.graph) {
            def.capture(cuda.stream(), launch);
            graph.instantiate(def);
        }
        for (int phase = 0; phase < (call.graph ? 2 : 1); ++phase) {
            auto values      = bits;
            auto corrections = bias;
            if (phase) {
                for (auto& v : values) v ^= 0x8000;
                std::reverse(corrections.begin(), corrections.end());
            }
            input.copy_from_host(values.data(), values.size() * 2);
            device_bias.copy_from_host(corrections.data(), E * 4);
            ids.fill(0xa6);
            coefficients.fill(0xff);
            if (!call.graph) ws.reset_peak();
            cuda_check(cudaDeviceSynchronize(), "route update ready");
            if (call.graph)
                graph.launch(cuda.stream());
            else
                launch();
            cuda_check(cudaStreamSynchronize(cuda.stream()), "route done");
            auto got_ids = from_device<int>(ids.data(), std::size_t(K) * call.t);
            auto got_coefficients =
                from_device<float>(coefficients.data(), std::size_t(K) * call.t);
            std::vector<int> ref_ids(got_ids.size());
            std::vector<double> got(got_coefficients.begin(), got_coefficients.end()),
                ref(got.size());
            Reference common{};
            if (repeated || recipe)
                common = reference(weight.host, std::span(values).first(H), corrections,
                                   p.scaling_factor, p.normalization_epsilon);
            for (int t = 0; t < call.t; ++t) {
                const auto value =
                    (repeated || recipe)
                        ? common
                        : reference(weight.host, std::span(values).subspan(std::size_t(t) * H, H),
                                    corrections, p.scaling_factor, p.normalization_epsilon);
                for (int j = 0; j < K; ++j) {
                    ref_ids[std::size_t(t) * K + j] = value.ids[j];
                    ref[std::size_t(t) * K + j]     = value.coefficients[j];
                }
            }
            failures += verify_exact(label.c_str(), got_ids, ref_ids) +
                        verify_reduction(label, got, ref, kCoefficient);
            failures += ids.verify_guards(label) + coefficients.verify_guards(label) +
                        scratch.verify_guards(label);
            failures += linear_common::verify_preserved(input.data(),
                                                        std::as_bytes(std::span(values)), label) +
                        linear_common::verify_preserved(
                            device_bias.data(), std::as_bytes(std::span(corrections)), label);
            if (ws.used() != 256 || ws.peak_used() > cap + 256 || cap > interval) ++failures;
        }
        std::array<std::byte, 256> prefix;
        prefix.fill(std::byte{0x5a});
        failures += linear_common::verify_preserved(scratch.data(), prefix, label);
    }
    return failures + weight.verify_preserved("sigmoid router weight") +
           input.verify_guards("router input") + device_bias.verify_guards("router bias");
}

int metadata() {
    ops::SigmoidRouteProfile p{{H, E, K}, 2.5f, 1e-20f};
    int failures        = 0;
    const auto rejected = [&](auto&& f) {
        try {
            f();
            ++failures;
        } catch (const std::invalid_argument&) {};
    };
    for (int field = 0; field < 3; ++field) {
        auto bad = p;
        if (field == 0) --bad.route.hidden;
        if (field == 1) ++bad.route.experts;
        if (field == 2) --bad.route.top_k;
        rejected([&] { (void)ops::topk_sigmoid_route_workspace_capacity_bytes(bad, 1, 17); });
    }
    for (float v : {0.f, -1.f, std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN()})
        for (int field = 0; field < 2; ++field) {
            auto bad                                                 = p;
            (field ? bad.normalization_epsilon : bad.scaling_factor) = v;
            rejected([&] { (void)ops::topk_sigmoid_route_workspace_capacity_bytes(bad, 1, 17); });
        }
    for (auto [first, last] : std::array<std::pair<int, int>, 3>{{{0, 1}, {2, 1}, {-1, 17}}})
        rejected([&] { (void)ops::topk_sigmoid_route_workspace_capacity_bytes(p, first, last); });
    input_projection::DevicePackedWeight weight(linear_weights::make_bf16_weight(E, H, 2611));
    const auto cap = ops::topk_sigmoid_route_workspace_capacity_bytes(p, 1, 1);
    GuardedDeviceBuffer input(H * 2), bias(E * 4), ids(K * 4), coeff(K * 4), scratch(cap);
    std::vector<std::byte> untouched(K * 4, std::byte{0xa6});
    ids.copy_from_host(untouched.data(), untouched.size());
    coeff.copy_from_host(untouched.data(), untouched.size());
    Tensor x(input.span(), DType::BF16, {H, 1}), b(bias.span(), DType::FP32, {E});
    ops::ExpertRoutes out{Tensor(ids.span(), DType::I32, {K, 1}),
                          Tensor(coeff.span(), DType::FP32, {K, 1})};
    MemoryArena ws(scratch.span());
    for (int field = 0; field < 6; ++field) {
        auto bad = weight.view();
        if (field == 0) bad.group = 16;
        if (field == 1) bad.scales = bad.qdata;
        if (field == 2) bad.n++;
        if (field == 3) bad.payload_bytes -= 16;
        if (field == 4) bad.qdata = static_cast<const std::byte*>(bad.qdata) + 2;
        if (field == 5) bad.device = Device::cpu();
        rejected([&] {
            ops::topk_sigmoid_route(x, p, bad, b, out, ws, cuda_environment().default_execution);
        });
    }
    MemoryArena short_ws(scratch.span().subspan(0, cap - 256));
    rejected([&] {
        ops::topk_sigmoid_route(x, p, weight.view(), b, out, short_ws,
                                cuda_environment().default_execution);
    });
    auto alias              = out;
    alias.coefficients.data = ids.data();
    rejected([&] {
        ops::topk_sigmoid_route(x, p, weight.view(), b, alias, ws,
                                cuda_environment().default_execution);
    });
    return failures + linear_common::verify_preserved(ids.data(), untouched, "route rejected IDs") +
           linear_common::verify_preserved(coeff.data(), untouched, "route rejected coefficients");
}
} // namespace
} // namespace ninfer::test::sigmoid_route

int main(int argc, char** argv) {
    using namespace ninfer::test;
    using namespace ninfer::test::sigmoid_route;
    OpSuite suite;
#if NINFER_TARGET_SPARK
    std::vector<Call> calls;
    for (int t :
         {1, 2, 6, 8, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 512, 513, 1024, 2048, 4096, 4097})
        calls.push_back({t, false});
    for (int t :
         {1, 4, 6, 8, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257, 512, 513, 1024, 2048, 4096})
        calls.push_back({t, true});
    std::vector<OpCase> cases;
    for (auto c : calls)
        cases.push_back({"sigmoid-route/T" + std::to_string(c.t) + (c.graph ? "/graph" : "/eager"),
                         {{"format", "bf16"}, {"execution", c.graph ? "graph" : "eager"}}});
    suite.add(std::move(cases), [calls](auto selected) {
        std::vector<Call> chosen;
        for (auto j : selected) chosen.push_back(calls[j]);
        return run(chosen);
    });
    for (int recipe = 1; recipe <= 4; ++recipe)
        suite.add("sigmoid-route/recipe" + std::to_string(recipe),
                  [recipe] {
                      const std::array<Call, 11> c{{{1, false},
                                                    {4, false},
                                                    {6, false},
                                                    {8, false},
                                                    {16, false},
                                                    {17, false},
                                                    {128, false},
                                                    {129, false},
                                                    {512, false},
                                                    {4097, false},
                                                    {4097, true}}};
                      return run(c, recipe);
                  },
                  {{"format", "bf16"}});
    suite.add("sigmoid-route/repeated-dense",
              [] {
                  const std::array<Call, 2> c{{{4097, false}, {4097, true}}};
                  return run(c, 0, true);
              },
              {{"format", "bf16"}});
    suite.add("sigmoid-route/metadata", metadata, {{"format", "bf16"}});
#endif
    return suite.main(argc, argv);
}
