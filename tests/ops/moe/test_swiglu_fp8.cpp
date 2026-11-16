#include "core/decode_graph.h"
#include "core/platform.h"
#include "ninfer/ops/moe.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear_common.h"
#include "swiglu_oracle.h"
#include "ops/moe/swiglu/dispatch.h"

#include <array>
#include <limits>
#include <optional>

namespace ninfer::test::swiglu {
namespace {
constexpr int Hidden = 4096;
constexpr ReductionCriterion kA16{1.0 / 256, 1.0 / 256, 2.0 / 256};
constexpr ReductionCriterion kA8{.08, 1e-7, .12, 1e-7};

struct Profile {
    const char* name;
    QType format;
    int intermediate;
    ops::LinearPolicy policy;
    float activation_scale = 1.F;
    float down_extent      = .05F;
};

struct Call {
    int tokens;
    bool graph;
    int join = 0; // 1: changing mask; 2: null values; 3: null mask; 4: four-byte-aligned values.
};

std::vector<std::uint16_t> activation(int tokens) {
    std::vector<std::uint16_t> out(std::size_t(Hidden) * tokens);
    for (int t = 0; t < tokens; ++t)
        for (int j = 0; j < (t ? 4 : Hidden); ++j) {
            const auto bits = qw::detail::mix64(std::uint64_t(t) * Hidden + j + 9101);
            const int k     = t ? int(bits % Hidden) : j;
            int value       = int((bits >> 32) % 31) - 15;
            if (!value) value = 1;
            out[std::size_t(t) * Hidden + k] = f32_to_bf16(value * (t ? .125f : .5f));
        }
    return out;
}

int run(const Profile& profile, std::span<const Call> calls) {
    const auto maximum = std::max_element(calls.begin(), calls.end(), [](auto a, auto b) {
                             return a.tokens < b.tokens;
                         })->tokens;
    input_projection::DevicePackedWeight gate_up(qw::make_patterned_weight(
        profile.format, 2 * profile.intermediate, Hidden, 9131, {.native_extent = .05f}));
    input_projection::DevicePackedWeight down(
        qw::make_patterned_weight(profile.format, Hidden, profile.intermediate, 9133,
                                  {.native_extent = profile.down_extent}));
    auto initial = activation(maximum);
    if (profile.activation_scale != 1.F)
        for (auto& value : initial)
            value = f32_to_bf16(std::bit_cast<float>(std::uint32_t(value) << 16) *
                                profile.activation_scale);
    Oracle oracle(gate_up.host, down.host, initial);
    GuardedDeviceBuffer input(initial.size() * sizeof(std::uint16_t));
    ops::SwiGLUWeights weights{{gate_up.view(), profile.policy}, {down.view(), profile.policy}};
    auto math           = ops::swiglu_profile(weights);
    math.clamp_limit    = 10;
    const auto interval = ops::swiglu_workspace_capacity_bytes(math, 1, maximum);
    const auto cuda     = cuda_environment().context.execution_view();
    int failures        = 0;
    for (const auto& call : calls) {
        const auto label = std::string("swiglu/") + profile.name + "/T" +
                           std::to_string(call.tokens) + (call.graph ? "/graph" : "/eager") +
                           (call.join == 1   ? "/join-mixed"
                            : call.join == 2 ? "/join-zero"
                            : call.join == 3 ? "/join-all"
                            : call.join == 4 ? "/join-unaligned"
                                             : "");
        const auto capacity = ops::swiglu_workspace_capacity_bytes(math, call.tokens, call.tokens);
        if (capacity > interval) ++failures;
        GuardedDeviceBuffer scratch(capacity + 256);
        MemoryArena workspace(scratch.span());
        (void)workspace.alloc_bytes(64, 1);
        input_projection::GuardedBf16Tensor output(Hidden, call.tokens);
        Tensor x(input.span(), DType::BF16, {Hidden, call.tokens});
        auto y = output.tensor();
        std::optional<GuardedDeviceBuffer> routed, validity;
        std::vector<float> routed_host;
        std::vector<int> valid_host;
        if (call.join) {
            validity.emplace(std::size_t(call.tokens) * sizeof(int));
            valid_host.resize(call.tokens);
            if (call.join == 1 || call.join >= 3) {
                routed.emplace((std::size_t(Hidden) * call.tokens + (call.join == 4)) *
                               sizeof(float));
                routed_host.resize(std::size_t(Hidden) * call.tokens);
            }
        }
        const auto prepare_join = [&](int phase) {
            if (!call.join) return;
            for (int t = 0; t < call.tokens; ++t) {
                valid_host[t] = call.join == 2 ? 0 : call.join >= 3 ? 1 : int((t + phase) % 3 != 1);
                if (routed)
                    for (int r = 0; r < Hidden; ++r)
                        routed_host[std::size_t(t) * Hidden + r] =
                            valid_host[t]
                                ? (int((std::uint32_t(r) * 13U + std::uint32_t(t) * 7U) % 17U) -
                                   8) *
                                      (phase ? -.03125F : .03125F)
                                : std::numeric_limits<float>::quiet_NaN();
            }
            validity->copy_from_host(valid_host.data(), valid_host.size() * sizeof(int));
            if (routed) {
                auto storage = routed_host;
                if (call.join == 4) storage.insert(storage.begin(), 99.F);
                routed->copy_from_host(storage.data(), storage.size() * sizeof(float));
            }
        };
        const auto launch = [&] {
            if (call.join)
                ops::detail::swiglu_ffn::launch_scaled_fp8_ffn_join(
                    x, weights, y,
                    {routed ? static_cast<const float*>(routed->data()) + (call.join == 4)
                            : nullptr,
                     call.join >= 3 ? nullptr : static_cast<const int*>(validity->data())},
                    *math.clamp_limit, workspace, cuda.stream());
            else
                ops::swiglu(x, math, weights, y, workspace, cuda);
        };
        prepare_join(0);
        input.copy_from_host(initial.data(), input.bytes());
        launch();
        cuda_synchronize(cuda.stream());
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        if (call.graph) {
            definition.capture(cuda.stream(), launch);
            graph.instantiate(definition);
        }
        for (int phase = 0; phase < (call.graph ? 2 : 1); ++phase) {
            auto values = initial;
            if (phase)
                for (auto& value : values) value ^= 0x8000;
            input.copy_from_host(values.data(), input.bytes());
            prepare_join(phase);
            scratch.fill(0xa5);
            output.repaint(cuda.stream());
            cuda_synchronize();
            if (call.graph)
                graph.launch(cuda.stream());
            else
                launch();
            cuda_synchronize(cuda.stream());
            if (workspace.used() != 64 || workspace.peak_used() > capacity + 256) ++failures;
            const auto actual = output.values();
            for (const auto value : actual)
                if (!std::isfinite(value)) {
                    ++failures;
                    break;
                }
            const auto rows   = call.tokens <= 8 ? linear_common::all_indices(Hidden)
                                                 : input_projection::sampled_rows(Hidden, 41);
            const auto tokens = call.tokens <= 33 ? linear_common::all_indices(call.tokens)
                                                  : input_projection::sampled_rows(call.tokens, 9);
            std::vector<double> got, reference;
            for (const auto t : tokens)
                for (const auto row : rows) {
                    got.push_back(actual[std::size_t(t) * Hidden + row]);
                    reference.push_back(oracle.output(row, t, phase) +
                                        (call.join && valid_host[t]
                                             ? routed_host[std::size_t(t) * Hidden + row]
                                             : 0.0));
                }
            failures += verify_reduction(label, got, reference,
                                         profile.policy == ops::LinearPolicy::AllowA8 ? kA8 : kA16);
            failures += linear_common::verify_preserved(input.data(),
                                                        std::as_bytes(std::span(values)), label);
            failures += output.verify_guards(label) + scratch.verify_guards(label);
            if (validity) {
                failures += validity->verify_guards(label);
                failures += linear_common::verify_preserved(
                    validity->data(), std::as_bytes(std::span(valid_host)), label);
            }
            if (routed) {
                failures += routed->verify_guards(label);
                auto storage = routed_host;
                if (call.join == 4) storage.insert(storage.begin(), 99.F);
                failures += linear_common::verify_preserved(
                    routed->data(), std::as_bytes(std::span(storage)), label);
            }
            std::array<std::byte, 64> prefix;
            prefix.fill(std::byte{0xa5});
            failures += linear_common::verify_preserved(scratch.data(), prefix, label);
        }
    }
    return failures + input.verify_guards(profile.name) + gate_up.verify_preserved(profile.name) +
           down.verify_preserved(profile.name);
}
} // namespace

void add_fp8_cases(OpSuite& suite) {
#if NINFER_TARGET_SPARK
    for (const Profile profile :
         {Profile{"mxfp8-dense", QType::MXFP8, 6144, ops::LinearPolicy::AllowA8},
          Profile{"mxfp8-shared", QType::MXFP8, 1024, ops::LinearPolicy::AllowA8},
          Profile{"fp8-block-fp32", QType::FP8_E4M3FN_BLOCK128X128_FP32, 1024,
                  ops::LinearPolicy::A16Only}}) {
        const bool dense  = profile.format == QType::MXFP8 && profile.intermediate == 6144;
        const bool block  = profile.format == QType::FP8_E4M3FN_BLOCK128X128_FP32;
        const bool shared = profile.format == QType::MXFP8 && profile.intermediate == 1024;
        std::vector<int> tokens =
            dense ? std::vector<int>{1,    2,    3,    4,    5,    7,    8,   9,   15,  16,   17,
                                     24,   31,   32,   33,   63,   64,   65,  95,  96,  97,   127,
                                     128,  129,  191,  192,  193,  255,  256, 257, 383, 384,  385,
                                     511,  512,  513,  639,  640,  641,  767, 768, 769, 1023, 1024,
                                     1025, 2047, 2048, 2049, 4095, 4096, 4097}
            : (block || shared)
                ? std::vector<int>{1,    2,    3,    4,    5,    8,    9,    15,   16,  17,  24,
                                   31,   32,   33,   47,   48,   49,   63,   64,   65,  95,  96,
                                   97,   127,  128,  129,  191,  192,  193,  255,  256, 257, 319,
                                   320,  321,  383,  384,  385,  511,  512,  513,  767, 768, 769,
                                   1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096, 4097}
                : std::vector<int>{1, 2, 3, 5, 65, 128, 129, 4096};
        if (shared) tokens.insert(tokens.end(), {1152, 1153, 1536, 1537, 1664, 1665, 1920, 1921});
        std::vector<Call> calls;
        for (int t : tokens) calls.push_back({t, false});
        if (dense || block || shared)
            for (int t : {1, 8, 9, 17, 33, 129, 193, 257, 385, 513, 641, 2049, 4097})
                calls.push_back({t, true});
        else
            calls.push_back({129, true});
        if (dense || block || shared) {
            for (int t : {1, 3, 9, 17, 24, 25, 33, 65, 129, 193, 257, 385, 513, 4097})
                calls.push_back({t, true, 1});
            for (int t : {1, 3, 33, 193, 4095}) calls.push_back({t, true, 2});
            for (int t : {1, 3, 17, 32, 33, 257, 4096}) calls.push_back({t, true, 3});
        }
        if (block) {
            calls.push_back({8, true, 1});
            calls.push_back({8, true, 3});
        }
        if (profile.format == QType::FP8_E4M3FN_BLOCK128X128_FP32) calls.push_back({127, true});
        if (shared)
            for (int t : {385, 513}) calls.push_back({t, true, 4});
        std::vector<OpCase> cases;
        for (const auto& call : calls)
            cases.push_back(
                {std::string("swiglu/") + profile.name + "/T" + std::to_string(call.tokens) +
                     (call.graph ? "/graph" : "/eager") +
                     (call.join == 1   ? "/join-mixed"
                      : call.join == 2 ? "/join-zero"
                      : call.join == 3 ? "/join-all"
                      : call.join == 4 ? "/join-unaligned"
                                       : ""),
                 {{"suite", "swiglu"},
                  {"format", profile.name},
                  {"policy", profile.policy == ops::LinearPolicy::AllowA8 ? "a8" : "a16"},
                  {"execution", call.graph ? "graph" : "eager"},
                  {"t", std::to_string(call.tokens)}}});
        suite.add(std::move(cases), [profile, calls](auto selected) {
            std::vector<Call> chosen;
            for (const auto i : selected) chosen.push_back(calls[i]);
            return run(profile, chosen);
        });
        if (dense || block || shared) {
            suite.add({std::string("swiglu/") + profile.name + "/capacity-intervals",
                       {{"suite", "swiglu"},
                        {"format", profile.name},
                        {"execution", "eager"},
                        {"policy", profile.policy == ops::LinearPolicy::AllowA8 ? "a8" : "a16"},
                        {"t", "0"}}},
                      [profile] {
                          ops::SwiGLUProfile p{
                              Hidden,
                              profile.intermediate,
                              profile.format,
                              profile.format,
                              profile.format == QType::MXFP8 ? QuantLayout::RowGroupScale
                                                             : QuantLayout::BlockScaleM128K128,
                              profile.format == QType::MXFP8 ? QuantLayout::RowGroupScale
                                                             : QuantLayout::BlockScaleM128K128,
                              profile.policy,
                              profile.policy,
                              10.F};
                          int failures = 0;
                          for (auto [first, last] :
                               {std::pair{1, 33}, std::pair{17, 48}, std::pair{17, 65},
                                std::pair{33, 49}, std::pair{97, 193}, std::pair{129, 257},
                                std::pair{193, 385}, std::pair{1, 4097}}) {
                              const auto capacity =
                                  ops::swiglu_workspace_capacity_bytes(p, first, last);
                              for (int t = first; t <= last; ++t)
                                  if (ops::swiglu_workspace_capacity_bytes(p, t, t) > capacity)
                                      ++failures;
                          }
                          return failures;
                      });
        }
        if (block) {
            for (int t : {17, 32, 65, 193, 513})
                for (bool small : {false, true})
                    suite.add({std::string("swiglu/") + profile.name + "/T" + std::to_string(t) +
                                   (small ? "/small-product-large-down" : "/saturated-product"),
                               {{"suite", "swiglu"},
                                {"format", profile.name},
                                {"policy", "a16"},
                                {"execution", "graph"},
                                {"t", std::to_string(t)}}},
                              [profile, t, small] {
                                  auto scaled             = profile;
                                  scaled.activation_scale = small ? std::ldexp(1.F, -68) : 1e10F;
                                  scaled.down_extent      = small ? 1e38F : .05F;
                                  const std::array calls{Call{t, true}};
                                  return run(scaled, calls);
                              });
        }
    }
#endif
}

} // namespace ninfer::test::swiglu
