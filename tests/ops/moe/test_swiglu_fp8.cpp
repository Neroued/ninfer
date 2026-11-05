#include "core/decode_graph.h"
#include "core/platform.h"
#include "ninfer/ops/moe.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear_common.h"
#include "swiglu_oracle.h"

#include <array>

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
};

struct Call {
    int tokens;
    bool graph;
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
    input_projection::DevicePackedWeight down(qw::make_patterned_weight(
        profile.format, Hidden, profile.intermediate, 9133, {.native_extent = .05f}));
    auto initial = activation(maximum);
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
                           std::to_string(call.tokens) + (call.graph ? "/graph" : "/eager");
        const auto capacity = ops::swiglu_workspace_capacity_bytes(math, call.tokens, call.tokens);
        if (capacity > interval) ++failures;
        GuardedDeviceBuffer scratch(capacity + 256);
        MemoryArena workspace(scratch.span());
        (void)workspace.alloc_bytes(64, 1);
        input_projection::GuardedBf16Tensor output(Hidden, call.tokens);
        Tensor x(input.span(), DType::BF16, {Hidden, call.tokens});
        auto y            = output.tensor();
        const auto launch = [&] { ops::swiglu(x, math, weights, y, workspace, cuda); };
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
                    reference.push_back(oracle.output(row, t, phase));
                }
            failures += verify_reduction(label, got, reference,
                                         profile.policy == ops::LinearPolicy::AllowA8 ? kA8 : kA16);
            failures += linear_common::verify_preserved(input.data(),
                                                        std::as_bytes(std::span(values)), label);
            failures += output.verify_guards(label) + scratch.verify_guards(label);
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
        const std::vector<int> tokens =
            profile.format == QType::FP8_E4M3FN_BLOCK128X128_FP32
                ? std::vector<int>{1, 2, 3, 5, 9, 17, 33, 64, 65, 127, 128, 129, 4096}
                : std::vector<int>{1, 2, 3, 5, 65, 128, 129, 4096};
        std::vector<Call> calls;
        for (int t : tokens) calls.push_back({t, false});
        calls.push_back({129, true});
        if (profile.format == QType::FP8_E4M3FN_BLOCK128X128_FP32) calls.push_back({127, true});
        std::vector<OpCase> cases;
        for (const auto& call : calls)
            cases.push_back(
                {std::string("swiglu/") + profile.name + "/T" + std::to_string(call.tokens) +
                     (call.graph ? "/graph" : "/eager"),
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
    }
#endif
}

} // namespace ninfer::test::swiglu
