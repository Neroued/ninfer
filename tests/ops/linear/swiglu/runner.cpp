#include "core/weight.h"
#include "cuda_fixture.h"
#include "ops/linear/swiglu/runner.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear_common.h"
#include "ops/linear/weights.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::test::linear_swiglu {
namespace {

using linear_common::checked_elements;

// The public policy determines the criterion, independently of the selected arithmetic.
constexpr ReductionCriterion tolerance_for(ops::LinearPolicy policy, bool clamped = false) {
    if (clamped) {
        // The tiny absolute allowance also detects an erroneous lower gate clamp at -10.
        // Policy, rather than the private arithmetic or token count, fixes the criterion.
        if (policy == ops::LinearPolicy::AllowA4) return {1.6e-1, 1.0e-7, 1.6e-1};
        if (policy == ops::LinearPolicy::AllowA8) return {8.0e-2, 1.0e-7, 1.2e-1};
        return {1.0 / 256, 1.0e-7, 2.0 / 256};
    }
    switch (policy) {
    case ops::LinearPolicy::A16Only:
        return {3.3e-3, 5.0e-3, 6.3e-3};
    case ops::LinearPolicy::AllowA8:
        // Both independently A8-quantized projections feed the nonlinear product, so this profile
        // allows twice Linear's relative-L2 quantization allowance plus a bounded gross tail.
        return {8.0e-2, 1.0e-2, 1.2e-1};
    case ops::LinearPolicy::AllowA4:
        return {1.6e-1, 1.0e-2, 1.6e-1};
    }
    throw std::invalid_argument("linear_swiglu test: unknown activation compute profile");
}

std::uint64_t mix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::vector<std::uint16_t> make_activation(const Profile& profile, std::int32_t tokens) {
    std::vector<std::uint16_t> activation(
        checked_elements(profile.input_rows, tokens, "activation size"), 0);

    // Token zero is dense, so every logical K contributes to every production implementation.
    // Later tokens use exact zeros plus a rotating set of nonzeros. This keeps a full-output,
    // full-formula oracle practical at large registered T boundaries without adopting any
    // production staging or reduction behavior.
    const bool native_float_weight =
        profile.qtype == QType::NVFP4 || profile.qtype == QType::FP8_E4M3FN_ROW_BF16;
    const float dense_scale =
        profile.qtype == QType::BF16
            ? 2.0e-2F
            : (profile.qtype == QType::Q4_G64_FP16 ? 1.25e-4F
                                                   : (native_float_weight ? 1.0e-3F : 1.0e-5F));
    for (std::int32_t column = 0; column < profile.input_rows; ++column) {
        const std::uint64_t mixed = mix64((static_cast<std::uint64_t>(profile.seed) << 32) |
                                          static_cast<std::uint32_t>(column));
        int numerator             = static_cast<int>((mixed >> 48) % 63U) - 31;
        if (numerator == 0) { numerator = (column & 1) == 0 ? 1 : -1; }
        const float value                            = static_cast<float>(numerator) * dense_scale;
        activation[static_cast<std::size_t>(column)] = test::f32_to_bf16(value);
    }

    constexpr std::int32_t kNonzerosPerSparseToken = 4;
    const float sparse_scale =
        profile.qtype == QType::BF16
            ? 5.0e-1F
            : (profile.qtype == QType::Q4_G64_FP16 ? 1.5e-2F
                                                   : (native_float_weight ? 2.0e-2F : 1.5e-3F));
    for (std::int32_t token = 1; token < tokens; ++token) {
        for (std::int32_t lane = 0; lane < kNonzerosPerSparseToken; ++lane) {
            const std::uint64_t mixed =
                mix64((static_cast<std::uint64_t>(profile.seed + 1U) << 32) |
                      static_cast<std::uint32_t>(token * kNonzerosPerSparseToken + lane));
            const std::int32_t column =
                static_cast<std::int32_t>(mixed % static_cast<std::uint64_t>(profile.input_rows));
            int numerator = static_cast<int>((mixed >> 48) % 31U) - 15;
            if (numerator == 0) { numerator = (lane & 1) == 0 ? 1 : -1; }
            const float value = static_cast<float>(numerator) * sparse_scale;
            activation[static_cast<std::size_t>(token) * profile.input_rows + column] =
                test::f32_to_bf16(value);
        }
    }
    return activation;
}

struct ActiveValue {
    std::int32_t token;
    double value;
};

std::vector<std::vector<ActiveValue>>
index_nonzero_activations(const std::vector<std::uint16_t>& activation, std::int32_t input_rows,
                          std::int32_t tokens) {
    std::vector<std::vector<ActiveValue>> by_column(static_cast<std::size_t>(input_rows));
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < input_rows; ++column) {
            const std::uint16_t bits =
                activation[static_cast<std::size_t>(token) * input_rows + column];
            if ((bits & 0x7fffU) == 0U) { continue; }
            by_column[static_cast<std::size_t>(column)].push_back(
                {token, static_cast<double>(test::bf16_to_f32(bits))});
        }
    }
    return by_column;
}

double silu_fp64(double value) {
    if (value >= 0.0) { return value / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return value * exponential / (1.0 + exponential);
}

std::vector<double> linear_swiglu_oracle_fp64(const Profile& profile,
                                              const quantized_weight::PackedWeight& weight,
                                              const std::vector<std::uint16_t>& activation,
                                              std::int32_t tokens) {
    const auto active_by_column = index_nonzero_activations(activation, profile.input_rows, tokens);
    std::vector<double> output(checked_elements(profile.output_rows, tokens, "oracle output size"));

    ninfer::test::parallel_rows(
        profile.output_rows,
        static_cast<std::uint64_t>(profile.output_rows) * profile.input_rows * tokens,
        [&](int row_begin, int row_end) {
            std::vector<double> gate(static_cast<std::size_t>(tokens));
            std::vector<double> up(static_cast<std::size_t>(tokens));
            for (std::int32_t row = row_begin; row < row_end; ++row) {
                std::fill(gate.begin(), gate.end(), 0.0);
                std::fill(up.begin(), up.end(), 0.0);
                const std::int32_t up_row = profile.output_rows + row;
                for (std::int32_t column = 0; column < profile.input_rows; ++column) {
                    const double gate_weight =
                        quantized_weight::logical_weight_fp64(weight, row, column);
                    const double up_weight =
                        quantized_weight::logical_weight_fp64(weight, up_row, column);
                    for (const ActiveValue active :
                         active_by_column[static_cast<std::size_t>(column)]) {
                        gate[static_cast<std::size_t>(active.token)] += gate_weight * active.value;
                        up[static_cast<std::size_t>(active.token)] += up_weight * active.value;
                    }
                }
                for (std::int32_t token = 0; token < tokens; ++token) {
                    const double g     = gate[static_cast<std::size_t>(token)];
                    const double u     = up[static_cast<std::size_t>(token)];
                    const double limit = profile.clamp_limit;
                    const double fused =
                        limit > 0 ? silu_fp64(std::min(g, limit)) * std::clamp(u, -limit, limit)
                                  : silu_fp64(g) * u;
                    output[static_cast<std::size_t>(token) * profile.output_rows + row] = fused;
                }
            }
        });
    return output;
}

std::vector<double> read_bf16_output(const test::GuardedDeviceBuffer& output,
                                     std::size_t elements) {
    std::vector<std::uint16_t> bits(elements);
    cuda_check(cudaMemcpy(bits.data(), output.data(), elements * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "copy LinearSwiGLU output");
    std::vector<double> values(elements);
    for (std::size_t index = 0; index < elements; ++index) {
        values[index] = static_cast<double>(test::bf16_to_f32(bits[index]));
    }
    return values;
}

int compare_output(std::string_view label, const std::vector<double>& actual,
                   const double* reference, ops::LinearPolicy policy, bool clamped = false) {
    if (actual.empty() || reference == nullptr) {
        std::cerr << label << ": invalid numerical comparison\n";
        return 1;
    }

    return verify_reduction(label, actual, std::span<const double>(reference, actual.size()),
                            tolerance_for(policy, clamped));
}

int verify_unchanged(std::string_view label, const test::GuardedDeviceBuffer& device,
                     const void* expected, std::size_t bytes) {
    if (device.bytes() != bytes) {
        std::cerr << label << ": read-only input size mismatch\n";
        return 1;
    }
    return linear_common::verify_preserved(
        device.data(), std::span(static_cast<const std::byte*>(expected), bytes), label);
}

void validate_profile(const Profile& profile) {
    if (profile.input_rows <= 0 || profile.output_rows <= 0 ||
        profile.gate_up_rows != 2 * static_cast<std::int64_t>(profile.output_rows)) {
        throw std::invalid_argument("linear_swiglu test: inconsistent projection dimensions");
    }
}

void validate_tokens(std::span<const std::int32_t> token_cases) {
    if (token_cases.empty()) { throw std::invalid_argument("linear_swiglu test: no token cases"); }
    for (std::size_t index = 0; index < token_cases.size(); ++index) {
        if (token_cases[index] <= 0 ||
            (index != 0 && token_cases[index] <= token_cases[index - 1])) {
            throw std::invalid_argument(
                "linear_swiglu test: token cases must be positive and increasing");
        }
    }
}

} // namespace

static int run_profile(std::string_view label, const Profile& profile,
                       std::span<const Invocation> calls) {
    validate_profile(profile);
    int maximum_tokens = 0;
    bool has_graph     = false;
    for (const auto& call : calls) {
        maximum_tokens = std::max(maximum_tokens, call.tokens);
        has_graph |= call.graph;
    }
    quantized_weight::PatternedWeightOptions weight_options;
    if (profile.qtype == QType::NVFP4) {
        weight_options.weight_scale_divisor = 0.125F;
        weight_options.input_scale_divisor  = 3.5F;
    }
    quantized_weight::PackedWeight host_weight =
        profile.qtype == QType::BF16
            ? linear_weights::make_bf16_weight(profile.gate_up_rows, profile.input_rows,
                                               profile.seed)
            : quantized_weight::make_patterned_weight(profile.qtype, profile.gate_up_rows,
                                                      profile.input_rows, profile.seed,
                                                      weight_options);
    const std::vector<std::uint16_t> host_activation = make_activation(profile, maximum_tokens);
    const std::vector<double> reference =
        linear_swiglu_oracle_fp64(profile, host_weight, host_activation, maximum_tokens);
    std::vector<std::uint16_t> negative_activation;
    std::vector<double> negative_reference;
    std::optional<DeviceContext> graph_context;
    if (has_graph) {
        negative_activation = host_activation;
        for (auto& bits : negative_activation) bits ^= 0x8000;
        negative_reference =
            linear_swiglu_oracle_fp64(profile, host_weight, negative_activation, maximum_tokens);
        graph_context.emplace();
    }

    test::GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.span());

    test::GuardedDeviceBuffer device_activation(host_activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(host_activation.data(),
                                     host_activation.size() * sizeof(std::uint16_t));

    const ops::LinearPolicy policy = profile.policy;
    int admitted_maximum           = 0;
    for (const auto& call : calls)
        if (call.form != CallForm::RejectA16Extent)
            admitted_maximum = std::max(admitted_maximum, call.tokens);
    const auto interval_capacity = admitted_maximum == 0
                                       ? 0
                                       : ops::linear_swiglu_workspace_capacity_bytes(
                                             profile.qtype, profile.gate_up_rows,
                                             profile.input_rows, policy, 1, admitted_maximum);
    int failures                 = 0;
    const auto run_case          = [&](const Invocation& call) {
        const auto tokens   = call.tokens;
        const auto replay   = call.graph;
        const auto elements = checked_elements(profile.output_rows, tokens, "output size");
        test::GuardedDeviceBuffer output(elements * sizeof(std::uint16_t));
        Tensor x(device_activation.span(), DType::BF16, {profile.input_rows, tokens});
        Tensor destination(output.span(), DType::BF16, {profile.output_rows, tokens});
        const auto capacity = call.form == CallForm::RejectA16Extent
                                           ? 0
                                           : ops::linear_swiglu_workspace_capacity_bytes(
                                        profile.qtype, profile.gate_up_rows, profile.input_rows,
                                        policy, tokens, tokens);
        if (capacity > interval_capacity)
            throw std::runtime_error("SwiGLU interval capacity is insufficient");
        GuardedDeviceBuffer scratch(capacity);
        MemoryArena workspace(scratch.span());
        if (tokens == 1 && !replay &&
            (profile.qtype == QType::NVFP4 || profile.qtype == QType::FP8_E4M3FN_ROW_BF16)) {
            for (const void* alias : std::array<const void*, 3>{x.data, weight.qdata, weight.scales}) {
                auto invalid = destination;
                invalid.data = const_cast<void*>(alias);
                try {
                    ops::linear_swiglu(x, weight, invalid, policy, workspace,
                                       cuda_environment().default_execution);
                    std::cerr << call.id << ": accepted aliased output\n";
                    ++failures;
                } catch (const std::invalid_argument&) {}
            }
        }
        if (call.form == CallForm::RejectA16Extent) {
            for (bool convenience : {false, true}) {
                try {
                    if (convenience)
                        ops::linear_swiglu(x, weight, destination, workspace,
                                                    cuda_environment().default_execution);
                    else
                        ops::linear_swiglu(x, weight, destination, policy, workspace,
                                                    cuda_environment().default_execution);
                    std::cerr << call.id << ": accepted NVFP4 A16 T=17\n";
                    ++failures;
                } catch (const std::invalid_argument&) {}
            }
            return;
        }
        if (profile.clamp_limit > 0 && tokens == 1 && !replay) {
            output.fill(0x5a);
            for (float invalid : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::quiet_NaN()}) {
                try {
                    ops::linear_swiglu(x, weight, destination, invalid, policy, workspace,
                                                cuda_environment().default_execution);
                    ++failures;
                } catch (const std::invalid_argument&) {}
            }
            std::vector<std::uint8_t> preserved(output.bytes(), 0x5a);
            failures += verify_unchanged("invalid clamp preserves output", output, preserved.data(),
                                                  preserved.size());
        }
        workspace.reset();
        workspace.reset_peak();
        const auto cuda =
            replay ? graph_context->execution_view() : cuda_environment().default_execution;
        const auto stream = cuda.stream();
        const auto launch = [&] {
            if (profile.clamp_limit > 0)
                ops::linear_swiglu(x, weight, destination, profile.clamp_limit, policy, workspace,
                                            cuda);
            else if (call.form == CallForm::Convenience)
                ops::linear_swiglu(x, weight, destination, workspace, cuda);
            else
                ops::linear_swiglu(x, weight, destination, policy, workspace, cuda);
        };
        const auto& label_case = call.id;
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        try {
            test::cuda_check(cudaDeviceSynchronize(), "finish fixture initialization");
            if (replay) {
                definition.capture(stream, launch);
                graph.instantiate(definition);
            }
            for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
                const auto& input_bits = phase ? negative_activation : host_activation;
                const auto& expected   = phase ? negative_reference : reference;
                if (replay)
                    device_activation.copy_from_host(input_bits.data(),
                                                              input_bits.size() * sizeof(std::uint16_t));
                output.fill(0xff);
                test::cuda_check(cudaDeviceSynchronize(), "finish replay input update");
                if (replay)
                    graph.launch(stream);
                else
                    launch();
                test::cuda_check(cudaStreamSynchronize(stream), "synchronize LinearSwiGLU");
                const auto exact = ops::linear_swiglu_workspace_capacity_bytes(
                    profile.qtype, profile.gate_up_rows, profile.input_rows, policy, tokens,
                    tokens);
                if (workspace.used() != 0 || workspace.peak_used() != exact) {
                    std::cerr << label_case << ": exact workspace query/execution mismatch\n";
                    ++failures;
                }
                failures += output.verify_guards(label_case);
                failures += scratch.verify_guards(label_case + " workspace");
                const auto actual = read_bf16_output(output, elements);
                failures += compare_output(label_case, actual, expected.data(), profile.policy,
                                                    profile.clamp_limit > 0);
                if (replay)
                    failures += verify_unchanged(label_case + " input", device_activation,
                                                          input_bits.data(),
                                                          input_bits.size() * sizeof(std::uint16_t));
            }
        } catch (const std::exception& error) {
            std::cerr << label_case << ": unexpected exception: " << error.what() << '\n';
            ++failures;
        }
    };
    for (const auto& call : calls)
        if (!call.graph) run_case(call);
    if (has_graph)
        failures += verify_unchanged(std::string(label) + " eager input", device_activation,
                                     host_activation.data(), device_activation.bytes());
    for (const auto& call : calls)
        if (call.graph) run_case(call);
    const auto& final_input = has_graph ? negative_activation : host_activation;
    failures += device_activation.verify_guards(std::string(label) + " activation");
    failures += device_weight.verify_guards(std::string(label) + " weight");
    failures += verify_unchanged(std::string(label) + " activation", device_activation,
                                 final_input.data(), final_input.size() * sizeof(std::uint16_t));
    failures += verify_unchanged(std::string(label) + " weight", device_weight,
                                 host_weight.payload.data(), host_weight.payload.size());
    return failures;
}

int run_negative_gate(std::string_view label, const quantized_weight::PackedWeight& packed,
                      ops::LinearPolicy policy, std::span<const std::int32_t> token_cases,
                      std::span<const std::int32_t> active_columns, float clamp_limit = 0.0f) {
    validate_tokens(token_cases);
    const std::int32_t rows           = packed.weight.n;
    const std::int32_t k              = packed.weight.k;
    const std::int32_t half           = rows / 2;
    const std::int32_t maximum_tokens = token_cases.back();
    validate_profile({packed.weight.qtype, rows, k, half, 0, policy});

    std::vector<std::uint16_t> activation(checked_elements(k, maximum_tokens, "activation size"),
                                          0);
    double gate = 0.0;
    double up   = 0.0;
    for (const std::int32_t column : active_columns) {
        if (column < 0 || column >= k) {
            throw std::invalid_argument("linear_swiglu test: negative-gate activation column");
        }
        for (std::int32_t token = 0; token < maximum_tokens; ++token) {
            activation[static_cast<std::size_t>(token) * k + column] = test::f32_to_bf16(1.0F);
        }
        gate += quantized_weight::logical_weight_fp64(packed, 0, column);
        up += quantized_weight::logical_weight_fp64(packed, half, column);
    }
    const double limit = clamp_limit;
    const double ideal = limit > 0
                             ? silu_fp64(std::min(gate, limit)) * std::clamp(up, -limit, limit)
                             : silu_fp64(gate) * up;

    test::GuardedDeviceBuffer weights(packed.payload.size());
    test::GuardedDeviceBuffer input_storage(activation.size() * sizeof(std::uint16_t));
    weights.copy_from_host(packed.payload.data(), packed.payload.size());
    input_storage.copy_from_host(activation.data(), input_storage.bytes());
    const Weight weight = packed.device_weight(weights.span());
    int failures        = 0;
    for (const std::int32_t tokens : token_cases) {
        const auto capacity = ops::linear_swiglu_workspace_capacity_bytes(weight.qtype, rows, k,
                                                                          policy, tokens, tokens);
        GuardedDeviceBuffer scratch(capacity);
        MemoryArena workspace(scratch.span());
        GuardedDeviceBuffer output_storage(checked_elements(half, tokens, "output size") *
                                           sizeof(std::uint16_t));
        output_storage.fill(0xff);

        Tensor input(input_storage.span(), DType::BF16, {k, tokens});
        Tensor output(output_storage.span(), DType::BF16, {half, tokens});
        if (clamp_limit > 0)
            ops::linear_swiglu(input, weight, output, clamp_limit, policy, workspace,
                               ninfer::test::cuda_environment().default_execution);
        else
            ops::linear_swiglu(input, weight, output, policy, workspace,
                               ninfer::test::cuda_environment().default_execution);
        test::cuda_synchronize();
        const auto elements = checked_elements(half, tokens, "output size");
        const auto actual   = read_bf16_output(output_storage, elements);
        const std::vector<double> reference(elements, ideal);
        failures +=
            verify_reduction(std::string(label) + " negative gate T=" + std::to_string(tokens),
                             actual, reference, tolerance_for(policy, clamp_limit > 0));
        failures += output_storage.verify_guards(label);
        failures += scratch.verify_guards(label);
        if (workspace.used() != 0 || workspace.peak_used() > capacity) ++failures;
    }
    return failures;
}

void add_profile(Suite& suite, const Profile& profile, std::span<const std::int32_t> tokens,
                 std::span<const std::int32_t> graphs) {
    std::vector<Invocation> calls;
    for (int t : tokens) calls.push_back({t});
    for (int t : graphs) calls.push_back({t, true});
    if (profile.policy == ops::LinearPolicy::A16Only) {
        if (profile.clamp_limit == 0)
            calls.push_back({profile.qtype == QType::NVFP4 ? 16 : tokens.front(), false,
                             CallForm::Convenience});
        if (profile.qtype == QType::NVFP4) calls.push_back({17, false, CallForm::RejectA16Extent});
    }
    std::vector<linear_suite::Case> cases;
    for (auto& call : calls) {
        const std::string recipe = "s" + std::to_string(profile.seed) +
                                   (profile.clamp_limit > 0 ? "/clamped/" : "/dense-sparse/") +
                                   (call.form == CallForm::Convenience       ? "convenience"
                                    : call.form == CallForm::RejectA16Extent ? "reject-extent"
                                                                             : "policy");
        auto test = linear_suite::make_case(
            linear_suite::format_name(profile.qtype), profile.gate_up_rows, profile.input_rows,
            call.tokens, profile.policy,
            call.graph ? linear_suite::Execution::Graph : linear_suite::Execution::Eager, recipe);
        cases.push_back(std::move(test));
    }
    suite.add(std::move(cases), [profile, calls = std::move(calls)](auto selected, auto declared) {
        std::vector<Invocation> chosen;
        for (auto index : selected) {
            chosen.push_back(calls[index]);
            chosen.back().id = declared[index].id;
        }
        return run_profile(chosen.front().id, profile, chosen);
    });
}

void add_negative_gate(Suite& suite, const Profile& profile, std::span<const std::int32_t> tokens,
                       std::span<const std::int32_t> active_columns,
                       std::function<quantized_weight::PackedWeight()> make_weight) {
    std::vector<int> owned_tokens(tokens.begin(), tokens.end()),
        owned_columns(active_columns.begin(), active_columns.end());
    std::vector<linear_suite::Case> cases;
    for (int t : tokens)
        cases.push_back(linear_suite::make_case(linear_suite::format_name(profile.qtype),
                                                profile.gate_up_rows, profile.input_rows, t,
                                                profile.policy, linear_suite::Execution::Eager,
                                                "negative-gate/s" + std::to_string(profile.seed)));
    suite.add(std::move(cases),
              [profile, owned_tokens, owned_columns, make_weight](auto selected, auto declared) {
                  std::vector<int> chosen;
                  for (auto index : selected) chosen.push_back(owned_tokens[index]);
                  return run_negative_gate("negative-gate", make_weight(), profile.policy, chosen,
                                           owned_columns, profile.clamp_limit);
              });
}

} // namespace ninfer::test::linear_swiglu
