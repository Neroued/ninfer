#include "ops/linear/split/runner.h"

#include "core/decode_graph.h"
#include "cuda_fixture.h"
#include "ninfer/ops/linear_split.h"
#include "ninfer/ops/weight_input.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear/weights.h"

#include <memory>
#include <numeric>
#include <set>

namespace ninfer::test::linear_split {
namespace {

using input_projection::DevicePackedWeight;

struct Outputs {
    std::vector<std::unique_ptr<GuardedDeviceBuffer>> backing;
    std::vector<Tensor> views;
    std::vector<void*> gaps;

    Outputs(const Profile& profile, int tokens, bool shared) {
        std::uint16_t* next = nullptr;
        if (shared) {
            std::size_t words = 0;
            for (int rows : profile.outputs) words += std::size_t(rows) * tokens + 128;
            backing.push_back(std::make_unique<GuardedDeviceBuffer>(words * 2));
            next = static_cast<std::uint16_t*>(backing.back()->data());
        }
        for (int rows : profile.outputs) {
            void* data = next;
            if (shared) {
                next += std::size_t(rows) * tokens;
                gaps.push_back(next);
                next += 128;
            } else {
                backing.push_back(
                    std::make_unique<GuardedDeviceBuffer>(std::size_t(rows) * tokens * 2));
                data = backing.back()->data();
            }
            views.emplace_back(data, backing.back()->device(), DType::BF16,
                               std::initializer_list<std::int32_t>{rows, tokens});
        }
    }

    void poison() {
        for (auto& buffer : backing) buffer->fill(0xff);
    }

    int verify_guards(std::string_view label) const {
        int failures = 0;
        for (const auto& buffer : backing) failures += buffer->verify_guards(label);
        for (const void* gap : gaps)
            failures += verify_exact((std::string(label) + " gap").c_str(),
                                     from_device<std::uint16_t>(gap, 128),
                                     std::vector<std::uint16_t>(128, 0xffff));
        return failures;
    }
};

ReductionCriterion criterion(const Profile& profile, ops::LinearPolicy policy) {
    switch (policy) {
    case ops::LinearPolicy::A16Only:
        return profile.a16;
    case ops::LinearPolicy::AllowA8:
        return {0.04, 1.0 / 256.0, 0.06};
    case ops::LinearPolicy::AllowA4:
        return {0.16, profile.a4_absolute, 0.16};
    }
    throw std::invalid_argument("unknown Linear policy");
}

std::vector<int> columns_to_check(const Invocation& call) {
    if (call.columns == Columns::All || (call.columns != Columns::Seven && call.t <= 32))
        return linear_common::all_indices(call.t);
    if (call.columns == Columns::Seven) return linear_common::sampled_indices(call.t);
    std::set<int> columns{0, 1, call.t / 2, call.t - 2, call.t - 1};
    if (call.columns == Columns::HeadBoundaries) columns.insert({15, 16, 31, 32});
    return {columns.begin(), columns.end()};
}

int verify_output(const Profile& profile, const Invocation& call,
                  const std::vector<std::unique_ptr<DevicePackedWeight>>& parents,
                  const Outputs& outputs, const std::vector<float>& input) {
    int failures = outputs.verify_guards(call.id);
    std::vector<std::vector<std::uint16_t>> values;
    for (const auto& output : outputs.views) {
        values.push_back(
            from_device<std::uint16_t>(output.data, std::size_t(output.ne[0]) * call.t));
        if (std::any_of(values.back().begin(), values.back().end(),
                        [](auto bits) { return (bits & 0x7f80) == 0x7f80; })) {
            std::cerr << call.id << ": unwritten or non-finite output\n";
            ++failures;
        }
    }
    const auto columns = columns_to_check(call);
    for (const auto region : profile.regions) {
        auto rows = call.full_rows ? linear_common::all_indices(region.rows)
                                   : input_projection::sampled_rows(region.rows, call.sample_rows);
        if (!call.full_rows && region.head_dim) {
            for (int head = region.head_dim; head < region.rows; head += region.head_dim) {
                rows.push_back(head - 1);
                rows.push_back(head);
            }
            std::sort(rows.begin(), rows.end());
            rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
        }
        std::size_t output_index = 0;
        int output_begin         = 0;
        while (output_begin + profile.outputs[output_index] <= region.first)
            output_begin += profile.outputs[output_index++];
        std::vector<double> actual(rows.size() * columns.size()), expected(actual.size());
        ninfer::test::parallel_rows(
            rows.size(), std::uint64_t(rows.size()) * columns.size() * profile.k,
            [&](int begin, int end) {
                std::vector<double> weight(profile.k);
                for (int r = begin; r < end; ++r) {
                    int physical_row   = region.first + rows[r];
                    std::size_t parent = 0;
                    while (physical_row >= parents[parent]->host.weight.n)
                        physical_row -= parents[parent++]->host.weight.n;
                    for (int k = 0; k < profile.k; ++k)
                        weight[k] = quantized_weight::logical_weight_fp64(parents[parent]->host,
                                                                          physical_row, k);
                    for (std::size_t c = 0; c < columns.size(); ++c) {
                        const int token = columns[c];
                        double sum      = 0;
                        for (int k = 0; k < profile.k; ++k)
                            sum += weight[k] * double(input[std::size_t(token) * profile.k + k]);
                        expected[r * columns.size() + c] = sum;
                        actual[r * columns.size() + c] =
                            bf16_to_f32(values[output_index]
                                              [std::size_t(token) * profile.outputs[output_index] +
                                               region.first - output_begin + rows[r]]);
                    }
                }
            });
        failures += verify_reduction(call.id + " rows=" + std::to_string(region.first), actual,
                                     expected, criterion(profile, call.policy));
    }
    return failures;
}

ops::ProjectionWeights prepare(const Profile& profile, const Invocation& call,
                               const std::vector<std::unique_ptr<DevicePackedWeight>>& weights) {
    if (profile.binding == Binding::Direct) {
        if (weights.size() == 2)
            return ops::PairedProjectionWeights{weights[0]->view(), weights[1]->view()};
        return ops::SingleProjectionWeight{weights[0]->view(), call.policy};
    }
    std::vector<WeightParent> parents;
    parents.reserve(weights.size());
    for (const auto& source : weights) {
        const auto weight = source->view();
        const std::array shape{std::uint64_t(weight.n), std::uint64_t(weight.k)};
        const auto divisors = weight.qtype == QType::NVFP4
                                  ? std::span(&source->host.weight.weight_scale_divisor, 1)
                                  : std::span<const float>{};
        parents.push_back({weight_geometry(weight.qtype, weight.layout, shape), weight.device,
                           static_cast<const std::byte*>(weight.payload), divisors});
    }
    std::vector<WeightView> views;
    const auto& slices = profile.input_slices.empty() ? profile.outputs : profile.input_slices;
    std::size_t parent = 0;
    std::uint64_t row  = 0;
    for (int count : slices) {
        if (row == parents[parent].geometry.shape[0]) {
            ++parent;
            row = 0;
        }
        views.push_back({{std::uint64_t(count), std::uint64_t(profile.k)},
                         {{&parents[parent], row * profile.k, (row + count) * profile.k}}});
        row += count;
    }
    std::vector<ops::WeightInput> inputs;
    for (std::size_t i = 0; i < views.size(); ++i) {
        std::optional<float> divisor;
        if (profile.binding == Binding::Nvfp4Inputs && !call.omit_divisors)
            divisor = weights[0]->view().input_scale_divisor +
                      (call.policy == ops::LinearPolicy::AllowA4 ? 0.0F : float(i));
        inputs.push_back({views[i], call.policy, divisor});
    }
    if (profile.binding == Binding::Nvfp4Inputs && call.policy == ops::LinearPolicy::AllowA8) {
        auto mixed = inputs;
        for (std::size_t i = 0; i < mixed.size(); ++i)
            mixed[i].activation_input_divisor = weights[0]->view().input_scale_divisor + float(i);
        mixed[0].policy = ops::LinearPolicy::AllowA4;
        if (std::get<ops::SingleProjectionWeight>(ops::prepare_projection_weights(mixed)).policy !=
            call.policy)
            throw std::runtime_error("projection inputs lost their permission intersection");
    }
    if (profile.binding == Binding::LinearInputs) return ops::prepare_linear_weight(inputs);
    return ops::prepare_projection_weights(inputs);
}

template <class F>
int rejects(F&& call) {
    try {
        call();
    } catch (const std::invalid_argument&) { return 0; }
    std::cerr << "LinearSplit accepted invalid operands\n";
    return 1;
}

int validate(const Profile& profile, const Weight& weight, CudaExecutionView cuda) {
    DeviceBuffer input(cuda_environment().context.device(), profile.k * 2 * 2);
    Tensor x(input.span(), DType::BF16, {profile.k, 2});
    Outputs outputs(profile, 2, false);
    auto out          = outputs.views;
    const auto launch = [&](const Tensor& in, const Weight& w) {
        ops::linear_split(in, w, out, cuda);
    };
    int failures = 0;
    out[0].ne[0] -= 1;
    failures += rejects([&] { launch(x, weight); });
    out         = outputs.views;
    out[1].data = static_cast<std::byte*>(out[0].data) + 16;
    failures += rejects([&] { launch(x, weight); });
    out = outputs.views;
    for (const void* overlap : {x.data, const_cast<void*>(weight.qdata)}) {
        out[0].data = const_cast<void*>(overlap);
        failures += rejects([&] { launch(x, weight); });
    }
    out          = outputs.views;
    out[0].dtype = DType::FP16;
    failures += rejects([&] { launch(x, weight); });
    out          = outputs.views;
    auto invalid = weight;
    invalid.padded_shape[1] += 128;
    failures += rejects([&] { launch(x, invalid); });
    invalid       = weight;
    invalid.qtype = weight.qtype == QType::BF16 ? QType::Q8_G32_FP16 : QType::BF16;
    failures += rejects([&] { launch(x, invalid); });
    if (weight.qtype == QType::Q8_G32_FP16) {
        out[0].data = const_cast<void*>(weight.scales);
        failures += rejects([&] { launch(x, weight); });
        out           = outputs.views;
        invalid       = weight;
        invalid.group = 64;
        failures += rejects([&] { launch(x, invalid); });
    }
    auto strided = x;
    strided.nb[1] += 16;
    failures += rejects([&] { launch(strided, weight); });
    auto empty  = x;
    empty.ne[1] = 0;
    failures += rejects([&] { launch(empty, weight); });
    return failures;
}

int run(const Profile& profile, std::span<const Invocation> calls) {
    std::vector<std::unique_ptr<DevicePackedWeight>> weights;
    for (const auto& recipe : profile.weights) {
        auto packed = recipe.type == QType::BF16
                          ? linear_weights::make_bf16_weight(recipe.rows, profile.k, recipe.seed)
                          : quantized_weight::make_patterned_weight(
                                recipe.type, recipe.rows, profile.k, recipe.seed, recipe.options);
        weights.push_back(std::make_unique<DevicePackedWeight>(std::move(packed)));
    }
    DeviceContext context;
    int failures = 0;
    for (const auto& call : calls) {
        if (call.validate_operands) {
            failures += validate(profile, weights[0]->view(), context.execution_view());
            continue;
        }
        const auto prepared = prepare(profile, call, weights);
        if (call.workspace_intervals) {
            for (auto [first, last] : {std::pair{1, 48}, std::pair{1, 128}, std::pair{1, 1024}}) {
                const auto capacity = ops::linear_split_workspace_capacity_bytes(
                    prepared, profile.outputs, first, last);
                for (int t = first; t <= last; ++t)
                    if (ops::linear_split_workspace_capacity_bytes(prepared, profile.outputs, t,
                                                                   t) > capacity)
                        ++failures;
            }
            continue;
        }
        auto activation =
            input_projection::make_bf16_activation(profile.k, call.t, call.input_seed + call.t);
        auto bits  = input_projection::bf16_bits(activation);
        auto input = to_device(bits);
        Tensor x(input.span(), DType::BF16, {profile.k, call.t});
        Outputs output(profile, call.t, call.shared_output);
        const auto capacity =
            ops::linear_split_workspace_capacity_bytes(prepared, profile.outputs, call.t, call.t);
        GuardedDeviceBuffer scratch(capacity);
        MemoryArena workspace(scratch.span());
        const auto launch = [&] {
            if (call.convenience)
                ops::linear_split(x, weights[0]->view(), output.views, context.execution_view());
            else
                ops::linear_split(x, prepared, output.views, workspace, context.execution_view());
        };
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        cuda_synchronize();
        if (call.eager_before_capture) {
            launch();
            context.synchronize();
        }
        if (call.graph) {
            definition.capture(context.stream(), launch);
            graph.instantiate(definition);
        }
        for (int phase = 0; phase < (call.graph ? 2 : 1); ++phase) {
            if (phase) {
                if (call.changed_seed)
                    activation = input_projection::make_bf16_activation(profile.k, call.t,
                                                                        call.changed_seed + call.t);
                else
                    for (auto& value : activation) value = -value;
                bits = input_projection::bf16_bits(activation);
                input.copy_from_host(bits.data(), input.size());
            }
            output.poison();
            scratch.fill(phase ? 0xa5 : 0x5a);
            cuda_synchronize();
            if (call.graph)
                graph.launch(context.stream());
            else
                launch();
            context.synchronize();
            failures += verify_output(profile, call, weights, output, activation);
            failures += input_projection::verify_preserved(call.id, input, bits);
            failures += scratch.verify_guards(call.id);
            if (workspace.used() != 0 || workspace.peak_used() > capacity) {
                std::cerr << call.id << ": workspace query/scope mismatch\n";
                ++failures;
            }
        }
    }
    for (const auto& weight : weights) failures += weight->verify_preserved("split weight");
    return failures;
}
} // namespace

void add_profile(Suite& suite, const Profile& profile, std::vector<Invocation> calls) {
    std::string format, recipe;
    int n = 0;
    for (const auto& weight : profile.weights) {
        if (!format.empty()) format += "_";
        format += linear_suite::format_name(weight.type);
        recipe += "s" + std::to_string(weight.seed) + "/";
        n += weight.rows;
    }
    recipe += "segments";
    for (int rows : profile.outputs) recipe += "-" + std::to_string(rows);
    std::vector<linear_suite::Case> cases;
    for (auto& call : calls) {
        std::string input =
            recipe + "/x" + std::to_string(call.input_seed) + "/next" +
            std::to_string(call.changed_seed) + "/rows" +
            (call.full_rows ? std::string("all") : std::to_string(call.sample_rows)) + "/columns" +
            std::to_string(static_cast<int>(call.columns)) +
            (call.convenience ? "/convenience" : "/policy") +
            (call.shared_output ? "/shared" : "/separate");
        if (call.omit_divisors) input += "/no-divisors";
        if (call.validate_operands) input += "/invalid-operands";
        if (call.workspace_intervals) input += "/workspace-intervals";
        auto test = linear_suite::make_case(
            format, n, profile.k, call.t, call.policy,
            call.graph ? linear_suite::Execution::Graph : linear_suite::Execution::Eager, input);
        cases.push_back(std::move(test));
    }
    suite.add(std::move(cases), [profile, calls = std::move(calls)](auto selected, auto declared) {
        std::vector<Invocation> chosen;
        for (auto index : selected) {
            chosen.push_back(calls[index]);
            chosen.back().id = declared[index].id;
        }
        return run(profile, chosen);
    });
}
} // namespace ninfer::test::linear_split
