#include "ninfer/ops/kimi_delta_attention.h"

#include "ops/kda_ref.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kStateDim     = 128;
constexpr float kLowerBound = -5.0F;
constexpr float kScale      = 1.0F / 11.313708498984761F;

constexpr ReductionCriterion output_criterion(bool chunked = false) {
    // Mixed BF16/TF32 MMA with FP32 master state is a distinct arithmetic profile.
    if (chunked) return {6.0e-3, 1.0e-5, 1.5e-2};
    return {/*relative_l2=*/2.2e-3, /*gross_absolute=*/1.0e-5,
            /*gross_relative_to_max_reference=*/4.0e-3};
}

constexpr ReductionCriterion state_criterion(bool chunked = false) {
    if (chunked) return {4.0e-3, 1.0e-5, 1.2e-2};
    return {/*relative_l2=*/1.5e-5, /*gross_absolute=*/5.0e-6,
            /*gross_relative_to_max_reference=*/3.0e-5};
}

struct Case {
    const char* name;
    int heads;
    int tokens;
    bool near_zero_qk     = false;
    bool saturated_gate   = false;
    bool zero_state       = false;
    int qk_heads          = 0;
    float lower_bound     = kLowerBound;
    bool weak_decay       = false;
    bool small_beta       = false;
    bool collinear_cancel = false;

    int qheads() const { return qk_heads == 0 ? heads : qk_heads; }
};

std::string case_label(const Case& c) {
    return std::string(c.name) + " Hq=" + std::to_string(c.qheads()) +
           " Hv=" + std::to_string(c.heads) + " T=" + std::to_string(c.tokens);
}

void fill_uniform(std::vector<float>& values, std::mt19937& generator, float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    for (float& value : values) { value = distribution(generator); }
}

kda_ref::Inputs make_inputs(const Case& test_case, std::uint32_t seed) {
    kda_ref::Inputs in;
    in.value_heads = test_case.heads;
    in.qk_heads    = test_case.qheads();
    in.tokens      = test_case.tokens;

    const std::size_t vector_size =
        static_cast<std::size_t>(kStateDim) * test_case.heads * test_case.tokens;
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim) * kStateDim * test_case.heads;
    in.q.resize(static_cast<std::size_t>(kStateDim) * test_case.qheads() * test_case.tokens);
    in.k.resize(in.q.size());
    in.v.resize(vector_size);
    in.g.resize(vector_size);
    in.beta.resize(static_cast<std::size_t>(test_case.heads * test_case.tokens));
    in.a_log.resize(static_cast<std::size_t>(test_case.heads));
    in.dt_bias.resize(static_cast<std::size_t>(kStateDim * test_case.heads));
    in.state.resize(state_size);

    std::mt19937 generator(seed);
    fill_uniform(in.q, generator, -1.0F, 1.0F);
    fill_uniform(in.k, generator, -1.0F, 1.0F);
    fill_uniform(in.v, generator, -0.5F, 0.5F);
    fill_uniform(in.g, generator, -4.0F, 4.0F);
    fill_uniform(in.beta, generator, -5.0F, 5.0F);
    fill_uniform(in.a_log, generator, -0.5F, 0.5F);
    fill_uniform(in.dt_bias, generator, -2.0F, 2.0F);
    fill_uniform(in.state, generator, -0.02F, 0.02F);

    if (test_case.near_zero_qk) {
        for (float& value : in.q) { value *= 1.0e-4F; }
        for (float& value : in.k) { value *= 1.0e-4F; }
    }
    if (test_case.saturated_gate) {
        for (std::size_t index = 0; index < in.g.size(); ++index) {
            in.g[index] = (index & 1U) == 0U ? -80.0F : 80.0F;
        }
        std::fill(in.a_log.begin(), in.a_log.end(), 0.0F);
        std::fill(in.dt_bias.begin(), in.dt_bias.end(), 0.0F);
    }
    if (test_case.zero_state) { std::fill(in.state.begin(), in.state.end(), 0.0F); }

    if (test_case.weak_decay) {
        std::fill(in.g.begin(), in.g.end(), -8.0F);
        std::fill(in.a_log.begin(), in.a_log.end(), 0.0F);
        std::fill(in.dt_bias.begin(), in.dt_bias.end(), 0.0F);
    }
    if (test_case.small_beta) { std::fill(in.beta.begin(), in.beta.end(), -6.0F); }
    if (test_case.collinear_cancel) {
        for (float& value : in.state) { value *= 50.0F; }
        const std::vector<float> base(in.k.begin(), in.k.begin() + kStateDim * in.qk_heads);
        for (int token = 0; token < in.tokens; ++token) {
            for (std::size_t i = 0; i < base.size(); ++i) {
                const auto offset = token * base.size() + i;
                in.k[offset]      = base[i] + 0.002F * in.k[offset];
            }
        }
        // Construct V near S*K from represented K, stressing cancellation in the residual.
        round_to_bf16(in.k);
        for (int token = 0; token < in.tokens; ++token) {
            for (int head = 0; head < in.value_heads; ++head) {
                const auto kb =
                    (token * in.qk_heads + head / (in.value_heads / in.qk_heads)) * kStateDim;
                double norm = 1.0e-6;
                for (int d = 0; d < kStateDim; ++d) { norm += double(in.k[kb + d]) * in.k[kb + d]; }
                const double inv_norm = 1.0 / std::sqrt(norm);
                for (int value = 0; value < kStateDim; ++value) {
                    double prediction = 0.0;
                    for (int d = 0; d < kStateDim; ++d) {
                        prediction += double(in.state[(head * kStateDim + value) * kStateDim + d]) *
                                      in.k[kb + d] * inv_norm;
                    }
                    in.v[(token * in.value_heads + head) * kStateDim + value] = float(prediction);
                }
            }
        }
        std::fill(in.beta.begin(), in.beta.end(), 4.0F);
    }
    round_to_bf16(in.q);
    round_to_bf16(in.k);
    round_to_bf16(in.v);
    round_to_bf16(in.g);
    round_to_bf16(in.beta);
    return in;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

std::vector<double> doubles(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

struct DeviceInputs {
    explicit DeviceInputs(const kda_ref::Inputs& in)
        : q(to_device_bf16(in.q)), k(to_device_bf16(in.k)), v(to_device_bf16(in.v)),
          g(to_device_bf16(in.g)), beta(to_device_bf16(in.beta)), a_log(to_device_f32(in.a_log)),
          dt_bias(to_device_f32(in.dt_bias)) {}

    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer g;
    DeviceBuffer beta;
    DeviceBuffer a_log;
    DeviceBuffer dt_bias;
};

struct Views {
    Views(const Case& test_case, DeviceInputs& device, void* state_in_data, void* state_out_data,
          void* out_data)
        : q(device.q.p, DType::BF16, {kStateDim, test_case.qheads(), test_case.tokens}),
          k(device.k.p, DType::BF16, {kStateDim, test_case.qheads(), test_case.tokens}),
          v(device.v.p, DType::BF16, {kStateDim, test_case.heads, test_case.tokens}),
          g(device.g.p, DType::BF16, {kStateDim, test_case.heads, test_case.tokens}),
          beta(device.beta.p, DType::BF16, {test_case.heads, test_case.tokens}),
          a_log(device.a_log.p, DType::FP32, {test_case.heads}),
          dt_bias(device.dt_bias.p, DType::FP32, {kStateDim, test_case.heads}),
          state_in(state_in_data, DType::FP32, {kStateDim, kStateDim, test_case.heads}),
          state_out(state_out_data, DType::FP32, {kStateDim, kStateDim, test_case.heads}),
          out(out_data, DType::BF16, {kStateDim, test_case.heads, test_case.tokens}),
          scratch(std::max<std::size_t>(
              256, ops::kimi_delta_attention_workspace_capacity_bytes(
                       test_case.qheads(), test_case.heads, test_case.tokens, test_case.tokens))),
          workspace(DeviceSpan{scratch.data(), scratch.bytes()}) {
        scratch.fill(0xff);
    }

    Tensor q;
    Tensor k;
    Tensor v;
    Tensor g;
    Tensor beta;
    Tensor a_log;
    Tensor dt_bias;
    Tensor state_in;
    Tensor state_out;
    Tensor out;
    GuardedDeviceBuffer scratch;
    WorkspaceArena workspace;
};

DeviceExecutionView execution(cudaStream_t stream = nullptr) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return {stream, properties.multiProcessorCount};
}

int verify_inputs_unchanged(const std::string& label, const kda_ref::Inputs& in,
                            const DeviceInputs& device) {
    int failures = 0;
    failures += verify_exact((label + " q unchanged").c_str(),
                             from_device<std::uint16_t>(device.q, in.q.size()), bf16_bits(in.q));
    failures += verify_exact((label + " k unchanged").c_str(),
                             from_device<std::uint16_t>(device.k, in.k.size()), bf16_bits(in.k));
    failures += verify_exact((label + " v unchanged").c_str(),
                             from_device<std::uint16_t>(device.v, in.v.size()), bf16_bits(in.v));
    failures += verify_exact((label + " g unchanged").c_str(),
                             from_device<std::uint16_t>(device.g, in.g.size()), bf16_bits(in.g));
    failures +=
        verify_exact((label + " beta unchanged").c_str(),
                     from_device<std::uint16_t>(device.beta, in.beta.size()), bf16_bits(in.beta));
    failures += verify_exact((label + " A_log unchanged").c_str(),
                             from_device<float>(device.a_log, in.a_log.size()), in.a_log);
    failures += verify_exact((label + " dt_bias unchanged").c_str(),
                             from_device<float>(device.dt_bias, in.dt_bias.size()), in.dt_bias);
    return failures;
}

int verify_result(const std::string& label, const kda_ref::Inputs& in,
                  const kda_ref::Result& reference, const GuardedDeviceBuffer& state,
                  const GuardedDeviceBuffer& out, bool chunked = false) {
    int failures = 0;
    failures += verify_reduction(label + " out", from_device_bf16(out.data(), in.v.size()),
                                 reference.out, output_criterion(chunked));
    failures += verify_reduction(label + " state",
                                 doubles(from_device<float>(state.data(), in.state.size())),
                                 reference.final_state, state_criterion(chunked));
    failures += state.verify_guards(label + " state");
    failures += out.verify_guards(label + " out");
    return failures;
}

int inplace_case(const Case& test_case, std::uint32_t seed) {
    const kda_ref::Inputs in        = make_inputs(test_case, seed);
    const kda_ref::Result reference = kda_ref::evaluate(
        in, static_cast<double>(test_case.lower_bound), static_cast<double>(kScale));
    DeviceInputs device(in);
    GuardedDeviceBuffer state(in.state.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state.copy_from_host(in.state.data(), state.bytes());
    out.fill(0xff);
    Views views(test_case, device, state.data(), state.data(), out.data());

    ops::kimi_delta_attention(views.q, views.k, views.v, views.g, views.beta, views.a_log,
                              views.dt_bias, test_case.lower_bound, kScale, views.workspace,
                              views.state_out, views.out, execution());
    cuda_synchronize();

    const std::string label = case_label(test_case) + " inplace";
    int failures = verify_result(label, in, reference, state, out, test_case.tokens >= 12);
    failures += views.scratch.verify_guards(label + " workspace");
    failures += verify_inputs_unchanged(label, in, device);
    return failures;
}

int continue_decode(const Case& prefix, const kda_ref::Inputs& prefix_input,
                    const kda_ref::Result& prefix_reference, GuardedDeviceBuffer& state) {
    Case c{"chunked to decode", prefix.heads, 128};
    c.qk_heads   = prefix.qheads();
    c.weak_decay = true;
    auto in      = make_inputs(c, 70123U);
    in.a_log     = prefix_input.a_log;
    in.dt_bias   = prefix_input.dt_bias;
    for (std::size_t i = 0; i < in.state.size(); ++i) {
        in.state[i] = static_cast<float>(prefix_reference.final_state[i]);
    }
    DeviceInputs device(in);
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    out.fill(0xff);
    Case single   = c;
    single.tokens = 1;
    Views views(single, device, state.data(), state.data(), out.data());
    auto step   = in;
    step.tokens = 1;
    kda_ref::Result expected;
    expected.out.reserve(in.v.size());
    const auto ex = execution();
    for (int token = 0; token < c.tokens; ++token) {
        const auto slice = [token](const std::vector<float>& source, int width,
                                   std::vector<float>& destination, const DeviceBuffer& buffer,
                                   Tensor& tensor) {
            const auto offset = static_cast<std::size_t>(token) * width;
            destination.assign(source.begin() + offset, source.begin() + offset + width);
            tensor.data = static_cast<std::uint16_t*>(buffer.p) + offset;
        };
        slice(in.q, kStateDim * c.qheads(), step.q, device.q, views.q);
        slice(in.k, kStateDim * c.qheads(), step.k, device.k, views.k);
        slice(in.v, kStateDim * c.heads, step.v, device.v, views.v);
        slice(in.g, kStateDim * c.heads, step.g, device.g, views.g);
        slice(in.beta, c.heads, step.beta, device.beta, views.beta);
        views.out.data = static_cast<std::uint16_t*>(out.data()) + token * kStateDim * c.heads;
        const auto reference = kda_ref::evaluate(step, prefix.lower_bound, kScale);
        expected.out.insert(expected.out.end(), reference.out.begin(), reference.out.end());
        // Every public one-token call publishes an FP32 state for the next call.
        for (std::size_t i = 0; i < step.state.size(); ++i) {
            step.state[i] = static_cast<float>(reference.final_state[i]);
        }
        expected.final_state = doubles(step.state);
        ops::kimi_delta_attention(views.q, views.k, views.v, views.g, views.beta, views.a_log,
                                  views.dt_bias, prefix.lower_bound, kScale, views.workspace,
                                  views.state_out, views.out, ex);
    }
    cuda_synchronize();
    const auto label = case_label(prefix) + " then 128 one-token calls";
    int failures     = verify_result(label, in, expected, state, out, true);
    failures += views.scratch.verify_guards(label + " workspace");
    return failures;
}

int distinct_case(const Case& test_case, std::uint32_t seed, bool decode = false) {
    const kda_ref::Inputs in        = make_inputs(test_case, seed);
    const kda_ref::Result reference = kda_ref::evaluate(
        in, static_cast<double>(test_case.lower_bound), static_cast<double>(kScale));
    DeviceInputs device(in);
    GuardedDeviceBuffer state_in(in.state.size() * sizeof(float));
    GuardedDeviceBuffer state_out(in.state.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state_in.copy_from_host(in.state.data(), state_in.bytes());
    state_out.fill(0xff);
    out.fill(0xff);
    Views views(test_case, device, state_in.data(), state_out.data(), out.data());

    ops::kimi_delta_attention(views.q, views.k, views.v, views.g, views.beta, views.a_log,
                              views.dt_bias, test_case.lower_bound, kScale, views.workspace,
                              static_cast<const Tensor&>(views.state_in), views.state_out,
                              views.out, execution());
    cuda_synchronize();

    const std::string label = case_label(test_case) + " distinct";
    int failures = verify_result(label, in, reference, state_out, out, test_case.tokens >= 12);
    failures += views.scratch.verify_guards(label + " workspace");
    failures += verify_exact((label + " state-in unchanged").c_str(),
                             from_device<float>(state_in.data(), in.state.size()), in.state);
    failures += state_in.verify_guards(label + " state-in");
    failures += verify_inputs_unchanged(label, in, device);
    if (decode) { failures += continue_decode(test_case, in, reference, state_out); }
    return failures;
}

int distinct_exact_alias_case(const Case& test_case, std::uint32_t seed) {
    const kda_ref::Inputs in        = make_inputs(test_case, seed);
    const kda_ref::Result reference = kda_ref::evaluate(
        in, static_cast<double>(test_case.lower_bound), static_cast<double>(kScale));
    DeviceInputs device(in);
    GuardedDeviceBuffer state(in.state.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state.copy_from_host(in.state.data(), state.bytes());
    out.fill(0xff);
    Views views(test_case, device, state.data(), state.data(), out.data());

    ops::kimi_delta_attention(views.q, views.k, views.v, views.g, views.beta, views.a_log,
                              views.dt_bias, test_case.lower_bound, kScale, views.workspace,
                              static_cast<const Tensor&>(views.state_in), views.state_out,
                              views.out, execution());
    cuda_synchronize();

    const std::string label = case_label(test_case) + " distinct exact-alias";
    int failures = verify_result(label, in, reference, state, out, test_case.tokens >= 12);
    failures += views.scratch.verify_guards(label + " workspace");
    failures += verify_inputs_unchanged(label, in, device);
    return failures;
}

int batch_update_case(const Case& test_case, const std::vector<int>& state_slots, int slots,
                      std::uint32_t seed) {
    if (test_case.tokens != 1) { throw std::logic_error("batch_update_case requires W=1"); }
    const int batch                   = static_cast<int>(state_slots.size());
    const std::size_t vector_row_size = static_cast<std::size_t>(kStateDim) * test_case.heads;
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim) * kStateDim * test_case.heads;

    const kda_ref::Inputs shared = make_inputs(test_case, seed ^ 0x51a7U);
    kda_ref::Inputs aggregate;
    aggregate.value_heads = test_case.heads;
    aggregate.qk_heads    = test_case.qheads();
    aggregate.tokens      = batch;
    aggregate.a_log       = shared.a_log;
    aggregate.dt_bias     = shared.dt_bias;
    aggregate.q.reserve(vector_row_size * static_cast<std::size_t>(batch));
    aggregate.k.reserve(vector_row_size * static_cast<std::size_t>(batch));
    aggregate.v.reserve(vector_row_size * static_cast<std::size_t>(batch));
    aggregate.g.reserve(vector_row_size * static_cast<std::size_t>(batch));
    aggregate.beta.reserve(static_cast<std::size_t>(test_case.heads * batch));

    std::vector<float> initial_states(state_size * static_cast<std::size_t>(slots));
    std::mt19937 state_generator(seed ^ 0xa3c59ac3U);
    fill_uniform(initial_states, state_generator, -0.02F, 0.02F);

    std::vector<kda_ref::Result> references;
    references.reserve(static_cast<std::size_t>(batch));
    std::vector<double> expected_output(vector_row_size * static_cast<std::size_t>(batch));
    std::vector<bool> written_slots(static_cast<std::size_t>(slots), false);
    for (int row = 0; row < batch; ++row) {
        const int slot = state_slots[static_cast<std::size_t>(row)];
        if (slot < 0 || slot >= slots || written_slots[static_cast<std::size_t>(slot)]) {
            throw std::logic_error("batch_update_case requires valid distinct slots");
        }

        kda_ref::Inputs input =
            make_inputs(test_case, seed + static_cast<std::uint32_t>(row) * 97U);
        input.a_log   = aggregate.a_log;
        input.dt_bias = aggregate.dt_bias;
        aggregate.q.insert(aggregate.q.end(), input.q.begin(), input.q.end());
        aggregate.k.insert(aggregate.k.end(), input.k.begin(), input.k.end());
        aggregate.v.insert(aggregate.v.end(), input.v.begin(), input.v.end());
        aggregate.g.insert(aggregate.g.end(), input.g.begin(), input.g.end());
        aggregate.beta.insert(aggregate.beta.end(), input.beta.begin(), input.beta.end());
        std::copy(input.state.begin(), input.state.end(),
                  initial_states.begin() + static_cast<std::size_t>(slot) * state_size);

        kda_ref::Result reference = kda_ref::evaluate(
            input, static_cast<double>(test_case.lower_bound), static_cast<double>(kScale));
        std::copy(reference.out.begin(), reference.out.end(),
                  expected_output.begin() + static_cast<std::size_t>(row) * vector_row_size);
        references.push_back(std::move(reference));
        written_slots[static_cast<std::size_t>(slot)] = true;
    }

    DeviceInputs device(aggregate);
    DeviceBuffer device_state_slots = to_device_i32(state_slots);
    GuardedDeviceBuffer states(initial_states.size() * sizeof(float));
    GuardedDeviceBuffer out(aggregate.v.size() * sizeof(std::uint16_t));
    states.copy_from_host(initial_states.data(), states.bytes());
    out.fill(0xff);

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qheads(), 1, batch});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qheads(), 1, batch});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.heads, 1, batch});
    Tensor g(device.g.p, DType::BF16, {kStateDim, test_case.heads, 1, batch});
    Tensor beta(device.beta.p, DType::BF16, {test_case.heads, 1, batch});
    Tensor a_log(device.a_log.p, DType::FP32, {test_case.heads});
    Tensor dt_bias(device.dt_bias.p, DType::FP32, {kStateDim, test_case.heads});
    Tensor states_tensor(states.data(), DType::FP32,
                         {kStateDim, kStateDim, test_case.heads, slots});
    Tensor state_slots_tensor(device_state_slots.p, DType::I32, {batch});
    Tensor out_tensor(out.data(), DType::BF16, {kStateDim, test_case.heads, 1, batch});

    ops::kimi_delta_attention_batch_update(q, k, v, g, beta, a_log, dt_bias, kLowerBound, kScale,
                                           states_tensor, state_slots_tensor, out_tensor, nullptr);
    cuda_synchronize();

    const std::string label =
        std::string(test_case.name) + " batch update B=" + std::to_string(batch);
    int failures =
        verify_reduction(label + " out", from_device_bf16(out.data(), aggregate.v.size()),
                         expected_output, output_criterion());
    const std::vector<float> got_states = from_device<float>(states.data(), initial_states.size());
    for (int row = 0; row < batch; ++row) {
        const int slot          = state_slots[static_cast<std::size_t>(row)];
        const std::size_t begin = static_cast<std::size_t>(slot) * state_size;
        failures += verify_reduction(
            label + " row " + std::to_string(row) + " state",
            doubles(std::vector<float>(got_states.begin() + begin,
                                       got_states.begin() + begin + state_size)),
            references[static_cast<std::size_t>(row)].final_state, state_criterion());
    }
    for (int slot = 0; slot < slots; ++slot) {
        if (written_slots[static_cast<std::size_t>(slot)]) continue;
        const std::size_t begin      = static_cast<std::size_t>(slot) * state_size;
        const std::string slot_label = label + " untouched slot " + std::to_string(slot);
        failures += verify_exact(
            slot_label.c_str(),
            std::vector<float>(got_states.begin() + begin, got_states.begin() + begin + state_size),
            std::vector<float>(initial_states.begin() + begin,
                               initial_states.begin() + begin + state_size));
    }
    const std::string selector_label = label + " state selectors unchanged";
    failures += verify_exact(selector_label.c_str(),
                             from_device_i32(device_state_slots, state_slots.size()), state_slots);
    failures += states.verify_guards(label + " states");
    failures += out.verify_guards(label + " out");
    failures += verify_inputs_unchanged(label, aggregate, device);
    return failures;
}

int graph_replay_case() {
    Case c{"chunked graph input updates", 48, 65};
    c.qk_heads            = 16;
    kda_ref::Inputs input = make_inputs(c, 45678U);
    DeviceInputs device(input);
    GuardedDeviceBuffer state_in(input.state.size() * sizeof(float));
    GuardedDeviceBuffer state_out(input.state.size() * sizeof(float));
    GuardedDeviceBuffer out(input.v.size() * sizeof(std::uint16_t));
    state_in.copy_from_host(input.state.data(), state_in.bytes());
    Views views(c, device, state_in.data(), state_out.data(), out.data());
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(
        cudaStreamSynchronize(nullptr)); // Drain fixture initialization on the default stream.
    const auto ex = execution(stream);
    auto launch   = [&] {
        ops::kimi_delta_attention(views.q, views.k, views.v, views.g, views.beta, views.a_log,
                                    views.dt_bias, c.lower_bound, kScale, views.workspace,
                                    views.state_in, views.state_out, views.out, ex);
    };
    launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    launch();
    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    cudaGraphExec_t executable = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
    int failures = 0;
    for (unsigned replay = 0; replay < 2; ++replay) {
        input                  = make_inputs(c, 56789U + replay);
        const auto upload_bf16 = [](DeviceBuffer& dst, const std::vector<float>& values) {
            const auto bits = bf16_bits(values);
            dst.copy_from_host(bits.data(), dst.bytes);
        };
        upload_bf16(device.q, input.q);
        upload_bf16(device.k, input.k);
        upload_bf16(device.v, input.v);
        upload_bf16(device.g, input.g);
        upload_bf16(device.beta, input.beta);
        device.a_log.copy_from_host(input.a_log.data(), device.a_log.bytes);
        device.dt_bias.copy_from_host(input.dt_bias.data(), device.dt_bias.bytes);
        state_in.copy_from_host(input.state.data(), state_in.bytes());
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        CUDA_CHECK(cudaMemsetAsync(out.data(), 0xff, out.bytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(state_out.data(), 0xff, state_out.bytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(views.scratch.data(), 0xff, views.scratch.bytes(), stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto ref = kda_ref::evaluate(input, c.lower_bound, kScale);
        failures += verify_result(std::string(c.name) + " " + std::to_string(replay), input, ref,
                                  state_out, out, true);
        failures += verify_inputs_unchanged(c.name, input, device);
        failures +=
            verify_exact("graph state input unchanged",
                         from_device<float>(state_in.data(), input.state.size()), input.state);
        failures += state_in.verify_guards(c.name);
        failures += views.scratch.verify_guards("graph workspace");
    }
    CUDA_CHECK(cudaGraphExecDestroy(executable));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return failures;
}

int batch_contract_rejection_cases() {
    constexpr int heads          = 2;
    constexpr int batch_capacity = 9;
    DeviceBuffer vectors(static_cast<std::size_t>(kStateDim) * heads * batch_capacity *
                         sizeof(std::uint16_t));
    DeviceBuffer beta_buffer(static_cast<std::size_t>(heads) * batch_capacity *
                             sizeof(std::uint16_t));
    DeviceBuffer a_log_buffer(static_cast<std::size_t>(heads) * sizeof(float));
    DeviceBuffer dt_buffer(static_cast<std::size_t>(kStateDim) * heads * sizeof(float));
    DeviceBuffer state_buffer(static_cast<std::size_t>(kStateDim) * kStateDim * heads * 2 *
                              sizeof(float));
    DeviceBuffer slot_buffer(static_cast<std::size_t>(batch_capacity) * sizeof(std::int32_t));

    const auto rejects = [&](int tokens, int batch, DType slot_dtype) {
        Tensor q(vectors.p, DType::BF16, {kStateDim, heads, tokens, batch});
        Tensor k(vectors.p, DType::BF16, {kStateDim, heads, tokens, batch});
        Tensor v(vectors.p, DType::BF16, {kStateDim, heads, tokens, batch});
        Tensor g(vectors.p, DType::BF16, {kStateDim, heads, tokens, batch});
        Tensor beta(beta_buffer.p, DType::BF16, {heads, tokens, batch});
        Tensor a_log(a_log_buffer.p, DType::FP32, {heads});
        Tensor dt_bias(dt_buffer.p, DType::FP32, {kStateDim, heads});
        Tensor states(state_buffer.p, DType::FP32, {kStateDim, kStateDim, heads, 2});
        Tensor state_slots(slot_buffer.p, slot_dtype, {batch});
        Tensor out(vectors.p, DType::BF16, {kStateDim, heads, tokens, batch});
        try {
            ops::kimi_delta_attention_batch_update(q, k, v, g, beta, a_log, dt_bias, kLowerBound,
                                                   kScale, states, state_slots, out, nullptr);
        } catch (const std::invalid_argument&) { return true; }
        cuda_synchronize();
        return false;
    };

    int failures = 0;
    if (!rejects(1, 9, DType::I32)) {
        std::cerr << "kimi_delta_attention batch update accepted B=9\n";
        ++failures;
    }
    if (!rejects(2, 2, DType::I32)) {
        std::cerr << "kimi_delta_attention batch update accepted W=2\n";
        ++failures;
    }
    if (!rejects(1, 2, DType::FP32)) {
        std::cerr << "kimi_delta_attention batch update accepted FP32 state selectors\n";
        ++failures;
    }
    return failures;
}

int contract_rejection_cases() {
    constexpr int heads = 2;
    DeviceBuffer vectors(static_cast<std::size_t>(kStateDim) * heads * sizeof(std::uint16_t));
    DeviceBuffer beta_buffer(static_cast<std::size_t>(heads) * sizeof(std::uint16_t));
    DeviceBuffer a_log_buffer(static_cast<std::size_t>(heads) * sizeof(float));
    DeviceBuffer dt_buffer(static_cast<std::size_t>(kStateDim) * heads * sizeof(float));
    DeviceBuffer state_buffer(static_cast<std::size_t>(kStateDim) * kStateDim * heads *
                              sizeof(float));

    Tensor q(vectors.p, DType::BF16, {kStateDim, heads, 1});
    Tensor k(vectors.p, DType::BF16, {kStateDim, heads, 1});
    Tensor v(vectors.p, DType::BF16, {kStateDim, heads, 1});
    Tensor g(vectors.p, DType::BF16, {kStateDim, heads, 1});
    Tensor beta(beta_buffer.p, DType::BF16, {heads, 1});
    Tensor a_log(a_log_buffer.p, DType::FP32, {heads});
    Tensor dt_bias(dt_buffer.p, DType::FP32, {kStateDim, heads});
    Tensor state(state_buffer.p, DType::FP32, {kStateDim, kStateDim, heads});
    Tensor out(vectors.p, DType::BF16, {kStateDim, heads, 1});

    WorkspaceArena workspace(256);
    const auto rejects = [&](const Tensor& test_g, float lower_bound, float scale) {
        try {
            ops::kimi_delta_attention(q, k, v, test_g, beta, a_log, dt_bias, lower_bound, scale,
                                      workspace, state, out, execution());
        } catch (const std::invalid_argument&) { return true; }
        cuda_synchronize();
        return false;
    };

    int failures = 0;
    Tensor wrong_g(vectors.p, DType::FP32, {kStateDim, heads, 1});
    if (!rejects(wrong_g, kLowerBound, kScale)) {
        std::cerr << "kimi_delta_attention accepted FP32 raw gate\n";
        ++failures;
    }
    if (!rejects(g, -5.01F, kScale) || !rejects(g, 0.01F, kScale)) {
        std::cerr << "kimi_delta_attention accepted lower_bound outside [-5,0]\n";
        ++failures;
    }
    if (!rejects(g, kLowerBound, 1.0F)) {
        std::cerr << "kimi_delta_attention accepted an invalid scale\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = contract_rejection_cases();
    failures += graph_replay_case();
    failures += batch_contract_rejection_cases();
    failures += inplace_case({"Kimi Linear ordinary", 32, 7}, 32007U);
    failures += distinct_case({"Kimi Linear ordinary", 32, 7}, 32107U);
    failures += inplace_case({"Kimi K3 ordinary", 96, 3}, 96003U);
    failures += distinct_case({"Kimi K3 zero-state", 96, 3, false, false, true}, 96103U);
    failures += inplace_case({"runtime head count", 5, 2}, 502U);
    failures += inplace_case({"near-zero normalized QK", 32, 1, true}, 32201U);
    failures += distinct_case({"saturated safe gate", 32, 2, false, true}, 32202U);
    failures += distinct_exact_alias_case({"state alias contract", 32, 2}, 32302U);
    failures += batch_update_case({"Kimi Linear ordinary", 32, 1}, {4, 0, 5, 2}, 6, 32401U);
    failures +=
        batch_update_case({"Kimi K3 ordinary", 96, 1}, {10, 3, 7, 0, 9, 5, 1, 8}, 11, 96801U);
    failures += batch_update_case({"runtime head count", 5, 1}, {3, 0, 4}, 5, 50301U);

    for (const auto [qheads, vheads] :
         {std::pair{16, 48}, {16, 32}, {96, 96}, {64, 64}, {1, 32}, {5, 15}}) {
        Case c{"grouped recurrent", vheads, 7};
        c.qk_heads = qheads;
        failures += distinct_case(c, 31415U + vheads);
        c.tokens = 1;
        failures += batch_update_case(c, {5, 1, 3}, 7, 27182U + vheads);
        c.tokens = 129;
        c.name   = "chunked grouped";
        failures += distinct_case(c, 16180U + vheads);
        failures += inplace_case(c, 14142U + vheads);
        if (qheads > 1 && vheads >= 32) {
            c.tokens = 1025;
            failures += distinct_case(c, 53212U + vheads);
        }
    }
    for (int t : {11, 12, 13, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1025}) {
        Case c{"chunk boundary", 3, t};
        c.qk_heads = 1;
        failures += distinct_case(c, 90000U + t);
    }
    Case weak{"weak decay long", 3, 16385};
    weak.qk_heads   = 1;
    weak.weak_decay = true;
    failures += distinct_case(weak, 99887U, true);
    weak.lower_bound = 0.0F;
    weak.name        = "zero decay long";
    failures += distinct_case(weak, 88776U, true);
    weak.name       = "small beta zero decay long";
    weak.small_beta = true;
    failures += distinct_case(weak, 332213U, true);
    Case cancellation{"near collinear residual cancellation", 3, 4097};
    cancellation.qk_heads         = 1;
    cancellation.lower_bound      = 0.0F;
    cancellation.collinear_cancel = true;
    failures += distinct_case(cancellation, 99891U);
    failures += distinct_case({"chunk saturated", 5, 257, false, true}, 77665U);
    failures += distinct_case({"chunk near zero QK", 5, 65, true}, 66554U);
    failures += distinct_exact_alias_case({"chunk state alias", 32, 65}, 55443U);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " kimi_delta_attention correctness\n";
    return failures == 0 ? 0 : 1;
}
