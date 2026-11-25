#include "ops/suite.h"
#include "ops/ragged_fixture.h"
#include "ops/state_fixture.h"
#include "ninfer/ops/kimi_delta_attention.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/platform.h"

#include "ops/linear_attention/kda_ref.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
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

// One criterion per public result, independent of private recurrence selection.
constexpr ReductionCriterion output_criterion() { return {6.0e-3, 1.0e-5, 1.5e-2}; }

constexpr ReductionCriterion state_criterion(DType state_dtype = DType::FP32) {
    return state_dtype == DType::BF16 ? ReductionCriterion{7.5e-3, 2.0e-5, 1.6e-2}
                                      : ReductionCriterion{4.0e-3, 1.0e-5, 1.2e-2};
}

std::size_t state_element_bytes(DType dtype) {
    return dtype == DType::BF16 ? sizeof(std::uint16_t) : sizeof(float);
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
    int capacity_tokens   = 0;

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

void upload_state(void* dst, const std::vector<float>& values, DType dtype) {
    if (dtype == DType::BF16) {
        const auto bits = bf16_bits(values);
        CUDA_CHECK(cudaMemcpy(dst, bits.data(), bits.size() * sizeof(std::uint16_t),
                              cudaMemcpyHostToDevice));
        return;
    }
    CUDA_CHECK(cudaMemcpy(dst, values.data(), values.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
}

std::vector<double> read_state(const void* src, std::size_t count, DType dtype) {
    return dtype == DType::BF16 ? from_device_bf16(src, count)
                                : doubles(from_device<float>(src, count));
}

int verify_state_exact(const char* label, const void* device, const std::vector<float>& expected,
                       DType dtype) {
    if (dtype == DType::BF16) {
        return verify_exact(label, from_device<std::uint16_t>(device, expected.size()),
                            bf16_bits(expected));
    }
    return verify_exact(label, from_device<float>(device, expected.size()), expected);
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
    Views(const Case& test_case, DeviceInputs& device, MemorySpan state_in_data,
          MemorySpan state_out_data, MemorySpan out_data, DType state_dtype = DType::FP32)
        : q(device.q.span(), DType::BF16, {kStateDim, test_case.qheads(), test_case.tokens}),
          k(device.k.span(), DType::BF16, {kStateDim, test_case.qheads(), test_case.tokens}),
          v(device.v.span(), DType::BF16, {kStateDim, test_case.heads, test_case.tokens}),
          g(device.g.span(), DType::BF16, {kStateDim, test_case.heads, test_case.tokens}),
          beta(device.beta.span(), DType::BF16, {test_case.heads, test_case.tokens}),
          a_log(device.a_log.span(), DType::FP32, {test_case.heads}),
          dt_bias(device.dt_bias.span(), DType::FP32, {kStateDim, test_case.heads}),
          state_in(state_in_data, state_dtype, {kStateDim, kStateDim, test_case.heads}),
          state_out(state_out_data, state_dtype, {kStateDim, kStateDim, test_case.heads}),
          out(out_data, DType::BF16, {kStateDim, test_case.heads, test_case.tokens}),
          scratch(std::max<std::size_t>(
              256, ops::kimi_delta_attention_workspace_capacity_bytes(
                       test_case.qheads(), test_case.heads,
                       test_case.capacity_tokens ? 1 : test_case.tokens,
                       test_case.capacity_tokens ? test_case.capacity_tokens : test_case.tokens))),
          workspace(scratch.span()) {
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
    MemoryArena workspace;
};

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
                  const GuardedDeviceBuffer& out, DType state_dtype = DType::FP32) {
    int failures = 0;
    failures += verify_reduction(label + " out", from_device_bf16(out.data(), in.v.size()),
                                 reference.out, output_criterion());
    failures += verify_reduction(label + " state",
                                 read_state(state.data(), in.state.size(), state_dtype),
                                 reference.final_state, state_criterion(state_dtype));
    failures += state.verify_guards(label + " state");
    failures += out.verify_guards(label + " out");
    return failures;
}

int inplace_case(const Case& test_case, std::uint32_t seed, DType state_dtype = DType::FP32) {
    kda_ref::Inputs in = make_inputs(test_case, seed);
    if (state_dtype == DType::BF16) { round_to_bf16(in.state); }
    const kda_ref::Result reference = kda_ref::evaluate(
        in, static_cast<double>(test_case.lower_bound), static_cast<double>(kScale));
    DeviceInputs device(in);
    GuardedDeviceBuffer state(in.state.size() * state_element_bytes(state_dtype));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    upload_state(state.data(), in.state, state_dtype);
    out.fill(0xff);
    Views views(test_case, device, state.span(), state.span(), out.span(), state_dtype);

    ops::kimi_delta_attention(views.q, views.k, views.v, views.g, views.beta, views.a_log,
                              views.dt_bias, test_case.lower_bound, kScale, views.workspace,
                              views.state_out, views.out, cuda_environment().default_execution);
    cuda_synchronize();

    const std::string label = case_label(test_case) + " inplace";
    int failures            = verify_result(label, in, reference, state, out, state_dtype);
    failures += views.scratch.verify_guards(label + " workspace");
    failures += verify_inputs_unchanged(label, in, device);
    return failures;
}

int continue_decode(const Case& prefix, const kda_ref::Inputs& prefix_input,
                    const kda_ref::Result& prefix_reference, GuardedDeviceBuffer& state,
                    DType state_dtype = DType::FP32) {
    Case c{"chunked to decode", prefix.heads, 128};
    c.qk_heads   = prefix.qheads();
    c.weak_decay = true;
    auto in      = make_inputs(c, 70123U);
    in.a_log     = prefix_input.a_log;
    in.dt_bias   = prefix_input.dt_bias;
    for (std::size_t i = 0; i < in.state.size(); ++i) {
        in.state[i] = static_cast<float>(prefix_reference.final_state[i]);
    }
    if (state_dtype == DType::BF16) { round_to_bf16(in.state); }
    DeviceInputs device(in);
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    out.fill(0xff);
    Case single   = c;
    single.tokens = 1;
    Views views(single, device, state.span(), state.span(), out.span(), state_dtype);
    auto step   = in;
    step.tokens = 1;
    kda_ref::Result expected;
    expected.out.reserve(in.v.size());
    const auto ex = cuda_environment().default_execution;
    for (int token = 0; token < c.tokens; ++token) {
        const auto slice = [token](const std::vector<float>& source, int width,
                                   std::vector<float>& destination, DeviceBuffer& buffer,
                                   Tensor& tensor) {
            const auto offset = static_cast<std::size_t>(token) * width;
            destination.assign(source.begin() + offset, source.begin() + offset + width);
            tensor.data = static_cast<std::uint16_t*>(buffer.data()) + offset;
        };
        slice(in.q, kStateDim * c.qheads(), step.q, device.q, views.q);
        slice(in.k, kStateDim * c.qheads(), step.k, device.k, views.k);
        slice(in.v, kStateDim * c.heads, step.v, device.v, views.v);
        slice(in.g, kStateDim * c.heads, step.g, device.g, views.g);
        slice(in.beta, c.heads, step.beta, device.beta, views.beta);
        views.out.data = static_cast<std::uint16_t*>(out.data()) + token * kStateDim * c.heads;
        const auto reference = kda_ref::evaluate(step, prefix.lower_bound, kScale);
        expected.out.insert(expected.out.end(), reference.out.begin(), reference.out.end());
        // Every public one-token call publishes a represented state for the next call.
        for (std::size_t i = 0; i < step.state.size(); ++i) {
            step.state[i] = static_cast<float>(reference.final_state[i]);
        }
        if (state_dtype == DType::BF16) { round_to_bf16(step.state); }
        expected.final_state = doubles(step.state);
        ops::kimi_delta_attention(views.q, views.k, views.v, views.g, views.beta, views.a_log,
                                  views.dt_bias, prefix.lower_bound, kScale, views.workspace,
                                  views.state_out, views.out, ex);
    }
    cuda_synchronize();
    const auto label = case_label(prefix) + " then 128 one-token calls";
    int failures     = verify_result(label, in, expected, state, out, state_dtype);
    failures += views.scratch.verify_guards(label + " workspace");
    return failures;
}

int distinct_case(const Case& test_case, std::uint32_t seed, bool decode = false,
                  DType state_dtype = DType::FP32) {
    kda_ref::Inputs in = make_inputs(test_case, seed);
    if (state_dtype == DType::BF16) { round_to_bf16(in.state); }
    const kda_ref::Result reference = kda_ref::evaluate(
        in, static_cast<double>(test_case.lower_bound), static_cast<double>(kScale));
    DeviceInputs device(in);
    GuardedDeviceBuffer state_in(in.state.size() * state_element_bytes(state_dtype));
    GuardedDeviceBuffer state_out(in.state.size() * state_element_bytes(state_dtype));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    upload_state(state_in.data(), in.state, state_dtype);
    state_out.fill(0xff);
    out.fill(0xff);
    Views views(test_case, device, state_in.span(), state_out.span(), out.span(), state_dtype);

    ops::kimi_delta_attention(views.q, views.k, views.v, views.g, views.beta, views.a_log,
                              views.dt_bias, test_case.lower_bound, kScale, views.workspace,
                              static_cast<const Tensor&>(views.state_in), views.state_out,
                              views.out, cuda_environment().default_execution);
    cuda_synchronize();

    const std::string label = case_label(test_case) + " distinct";
    int failures            = verify_result(label, in, reference, state_out, out, state_dtype);
    failures += views.scratch.verify_guards(label + " workspace");
    failures += verify_state_exact((label + " state-in unchanged").c_str(), state_in.data(),
                                   in.state, state_dtype);
    failures += state_in.verify_guards(label + " state-in");
    failures += verify_inputs_unchanged(label, in, device);
    if (decode) { failures += continue_decode(test_case, in, reference, state_out, state_dtype); }
    return failures;
}

int distinct_exact_alias_case(const Case& test_case, std::uint32_t seed,
                              DType state_dtype = DType::FP32) {
    kda_ref::Inputs in = make_inputs(test_case, seed);
    if (state_dtype == DType::BF16) { round_to_bf16(in.state); }
    const kda_ref::Result reference = kda_ref::evaluate(
        in, static_cast<double>(test_case.lower_bound), static_cast<double>(kScale));
    DeviceInputs device(in);
    GuardedDeviceBuffer state(in.state.size() * state_element_bytes(state_dtype));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    upload_state(state.data(), in.state, state_dtype);
    out.fill(0xff);
    Views views(test_case, device, state.span(), state.span(), out.span(), state_dtype);

    ops::kimi_delta_attention(views.q, views.k, views.v, views.g, views.beta, views.a_log,
                              views.dt_bias, test_case.lower_bound, kScale, views.workspace,
                              static_cast<const Tensor&>(views.state_in), views.state_out,
                              views.out, cuda_environment().default_execution);
    cuda_synchronize();

    const std::string label = case_label(test_case) + " distinct exact-alias";
    int failures            = verify_result(label, in, reference, state, out, state_dtype);
    failures += views.scratch.verify_guards(label + " workspace");
    failures += verify_inputs_unchanged(label, in, device);
    return failures;
}

int batch_update_case(const Case& test_case, const std::vector<int>& state_slots, int slots,
                      std::uint32_t seed, DType state_dtype = DType::FP32) {
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
    if (state_dtype == DType::BF16) { round_to_bf16(initial_states); }

    std::vector<kda_ref::Result> references;
    references.reserve(static_cast<std::size_t>(batch));
    std::vector<double> expected_output(vector_row_size * static_cast<std::size_t>(batch));
    std::vector<bool> written_slots(static_cast<std::size_t>(slots), false);
    std::vector<int> valid(batch);
    for (int row = 0; row < batch; ++row) {
        const int slot = state_slots[static_cast<std::size_t>(row)];
        if (slot < -1 || slot >= slots ||
            (slot >= 0 && written_slots[static_cast<std::size_t>(slot)])) {
            throw std::logic_error("batch_update_case requires valid distinct slots");
        }

        kda_ref::Inputs input =
            make_inputs(test_case, seed + static_cast<std::uint32_t>(row) * 97U);
        if (state_dtype == DType::BF16) { round_to_bf16(input.state); }
        input.a_log   = aggregate.a_log;
        input.dt_bias = aggregate.dt_bias;
        aggregate.q.insert(aggregate.q.end(), input.q.begin(), input.q.end());
        aggregate.k.insert(aggregate.k.end(), input.k.begin(), input.k.end());
        aggregate.v.insert(aggregate.v.end(), input.v.begin(), input.v.end());
        aggregate.g.insert(aggregate.g.end(), input.g.begin(), input.g.end());
        aggregate.beta.insert(aggregate.beta.end(), input.beta.begin(), input.beta.end());
        if (slot == -1) {
            references.emplace_back();
            continue;
        }
        valid[row] = 1;
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
    GuardedDeviceBuffer states(initial_states.size() * state_element_bytes(state_dtype));
    GuardedDeviceBuffer out(aggregate.v.size() * sizeof(std::uint16_t));
    upload_state(states.data(), initial_states, state_dtype);
    out.fill(0xff);

    Tensor q(device.q.span(), DType::BF16, {kStateDim, test_case.qheads(), 1, batch});
    Tensor k(device.k.span(), DType::BF16, {kStateDim, test_case.qheads(), 1, batch});
    Tensor v(device.v.span(), DType::BF16, {kStateDim, test_case.heads, 1, batch});
    Tensor g(device.g.span(), DType::BF16, {kStateDim, test_case.heads, 1, batch});
    Tensor beta(device.beta.span(), DType::BF16, {test_case.heads, 1, batch});
    Tensor a_log(device.a_log.span(), DType::FP32, {test_case.heads});
    Tensor dt_bias(device.dt_bias.span(), DType::FP32, {kStateDim, test_case.heads});
    Tensor states_tensor(states.span(), state_dtype,
                         {kStateDim, kStateDim, test_case.heads, slots});
    RecordStateFixture history(states_tensor, 3);
    std::vector<std::int64_t> state_offsets(batch, -1);
    for (int row = 0; row < batch; ++row)
        if (valid[row]) state_offsets[row] = history.byte_offset(state_slots[row]);
    DeviceBuffer device_state_slots = to_device(state_offsets);
    Tensor state_slots_tensor(device_state_slots.span(), DType::I64, {batch});
    Tensor out_tensor(out.span(), DType::BF16, {kStateDim, test_case.heads, 1, batch});

    // Independent arenas and selectors distinguish source reads from destination writes.
    RecordStateFixture source_history(states_tensor, 3);
    auto destination_slots = state_slots;
    std::vector<int> active_slots;
    for (const auto slot : state_slots)
        if (slot >= 0) active_slots.push_back(slot);
    auto next = active_slots.rbegin();
    std::vector<std::int64_t> destination_offsets(batch, -1);
    for (int row = 0; row < batch; ++row)
        if (valid[row]) {
            destination_slots[row]   = *next++;
            destination_offsets[row] = history.byte_offset(destination_slots[row]);
        }
    auto destination_buffer = to_device(destination_offsets);
    Tensor destination_table(destination_buffer.span(), DType::I64, {batch});
    for (int slot : destination_slots) {
        if (slot < 0) continue;
        const auto destination = history.tensor(slot);
        CUDA_CHECK(cudaMemset(destination.data, 0xff, destination.bytes()));
    }
    auto valid_buffer        = to_device(valid);
    const bool masked        = std::find(valid.begin(), valid.end(), 0) != valid.end();
    const Tensor valid_table = masked ? Tensor(valid_buffer.span(), DType::I32, {batch}) : Tensor{};
    const auto execution     = cuda_environment().context.execution_view();
    const auto launch        = [&] {
        ops::kimi_delta_attention_batch_update(q, k, v, g, beta, a_log, dt_bias, kLowerBound,
                                                      kScale, source_history.view(state_slots_tensor),
                                                      history.view(destination_table), valid_table,
                                                      out_tensor, execution);
    };
    cuda_synchronize();
    launch();
    cuda_synchronize();
    if (masked) {
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        CUDA_CHECK(cudaStreamBeginCapture(execution.stream(), cudaStreamCaptureModeThreadLocal));
        launch();
        CUDA_CHECK(cudaStreamEndCapture(execution.stream(), &graph));
        CUDA_CHECK(cudaGraphInstantiate(&executable, graph, 0));
        const std::vector<int> inactive(batch, 0);
        const std::vector<std::int64_t> invalid(batch, -1);
        const auto storage = history.view().records.storage;
        const auto before  = from_device<std::uint8_t>(storage.data, storage.bytes);
        valid_buffer.copy_from_host(inactive.data(), valid_buffer.size());
        device_state_slots.copy_from_host(invalid.data(), device_state_slots.size());
        destination_buffer.copy_from_host(invalid.data(), destination_buffer.size());
        CUDA_CHECK(cudaGraphLaunch(executable, execution.stream()));
        cuda_synchronize();
        if (from_device<std::uint8_t>(storage.data, storage.bytes) != before ||
            from_device<std::uint16_t>(out.data(), aggregate.v.size()) !=
                std::vector<std::uint16_t>(aggregate.v.size(), 0))
            throw std::runtime_error("inactive KDA update accessed State or failed to zero output");
        valid_buffer.copy_from_host(valid.data(), valid_buffer.size());
        device_state_slots.copy_from_host(state_offsets.data(), device_state_slots.size());
        destination_buffer.copy_from_host(destination_offsets.data(), destination_buffer.size());
        CUDA_CHECK(cudaGraphLaunch(executable, execution.stream()));
        cuda_synchronize();
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(graph));
    }

    const std::string label =
        std::string(test_case.name) + " batch update B=" + std::to_string(batch);
    int failures =
        verify_reduction(label + " out", from_device_bf16(out.data(), aggregate.v.size()),
                         expected_output, output_criterion());
    source_history.copy_back();
    failures += verify_state_exact("batch source remains unchanged", states.data(), initial_states,
                                   state_dtype);
    history.copy_back();
    failures += history.verify_guards("pooled state");
    const std::vector<double> got_states =
        read_state(states.data(), initial_states.size(), state_dtype);
    for (int row = 0; row < batch; ++row) {
        const int slot = destination_slots[static_cast<std::size_t>(row)];
        if (slot < 0) continue;
        const std::size_t begin = static_cast<std::size_t>(slot) * state_size;
        failures += verify_reduction(
            label + " row " + std::to_string(row) + " state",
            std::vector<double>(got_states.begin() + begin,
                                got_states.begin() + begin + state_size),
            references[static_cast<std::size_t>(row)].final_state, state_criterion(state_dtype));
    }
    for (int slot = 0; slot < slots; ++slot) {
        if (written_slots[static_cast<std::size_t>(slot)]) continue;
        const std::size_t begin      = static_cast<std::size_t>(slot) * state_size;
        const std::string slot_label = label + " untouched slot " + std::to_string(slot);
        const std::vector<float> expected_slot(initial_states.begin() + begin,
                                               initial_states.begin() + begin + state_size);
        const auto* slot_ptr =
            static_cast<const std::byte*>(states.data()) + begin * state_element_bytes(state_dtype);
        failures += verify_state_exact(slot_label.c_str(), slot_ptr, expected_slot, state_dtype);
    }
    const std::string selector_label = label + " state selectors unchanged";
    failures += verify_exact(selector_label.c_str(),
                             from_device<std::int64_t>(device_state_slots, state_offsets.size()),
                             state_offsets);
    failures += states.verify_guards(label + " states");
    failures += out.verify_guards(label + " out");
    failures += verify_inputs_unchanged(label, aggregate, device);
    return failures;
}

int graph_replay_case(DType state_dtype = DType::FP32) {
    Case c{"chunked graph input updates", 48, 65};
    c.qk_heads            = 16;
    kda_ref::Inputs input = make_inputs(c, 45678U);
    if (state_dtype == DType::BF16) { round_to_bf16(input.state); }
    DeviceInputs device(input);
    GuardedDeviceBuffer state_in(input.state.size() * state_element_bytes(state_dtype));
    GuardedDeviceBuffer state_out(input.state.size() * state_element_bytes(state_dtype));
    GuardedDeviceBuffer out(input.v.size() * sizeof(std::uint16_t));
    upload_state(state_in.data(), input.state, state_dtype);
    Views views(c, device, state_in.span(), state_out.span(), out.span(), state_dtype);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    const auto stream_execution = ninfer::test::cuda_environment().context.execution_view(stream);
    CUDA_CHECK(
        cudaStreamSynchronize(nullptr)); // Drain fixture initialization on the default stream.
    auto launch = [&] {
        ops::kimi_delta_attention(views.q, views.k, views.v, views.g, views.beta, views.a_log,
                                  views.dt_bias, c.lower_bound, kScale, views.workspace,
                                  views.state_in, views.state_out, views.out, stream_execution);
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
        input = make_inputs(c, 56789U + replay);
        if (state_dtype == DType::BF16) { round_to_bf16(input.state); }
        const auto upload_bf16 = [](DeviceBuffer& dst, const std::vector<float>& values) {
            const auto bits = bf16_bits(values);
            dst.copy_from_host(bits.data(), dst.size());
        };
        upload_bf16(device.q, input.q);
        upload_bf16(device.k, input.k);
        upload_bf16(device.v, input.v);
        upload_bf16(device.g, input.g);
        upload_bf16(device.beta, input.beta);
        device.a_log.copy_from_host(input.a_log.data(), device.a_log.size());
        device.dt_bias.copy_from_host(input.dt_bias.data(), device.dt_bias.size());
        upload_state(state_in.data(), input.state, state_dtype);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
        CUDA_CHECK(cudaMemsetAsync(out.data(), 0xff, out.bytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(state_out.data(), 0xff, state_out.bytes(), stream));
        CUDA_CHECK(cudaMemsetAsync(views.scratch.data(), 0xff, views.scratch.bytes(), stream));
        CUDA_CHECK(cudaGraphLaunch(executable, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto ref = kda_ref::evaluate(input, c.lower_bound, kScale);
        failures += verify_result(std::string(c.name) + " " + std::to_string(replay), input, ref,
                                  state_out, out, state_dtype);
        failures += verify_inputs_unchanged(c.name, input, device);
        failures += verify_state_exact("graph state input unchanged", state_in.data(), input.state,
                                       state_dtype);
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
    DeviceBuffer vectors(ninfer::test::cuda_environment().context.device(),
                         static_cast<std::size_t>(kStateDim) * heads * batch_capacity *
                             sizeof(std::uint16_t));
    DeviceBuffer beta_buffer(ninfer::test::cuda_environment().context.device(),
                             static_cast<std::size_t>(heads) * batch_capacity *
                                 sizeof(std::uint16_t));
    DeviceBuffer a_log_buffer(ninfer::test::cuda_environment().context.device(),
                              static_cast<std::size_t>(heads) * sizeof(float));
    DeviceBuffer dt_buffer(ninfer::test::cuda_environment().context.device(),
                           static_cast<std::size_t>(kStateDim) * heads * sizeof(float));
    DeviceBuffer state_buffer(ninfer::test::cuda_environment().context.device(),
                              static_cast<std::size_t>(kStateDim) * kStateDim * heads * 2 *
                                  sizeof(float));
    DeviceBuffer slot_buffer(ninfer::test::cuda_environment().context.device(),
                             static_cast<std::size_t>(batch_capacity) * sizeof(std::int64_t));

    const auto rejects = [&](int tokens, int batch, DType slot_dtype,
                             DType src_state_dtype = DType::FP32,
                             DType dst_state_dtype = DType::FP32) {
        Tensor q(vectors.span(), DType::BF16, {kStateDim, heads, tokens, batch});
        Tensor k(vectors.span(), DType::BF16, {kStateDim, heads, tokens, batch});
        Tensor v(vectors.span(), DType::BF16, {kStateDim, heads, tokens, batch});
        Tensor g(vectors.span(), DType::BF16, {kStateDim, heads, tokens, batch});
        Tensor beta(beta_buffer.span(), DType::BF16, {heads, tokens, batch});
        Tensor a_log(a_log_buffer.span(), DType::FP32, {heads});
        Tensor dt_bias(dt_buffer.span(), DType::FP32, {kStateDim, heads});
        Tensor src_states(state_buffer.span(), src_state_dtype, {kStateDim, kStateDim, heads, 2});
        Tensor dst_states(state_buffer.span(), dst_state_dtype, {kStateDim, kStateDim, heads, 2});
        Tensor state_slots(slot_buffer.span(), slot_dtype, {batch});
        Tensor out(vectors.span(), DType::BF16, {kStateDim, heads, tokens, batch});
        try {
            ops::kimi_delta_attention_batch_update(
                q, k, v, g, beta, a_log, dt_bias, kLowerBound, kScale,
                dense_record_field(src_states, 3, state_slots),
                dense_record_field(dst_states, 3, state_slots), {}, out,
                ninfer::test::cuda_environment().default_execution);
        } catch (const std::invalid_argument&) { return true; }
        cuda_synchronize();
        return false;
    };

    int failures = 0;
    if (!rejects(2, 2, DType::I64)) {
        std::cerr << "kimi_delta_attention batch update accepted W=2\n";
        ++failures;
    }
    if (!rejects(1, 2, DType::FP32)) {
        std::cerr << "kimi_delta_attention batch update accepted FP32 state selectors\n";
        ++failures;
    }
    if (!rejects(1, 2, DType::I64, DType::FP16, DType::FP16)) {
        std::cerr << "kimi_delta_attention batch update accepted FP16 state\n";
        ++failures;
    }
    if (!rejects(1, 2, DType::I64, DType::FP32, DType::BF16) ||
        !rejects(1, 2, DType::I64, DType::BF16, DType::FP32)) {
        std::cerr << "kimi_delta_attention batch update accepted mixed FP32/BF16 states\n";
        ++failures;
    }
    return failures;
}

int contract_rejection_cases() {
    constexpr int heads = 2;
    DeviceBuffer vectors(ninfer::test::cuda_environment().context.device(),
                         static_cast<std::size_t>(kStateDim) * heads * sizeof(std::uint16_t));
    DeviceBuffer beta_buffer(ninfer::test::cuda_environment().context.device(),
                             static_cast<std::size_t>(heads) * sizeof(std::uint16_t));
    DeviceBuffer a_log_buffer(ninfer::test::cuda_environment().context.device(),
                              static_cast<std::size_t>(heads) * sizeof(float));
    DeviceBuffer dt_buffer(ninfer::test::cuda_environment().context.device(),
                           static_cast<std::size_t>(kStateDim) * heads * sizeof(float));
    DeviceBuffer state_buffer(ninfer::test::cuda_environment().context.device(),
                              static_cast<std::size_t>(kStateDim) * kStateDim * heads *
                                  sizeof(float));

    Tensor q(vectors.span(), DType::BF16, {kStateDim, heads, 1});
    Tensor k(vectors.span(), DType::BF16, {kStateDim, heads, 1});
    Tensor v(vectors.span(), DType::BF16, {kStateDim, heads, 1});
    Tensor g(vectors.span(), DType::BF16, {kStateDim, heads, 1});
    Tensor beta(beta_buffer.span(), DType::BF16, {heads, 1});
    Tensor a_log(a_log_buffer.span(), DType::FP32, {heads});
    Tensor dt_bias(dt_buffer.span(), DType::FP32, {kStateDim, heads});
    Tensor state(state_buffer.span(), DType::FP32, {kStateDim, kStateDim, heads});
    Tensor out(vectors.span(), DType::BF16, {kStateDim, heads, 1});

    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(), 256);

    MemoryArena workspace(workspace_storage.span());
    const auto rejects = [&](const Tensor& test_g, float lower_bound, float scale) {
        try {
            ops::kimi_delta_attention(q, k, v, test_g, beta, a_log, dt_bias, lower_bound, scale,
                                      workspace, state, out, cuda_environment().default_execution);
        } catch (const std::invalid_argument&) { return true; }
        cuda_synchronize();
        return false;
    };

    int failures  = 0;
    auto wrong_g  = g;
    wrong_g.dtype = DType::FP32;
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
    const auto rejects_states = [&](DType src_dtype, DType dst_dtype) {
        Tensor s_in(state_buffer.span(), src_dtype, {kStateDim, kStateDim, heads});
        Tensor s_out(state_buffer.span(), dst_dtype, {kStateDim, kStateDim, heads});
        try {
            ops::kimi_delta_attention(q, k, v, g, beta, a_log, dt_bias, kLowerBound, kScale,
                                      workspace, static_cast<const Tensor&>(s_in), s_out, out,
                                      cuda_environment().default_execution);
        } catch (const std::invalid_argument&) { return true; }
        cuda_synchronize();
        return false;
    };
    if (!rejects_states(DType::FP16, DType::FP16)) {
        std::cerr << "kimi_delta_attention accepted FP16 state\n";
        ++failures;
    }
    if (!rejects_states(DType::FP32, DType::BF16) || !rejects_states(DType::BF16, DType::FP32)) {
        std::cerr << "kimi_delta_attention accepted mixed FP32/BF16 states\n";
        ++failures;
    }
    RaggedFixture ragged({1});
    DeviceBuffer slot_buffer = to_device(std::vector<std::int64_t>{0});
    Tensor slot_table(slot_buffer.span(), DType::I64, {1});
    const auto rejects_ragged_states = [&](DType src_dtype, DType dst_dtype) {
        Tensor s_in(state_buffer.span(), src_dtype, {kStateDim, kStateDim, heads, 1});
        Tensor s_out(state_buffer.span(), dst_dtype, {kStateDim, kStateDim, heads, 1});
        try {
            ops::kimi_delta_attention(q, k, v, g, beta, a_log, dt_bias, kLowerBound, kScale,
                                      ragged.view, workspace,
                                      dense_record_field(s_in, 3, slot_table),
                                      dense_record_field(s_out, 3, slot_table), out,
                                      cuda_environment().default_execution);
        } catch (const std::invalid_argument&) { return true; }
        cuda_synchronize();
        return false;
    };
    if (!rejects_ragged_states(DType::FP16, DType::FP16)) {
        std::cerr << "kimi_delta_attention ragged accepted FP16 state\n";
        ++failures;
    }
    if (!rejects_ragged_states(DType::FP32, DType::BF16) ||
        !rejects_ragged_states(DType::BF16, DType::FP32)) {
        std::cerr << "kimi_delta_attention ragged accepted mixed FP32/BF16 states\n";
        ++failures;
    }
    return failures;
}

enum class RaggedState { Distinct, InPlace, Mixed, SharedSource };
enum class RaggedInputs { Random, Adversarial, Cancellation, PoisonedNeighbor };

int ragged_recurrence(const std::string& label, std::vector<int> lengths, int qk_heads, int heads,
                      RaggedState form, bool graph = false, bool tight = false, int min_tokens = 0,
                      int max_tokens = 0, RaggedInputs inputs = RaggedInputs::Random,
                      DType state_dtype = DType::FP32) {
    RaggedFixture ragged(std::move(lengths));
    const int n = ragged.tokens(), rows = ragged.segments();
    const int source_slots           = form == RaggedState::SharedSource ? 2 : rows;
    const int slots                  = source_slots + rows + 1;
    const std::size_t state_elements = 128 * 128 * heads;
    const std::size_t out_column     = 128 * heads;
    const std::size_t qk_column      = 128 * qk_heads;
    const bool cancellation          = inputs == RaggedInputs::Cancellation;
    const bool poisoned_neighbor     = inputs == RaggedInputs::PoisonedNeighbor;
    const Case c{"ragged", heads, n, false, false, false, qk_heads, cancellation ? 0.0F : -0.5F};
    auto make_packed = [&](unsigned seed) {
        auto input = make_inputs(c, seed);
        if (cancellation) {
            std::fill(input.q.begin(), input.q.end(), 0.0F);
            std::fill(input.k.begin(), input.k.end(), 0.0F);
            std::fill(input.v.begin(), input.v.end(), 0.0F);
            std::fill(input.state.begin(), input.state.end(), 0.0F);
            input.q[0]     = 1.0F;
            input.q[1]     = -1.0F;
            input.state[0] = 1.0F + 0x1p-9F;
            input.state[1] = 1.0F;
        }
        if (poisoned_neighbor) {
            const auto first = input.v.begin() + ragged.prefix[1] * out_column;
            for (std::size_t i = 0; i < out_column; ++i)
                first[i] = i % 2 ? std::numeric_limits<float>::infinity()
                                 : std::numeric_limits<float>::quiet_NaN();
        }
        if (inputs != RaggedInputs::Adversarial) return input;
        for (int s = 1; s < rows; ++s) {
            const int col = ragged.prefix[s];
            // A conspicuous first row adjacent to the preceding segment's padded tail.
            for (std::size_t i = 0; i < qk_column; ++i) {
                input.q[col * qk_column + i] *= 64;
                input.k[col * qk_column + i] *= -64;
            }
            for (std::size_t i = 0; i < out_column; ++i) {
                input.v[col * out_column + i] *= -8;
                input.g[col * out_column + i] = i % 2 ? -12 : 12;
            }
            for (int h = 0; h < heads; ++h) input.beta[col * heads + h] = h % 2 ? -8 : 8;
        }
        return input;
    };
    auto packed = make_packed(937);
    DeviceInputs device(packed);
    GuardedDeviceBuffer states(slots * state_elements * state_element_bytes(state_dtype)),
        output(packed.v.size() * 2);
    states.fill(0);
    RecordStateFixture history(Tensor(states.span(), state_dtype, {128, 128, heads, slots}), 3);
    auto so  = to_device(std::vector<std::int64_t>(rows));
    auto do_ = to_device(std::vector<std::int64_t>(rows));
    auto src = history.view(Tensor(so.span(), DType::I64, {rows}));
    auto dst = history.view(Tensor(do_.span(), DType::I64, {rows}));
    Tensor q(device.q.span(), DType::BF16, {128, qk_heads, n});
    Tensor k(device.k.span(), DType::BF16, {128, qk_heads, n});
    Tensor v(device.v.span(), DType::BF16, {128, heads, n});
    Tensor g(device.g.span(), DType::BF16, {128, heads, n});
    Tensor beta(device.beta.span(), DType::BF16, {heads, n});
    Tensor a(device.a_log.span(), DType::FP32, {heads});
    Tensor dt(device.dt_bias.span(), DType::FP32, {128, heads});
    Tensor out(output.span(), DType::BF16, {128, heads, n});
    const int chunked   = std::count_if(ragged.lengths.begin(), ragged.lengths.end(),
                                        [](int length) { return length >= 12; });
    const auto capacity = ops::kimi_delta_attention_ragged_workspace_capacity_bytes(
        qk_heads, heads, min_tokens ? min_tokens : n, max_tokens ? max_tokens : n, rows, rows);
    const int longest = *std::max_element(ragged.lengths.begin(), ragged.lengths.end());
    // A direct segment must fit in the dense allocation for its exact length.
    const auto bytes =
        chunked <= 1
            ? ops::kimi_delta_attention_workspace_capacity_bytes(qk_heads, heads, longest, longest)
            : capacity;
    if (capacity < bytes) {
        std::cerr << label << ": insufficient ragged workspace query\n";
        return 1;
    }
    GuardedDeviceBuffer scratch(bytes);
    MemoryArena ws(scratch.span());
    auto launch = [&](CudaExecutionView ex) {
        ops::kimi_delta_attention(q, k, v, g, beta, a, dt, c.lower_bound, kScale, ragged.view, ws,
                                  src, dst, out, ex);
    };

    cudaStream_t stream        = nullptr;
    cudaGraph_t captured       = nullptr;
    cudaGraphExec_t executable = nullptr;
    if (graph) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto stream_execution = cuda_environment().context.execution_view(stream);
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        launch(stream_execution);
        CUDA_CHECK(cudaStreamEndCapture(stream, &captured));
        CUDA_CHECK(cudaGraphInstantiate(&executable, captured, 0));
    }

    int failures = 0;
    for (int replay = 0; replay < (graph ? 2 : 1); ++replay) {
        if (replay) {
            auto next         = make_packed(1153);
            const auto upload = [](DeviceBuffer& target, const std::vector<float>& values) {
                const auto bits = bf16_bits(values);
                target.copy_from_host(bits.data(), target.size());
            };
            upload(device.q, next.q);
            upload(device.k, next.k);
            upload(device.v, next.v);
            upload(device.g, next.g);
            upload(device.beta, next.beta);
            next.a_log   = packed.a_log;
            next.dt_bias = packed.dt_bias;
            packed       = std::move(next);
        }
        std::vector<int> source(rows), destination(rows);
        for (int s = 0; s < rows; ++s) {
            source[s] = form == RaggedState::SharedSource ? (s + replay) % 2 : rows - 1 - s;
            destination[s] =
                form == RaggedState::InPlace || (form == RaggedState::Mixed && s % 2 == 0)
                    ? source[s]
                    : source_slots + (s + replay) % rows;
        }
        // A replay must read the current selectors. Reverse the order of the distinct
        // sources, and move the lone source/destination pair for a one-segment graph.
        if (replay && form == RaggedState::Distinct) {
            for (auto& slot : source) slot = rows - 1 - slot;
            if (rows == 1) std::swap(source[0], destination[0]);
        }
        const auto sources      = history.record_offsets(source);
        const auto destinations = history.record_offsets(destination);
        so.copy_from_host(sources.data(), so.size());
        do_.copy_from_host(destinations.data(), do_.size());

        auto input_state = [&](int slot) {
            if (cancellation) return packed.state;
            std::vector<float> state(state_elements);
            for (std::size_t i = 0; i < state_elements; ++i)
                state[i] = packed.state[i] + 0.005F * ((slot % 11) - 5);
            if (state_dtype == DType::BF16) { round_to_bf16(state); }
            return state;
        };
        const std::vector<float> sentinel(state_elements, -0.0625F);
        const std::vector<float> poison(state_elements, std::numeric_limits<float>::quiet_NaN());
        std::vector<bool> is_source(slots), is_destination(slots);
        for (int s : source) is_source[s] = true;
        for (int s : destination) is_destination[s] = true;
        for (int slot = 0; slot < slots; ++slot) {
            const auto field    = history.tensor(slot);
            const auto& initial = is_source[slot]        ? input_state(slot)
                                  : is_destination[slot] ? poison
                                                         : sentinel;
            upload_state(field.data, initial, state_dtype);
        }
        scratch.fill(0xff);
        output.fill(0xff);
        if (graph) {
            CUDA_CHECK(cudaGraphLaunch(executable, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            launch(cuda_environment().default_execution);
            cuda_synchronize();
        }

        failures += output.verify_guards(label) + scratch.verify_guards(label) +
                    states.verify_guards(label) + history.verify_guards(label);
        if (ws.used() != 0 || ws.peak_used() > bytes || (tight && ws.peak_used() != bytes)) {
            std::cerr << label << ": workspace capacity " << bytes << ", peak " << ws.peak_used()
                      << ", used after call " << ws.used() << '\n';
            ++failures;
        }
        const auto actual = from_device_bf16(output.data(), packed.v.size());
        for (int s = 0; s < rows; ++s) {
            if (poisoned_neighbor && s == 1) continue;
            const auto slice = [&](const std::vector<float>& tensor, std::size_t stride) {
                return std::vector<float>(tensor.begin() + ragged.prefix[s] * stride,
                                          tensor.begin() + ragged.prefix[s + 1] * stride);
            };
            const kda_ref::Inputs step{qk_heads,
                                       heads,
                                       ragged.lengths[s],
                                       slice(packed.q, qk_column),
                                       slice(packed.k, qk_column),
                                       slice(packed.v, out_column),
                                       slice(packed.g, out_column),
                                       slice(packed.beta, heads),
                                       packed.a_log,
                                       packed.dt_bias,
                                       input_state(source[s])};
            const auto expected = kda_ref::evaluate(step, c.lower_bound, kScale);
            const auto name =
                label + " replay " + std::to_string(replay) + " segment " + std::to_string(s);
            const auto first = actual.begin() + ragged.prefix[s] * out_column;
            const auto last  = actual.begin() + ragged.prefix[s + 1] * out_column;
            failures += verify_reduction(name + " output", std::vector<double>(first, last),
                                         expected.out, output_criterion());
            const auto state =
                read_state(history.tensor(destination[s]).data, state_elements, state_dtype);
            failures += verify_reduction(name + " state", state, expected.final_state,
                                         state_criterion(state_dtype));
        }
        for (int slot = 0; slot < slots; ++slot) {
            if (is_destination[slot]) continue;
            failures += verify_state_exact("ragged unselected state", history.tensor(slot).data,
                                           is_source[slot] ? input_state(slot) : sentinel,
                                           state_dtype);
        }
    }
    if (graph) {
        CUDA_CHECK(cudaGraphExecDestroy(executable));
        CUDA_CHECK(cudaGraphDestroy(captured));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    return failures;
}

int ragged_workspace_envelopes() {
    constexpr std::size_t packet_column = 14848 * 32;
    const auto bytes                    = [](int min_t, int max_t, int min_s, int max_s) {
        return ops::kimi_delta_attention_ragged_workspace_capacity_bytes(32, 32, min_t, max_t,
                                                                                            min_s, max_s);
    };
#if NINFER_TARGET_SPARK
    const std::vector<std::size_t> expected{0, 0, 16, 16, 16, 8, 16, 16, 16, 16, 8, 1};
#else
    const std::vector<std::size_t> expected{0, 0, 17, 17, 16, 256, 257, 257, 17, 18, 218, 1};
#endif
    constexpr int largest = std::numeric_limits<int>::max();
    const std::vector<std::size_t> actual{
        bytes(1, 11, 1, 1),          bytes(160, 170, 160, 160),
        bytes(255, 257, 1, 1),       bytes(256, 257, 1, 1),
        bytes(255, 256, 1, 1),       bytes(257, 4096, 1, 1),
        bytes(240, 4096, 2, 2),      bytes(257, 4096, 2, 2),
        bytes(244, 257, 2, 2),       bytes(257, 257, 2, 3),
        bytes(2000, 2000, 100, 100), bytes(largest, largest, largest - 11, largest - 11)};
    std::vector<std::size_t> expected_bytes;
    for (const auto packets : expected) expected_bytes.push_back(packets * packet_column);
    expected_bytes[10] += 100 * 2 * sizeof(int);
    return verify_exact("ragged workspace envelope high watermarks", actual, expected_bytes);
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    const auto ragged = [&suite](std::string label, std::vector<int> lengths, int qk, int value,
                                 RaggedState form, bool graph = false, bool tight = false,
                                 int min_tokens = 0, int max_tokens = 0,
                                 RaggedInputs inputs = RaggedInputs::Random) {
        suite.add("kimi_delta_attention/ragged/" + label,
                  [=] {
                      return ragged_recurrence(label, lengths, qk, value, form, graph, tight,
                                               min_tokens, max_tokens, inputs);
                  },
                  {{"execution", graph ? "graph" : "eager"}});
    };
    using enum RaggedState;
    for (const int t : {11, 12, 255, 256, 257})
        ragged("single_" + std::to_string(t), {t}, 32, 32, Distinct, false, true);
#if NINFER_TARGET_SPARK
    ragged("single_interior_peak", {256}, 8, 32, InPlace, false, true, 255, 257);
    ragged("direct_interior_peak", {256, 1}, 8, 32, Distinct, false, true, 255, 257);
#endif
    ragged("partial_chunks", {1, 2, 11, 12, 15, 16, 17, 65}, 4, 16, Mixed, true, false, 0, 0,
           RaggedInputs::Adversarial);
    ragged("fp32_short_readout", {1, 12}, 4, 16, Distinct, false, true, 0, 0,
           RaggedInputs::Cancellation);
    ragged("poisoned_neighbor", {17, 17}, 4, 16, Distinct, false, true, 0, 0,
           RaggedInputs::PoisonedNeighbor);
    ragged("window_handoff_two", {140, 140}, 32, 32, Distinct);
    ragged("window_handoff_three", {200, 80, 80}, 8, 32, Distinct, true);
    ragged("window_handoff_larger_cap", {300, 140, 140}, 8, 32, Distinct, true);
    ragged("long_then_short", {4095, 1}, 4, 16, InPlace, false, true);
    ragged("short_then_long", {1, 4095}, 8, 32, SharedSource, false, true);
    ragged("eight_by_three", std::vector<int>(8, 3), 32, 32, Mixed, true, true);
    ragged("thirty_two_by_one", std::vector<int>(32, 1), 8, 32, SharedSource, false, true);
    auto spill = std::vector<int>(160, 1);
    for (int s = 0; s < 130; s += 2) spill[s] = 12;
    ragged("device_chunk_offsets", std::move(spill), 8, 32, SharedSource, true, false, 0, 0,
           RaggedInputs::Adversarial);
    suite.add("kimi_delta_attention/ragged/workspace_envelopes", ragged_workspace_envelopes);
    suite.add("kda/contract", contract_rejection_cases);
    suite.add("kda/batch_contract", batch_contract_rejection_cases);
    suite.add("kda/graph_updates", [] { return graph_replay_case(); }, {{"execution", "graph"}});
    const auto id = [](const Case& c, const char* form) {
        return std::string("kda/") + form + "/" + case_label(c);
    };
    const auto inplace = [&](Case c, unsigned seed) {
        suite.add(id(c, "inplace"), [=] { return inplace_case(c, seed); });
    };
    const auto distinct = [&](Case c, unsigned seed, bool decode = false) {
        suite.add(id(c, decode ? "continuation" : "distinct"),
                  [=] { return distinct_case(c, seed, decode); });
    };
    const auto distinct_exact_alias = [&](Case c, unsigned seed) {
        suite.add(id(c, "alias"), [=] { return distinct_exact_alias_case(c, seed); });
    };
    const auto batch_update = [&](Case c, std::vector<int> selected, int slots, unsigned seed) {
        suite.add(id(c, "batch"), [=] { return batch_update_case(c, selected, slots, seed); });
    };
    inplace({"Kimi Linear ordinary", 32, 7}, 32007U);
    distinct({"Kimi Linear ordinary", 32, 7}, 32107U);
    inplace({"Kimi K3 ordinary", 96, 3}, 96003U);
    distinct({"Kimi K3 zero-state", 96, 3, false, false, true}, 96103U);
    inplace({"runtime head count", 5, 2}, 502U);
    inplace({"near-zero normalized QK", 32, 1, true}, 32201U);
    distinct({"saturated safe gate", 32, 2, false, true}, 32202U);
    distinct_exact_alias({"state alias contract", 32, 2}, 32302U);
    batch_update({"Kimi Linear ordinary", 32, 1}, {4, 0, 5, 2}, 6, 32401U);
    batch_update({"Kimi K3 ordinary", 96, 1}, {10, 3, 7, 0, 9, 5, 1, 8}, 11, 96801U);
    batch_update({"runtime head count", 5, 1}, {3, 0, 4}, 5, 50301U);
    std::vector<int> wide(17);
    for (int row = 0; row < int(wide.size()); ++row) wide[row] = row % 3 ? row : -1;
    suite.add("kda/batch/expanded",
              [=] { return batch_update_case({"masked expanded batch", 1, 1}, wide, 20, 17403U); },
              {{"suite", "concurrency"}});

    for (const auto [qheads, vheads] :
         {std::pair{16, 48}, {16, 32}, {96, 96}, {64, 64}, {1, 32}, {5, 15}}) {
        Case c{"grouped recurrent", vheads, 7};
        c.qk_heads = qheads;
        distinct(c, 31415U + vheads);
        c.tokens = 1;
        batch_update(c, {5, 1, 3}, 7, 27182U + vheads);
        c.tokens = 129;
        c.name   = "chunked grouped";
        distinct(c, 16180U + vheads);
        inplace(c, 14142U + vheads);
        if (qheads > 1 && vheads >= 32) {
            c.tokens = 1025;
            distinct(c, 53212U + vheads);
        }
    }
    for (int t : {11, 12, 13, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257, 1025}) {
        Case c{"chunk boundary", 3, t};
        c.qk_heads = 1;
        distinct(c, 90000U + t);
    }
    for (int t : {255, 256, 257}) {
        Case c{"workspace interval", 32, t};
        c.capacity_tokens = 4096;
        distinct(c, 45200U + t);
    }
    inplace({"windowed state alias", 32, 513}, 44513U);
    Case weak{"weak decay long", 3, 16385};
    weak.qk_heads   = 1;
    weak.weak_decay = true;
    distinct(weak, 99887U, true);
    weak.lower_bound = 0.0F;
    weak.name        = "zero decay long";
    distinct(weak, 88776U, true);
    weak.name       = "small beta zero decay long";
    weak.small_beta = true;
    distinct(weak, 332213U, true);
    Case cancellation{"near collinear residual cancellation", 3, 4097};
    cancellation.qk_heads         = 1;
    cancellation.lower_bound      = 0.0F;
    cancellation.collinear_cancel = true;
    distinct(cancellation, 99891U);
    distinct({"chunk saturated", 5, 257, false, true}, 77665U);
    distinct({"chunk near zero QK", 5, 65, true}, 66554U);
    distinct_exact_alias({"chunk state alias", 32, 65}, 55443U);

    // BF16 recurrent state coverage across recurrent, chunked, batch, graph, and ragged paths.
    const auto ragged_bf16 = [&suite](std::string label, std::vector<int> lengths, int qk,
                                      int value, RaggedState form, bool graph = false,
                                      bool tight = false, int min_tokens = 0, int max_tokens = 0,
                                      RaggedInputs inputs = RaggedInputs::Random) {
        suite.add("kimi_delta_attention/ragged_bf16/" + label,
                  [=] {
                      return ragged_recurrence(label, lengths, qk, value, form, graph, tight,
                                               min_tokens, max_tokens, inputs, DType::BF16);
                  },
                  {{"execution", graph ? "graph" : "eager"}});
    };
    for (const int t : {11, 12, 256, 257})
        ragged_bf16("single_" + std::to_string(t), {t}, 32, 32, Distinct, false, true);
    ragged_bf16("partial_chunks", {1, 2, 11, 12, 15, 16, 17, 65}, 4, 16, Mixed, true, false, 0, 0,
                RaggedInputs::Adversarial);
    ragged_bf16("poisoned_neighbor", {17, 17}, 4, 16, Distinct, false, true, 0, 0,
                RaggedInputs::PoisonedNeighbor);
    ragged_bf16("window_handoff_two", {140, 140}, 32, 32, Distinct);
    ragged_bf16("window_handoff_three", {200, 80, 80}, 8, 32, Distinct, true);
    ragged_bf16("long_then_short", {4095, 1}, 4, 16, InPlace, false, true);
    ragged_bf16("short_then_long", {1, 4095}, 8, 32, SharedSource, false, true);
    ragged_bf16("eight_by_three", std::vector<int>(8, 3), 32, 32, Mixed, true, true);
    ragged_bf16("thirty_two_by_one", std::vector<int>(32, 1), 8, 32, SharedSource, false, true);
    auto spill_bf16 = std::vector<int>(160, 1);
    for (int s = 0; s < 130; s += 2) spill_bf16[s] = 12;
    ragged_bf16("device_chunk_offsets", std::move(spill_bf16), 8, 32, SharedSource, true, false, 0,
                0, RaggedInputs::Adversarial);

    suite.add("kda_bf16/graph_updates", [] { return graph_replay_case(DType::BF16); },
              {{"execution", "graph"}});
    const auto id_bf16 = [](const Case& c, const char* form) {
        return std::string("kda_bf16/") + form + "/" + case_label(c);
    };
    const auto inplace_bf16 = [&](Case c, unsigned seed) {
        suite.add(id_bf16(c, "inplace"), [=] { return inplace_case(c, seed, DType::BF16); });
    };
    const auto distinct_bf16 = [&](Case c, unsigned seed) {
        suite.add(id_bf16(c, "distinct"), [=] { return distinct_case(c, seed, false, DType::BF16); });
    };
    const auto alias_bf16 = [&](Case c, unsigned seed) {
        suite.add(id_bf16(c, "alias"),
                  [=] { return distinct_exact_alias_case(c, seed, DType::BF16); });
    };
    const auto batch_bf16 = [&](Case c, std::vector<int> selected, int slots, unsigned seed) {
        suite.add(id_bf16(c, "batch"),
                  [=] { return batch_update_case(c, selected, slots, seed, DType::BF16); });
    };
    inplace_bf16({"Kimi Linear ordinary", 32, 7}, 32007U);
    distinct_bf16({"Kimi Linear ordinary", 32, 7}, 32107U);
    inplace_bf16({"Kimi K3 ordinary", 96, 3}, 96003U);
    distinct_bf16({"Kimi K3 zero-state", 96, 3, false, false, true}, 96103U);
    inplace_bf16({"runtime head count", 5, 2}, 502U);
    inplace_bf16({"near-zero normalized QK", 32, 1, true}, 32201U);
    distinct_bf16({"saturated safe gate", 32, 2, false, true}, 32202U);
    alias_bf16({"state alias contract", 32, 2}, 32302U);
    batch_bf16({"Kimi Linear ordinary", 32, 1}, {4, 0, 5, 2}, 6, 32401U);
    batch_bf16({"Kimi K3 ordinary", 96, 1}, {10, 3, 7, 0, 9, 5, 1, 8}, 11, 96801U);
    batch_bf16({"runtime head count", 5, 1}, {3, 0, 4}, 5, 50301U);
    suite.add("kda_bf16/batch/expanded",
              [=] {
                  return batch_update_case({"masked expanded batch", 1, 1}, wide, 20, 17403U,
                                           DType::BF16);
              },
              {{"suite", "concurrency"}});
    for (const auto [qheads, vheads] : {std::pair{16, 48}, {16, 32}, {1, 32}, {5, 15}}) {
        Case c{"grouped recurrent", vheads, 7};
        c.qk_heads = qheads;
        distinct_bf16(c, 31415U + vheads);
        c.tokens = 1;
        batch_bf16(c, {5, 1, 3}, 7, 27182U + vheads);
        c.tokens = 129;
        c.name   = "chunked grouped";
        distinct_bf16(c, 16180U + vheads);
        inplace_bf16(c, 14142U + vheads);
        if (qheads > 1 && vheads >= 32) {
            c.tokens = 1025;
            distinct_bf16(c, 53212U + vheads);
        }
    }
    for (int t : {11, 12, 16, 17, 64, 65, 255, 256, 257}) {
        Case c{"chunk boundary", 3, t};
        c.qk_heads = 1;
        distinct_bf16(c, 90000U + t);
    }
    inplace_bf16({"windowed state alias", 32, 513}, 44513U);
    distinct_bf16({"chunk saturated", 5, 257, false, true}, 77665U);
    distinct_bf16({"chunk near zero QK", 5, 65, true}, 66554U);
    alias_bf16({"chunk state alias", 32, 65}, 55443U);


    return suite.main(argc, argv);
}
