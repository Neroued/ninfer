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

constexpr ReductionCriterion output_criterion() {
    return {/*relative_l2=*/2.2e-3, /*gross_absolute=*/1.0e-5,
            /*gross_relative_to_max_reference=*/4.0e-3};
}

constexpr ReductionCriterion state_criterion() {
    return {/*relative_l2=*/1.5e-5, /*gross_absolute=*/5.0e-6,
            /*gross_relative_to_max_reference=*/3.0e-5};
}

struct Case {
    const char* name;
    int heads;
    int tokens;
    bool near_zero_qk   = false;
    bool saturated_gate = false;
    bool zero_state     = false;
};

void fill_uniform(std::vector<float>& values, std::mt19937& generator, float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    for (float& value : values) { value = distribution(generator); }
}

kda_ref::Inputs make_inputs(const Case& test_case, std::uint32_t seed) {
    kda_ref::Inputs in;
    in.heads  = test_case.heads;
    in.tokens = test_case.tokens;

    const std::size_t vector_size =
        static_cast<std::size_t>(kStateDim) * test_case.heads * test_case.tokens;
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim) * kStateDim * test_case.heads;
    in.q.resize(vector_size);
    in.k.resize(vector_size);
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
        : q(device.q.p, DType::BF16, {kStateDim, test_case.heads, test_case.tokens}),
          k(device.k.p, DType::BF16, {kStateDim, test_case.heads, test_case.tokens}),
          v(device.v.p, DType::BF16, {kStateDim, test_case.heads, test_case.tokens}),
          g(device.g.p, DType::BF16, {kStateDim, test_case.heads, test_case.tokens}),
          beta(device.beta.p, DType::BF16, {test_case.heads, test_case.tokens}),
          a_log(device.a_log.p, DType::FP32, {test_case.heads}),
          dt_bias(device.dt_bias.p, DType::FP32, {kStateDim, test_case.heads}),
          state_in(state_in_data, DType::FP32, {kStateDim, kStateDim, test_case.heads}),
          state_out(state_out_data, DType::FP32, {kStateDim, kStateDim, test_case.heads}),
          out(out_data, DType::BF16, {kStateDim, test_case.heads, test_case.tokens}) {}

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
                  const GuardedDeviceBuffer& out) {
    int failures = 0;
    failures += verify_reduction(label + " out", from_device_bf16(out.data(), in.v.size()),
                                 reference.out, output_criterion());
    failures += verify_reduction(label + " state",
                                 doubles(from_device<float>(state.data(), in.state.size())),
                                 reference.final_state, state_criterion());
    failures += state.verify_guards(label + " state");
    failures += out.verify_guards(label + " out");
    return failures;
}

int inplace_case(const Case& test_case, std::uint32_t seed) {
    const kda_ref::Inputs in = make_inputs(test_case, seed);
    const kda_ref::Result reference =
        kda_ref::evaluate(in, static_cast<double>(kLowerBound), static_cast<double>(kScale));
    DeviceInputs device(in);
    GuardedDeviceBuffer state(in.state.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state.copy_from_host(in.state.data(), state.bytes());
    out.fill(0xff);
    Views views(test_case, device, state.data(), state.data(), out.data());

    ops::kimi_delta_attention(views.q, views.k, views.v, views.g, views.beta, views.a_log,
                              views.dt_bias, kLowerBound, kScale, views.state_out, views.out,
                              nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) + " inplace";
    int failures            = verify_result(label, in, reference, state, out);
    failures += verify_inputs_unchanged(label, in, device);
    return failures;
}

int distinct_case(const Case& test_case, std::uint32_t seed) {
    const kda_ref::Inputs in = make_inputs(test_case, seed);
    const kda_ref::Result reference =
        kda_ref::evaluate(in, static_cast<double>(kLowerBound), static_cast<double>(kScale));
    DeviceInputs device(in);
    GuardedDeviceBuffer state_in(in.state.size() * sizeof(float));
    GuardedDeviceBuffer state_out(in.state.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state_in.copy_from_host(in.state.data(), state_in.bytes());
    state_out.fill(0xff);
    out.fill(0xff);
    Views views(test_case, device, state_in.data(), state_out.data(), out.data());

    ops::kimi_delta_attention(
        views.q, views.k, views.v, views.g, views.beta, views.a_log, views.dt_bias, kLowerBound,
        kScale, static_cast<const Tensor&>(views.state_in), views.state_out, views.out, nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) + " distinct";
    int failures            = verify_result(label, in, reference, state_out, out);
    failures += verify_exact((label + " state-in unchanged").c_str(),
                             from_device<float>(state_in.data(), in.state.size()), in.state);
    failures += state_in.verify_guards(label + " state-in");
    failures += verify_inputs_unchanged(label, in, device);
    return failures;
}

int distinct_exact_alias_case(const Case& test_case, std::uint32_t seed) {
    const kda_ref::Inputs in = make_inputs(test_case, seed);
    const kda_ref::Result reference =
        kda_ref::evaluate(in, static_cast<double>(kLowerBound), static_cast<double>(kScale));
    DeviceInputs device(in);
    GuardedDeviceBuffer state(in.state.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state.copy_from_host(in.state.data(), state.bytes());
    out.fill(0xff);
    Views views(test_case, device, state.data(), state.data(), out.data());

    ops::kimi_delta_attention(
        views.q, views.k, views.v, views.g, views.beta, views.a_log, views.dt_bias, kLowerBound,
        kScale, static_cast<const Tensor&>(views.state_in), views.state_out, views.out, nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) + " distinct exact-alias";
    int failures            = verify_result(label, in, reference, state, out);
    failures += verify_inputs_unchanged(label, in, device);
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

    const auto rejects = [&](const Tensor& test_g, float lower_bound, float scale) {
        try {
            ops::kimi_delta_attention(q, k, v, test_g, beta, a_log, dt_bias, lower_bound, scale,
                                      state, out, nullptr);
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
    failures += inplace_case({"Kimi Linear ordinary", 32, 7}, 32007U);
    failures += distinct_case({"Kimi Linear ordinary", 32, 7}, 32107U);
    failures += inplace_case({"Kimi K3 ordinary", 96, 3}, 96003U);
    failures += distinct_case({"Kimi K3 zero-state", 96, 3, false, false, true}, 96103U);
    failures += inplace_case({"runtime head count", 5, 2}, 502U);
    failures += inplace_case({"near-zero normalized QK", 32, 1, true}, 32201U);
    failures += distinct_case({"saturated safe gate", 32, 2, false, true}, 32202U);
    failures += distinct_exact_alias_case({"state alias contract", 32, 2}, 32302U);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " kimi_delta_attention correctness\n";
    return failures == 0 ? 0 : 1;
}
