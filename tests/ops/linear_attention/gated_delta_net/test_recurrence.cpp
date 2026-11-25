#include "ops/suite.h"
#include "ops/ragged_fixture.h"
#include "ops/state_fixture.h"
#include "ninfer/ops/gated_delta_net.h"
#include "cuda_fixture.h"
#include "core/buffer.h"
#include "core/decode_graph.h"
#include "core/platform.h"

#include "gdn_ref.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kStateDim = 128;

// One error budget for the complete recurrence over represented public inputs, including
// continuation. FP32 state storage does not prescribe the precision of private arithmetic.
using gdn_ref::kOutputCriterion;
using gdn_ref::kStateCriterion;

enum class InputPattern {
    Ordinary,
    WeakDecay,
    ZeroDecay,
    SmallBeta,
    StrongDecay,
    NearCollinear,
    ZeroState,
    ZeroQuery,
    ZeroSignal,
    ZeroBeta,
    UnitBeta,
};

struct Case {
    const char* name;
    int qk_heads;
    int value_heads;
    int tokens;
    bool normalize_qk;
    bool near_zero_qk = false;
};

void fill_uniform(std::vector<float>& values, std::mt19937& generator, float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    for (float& value : values) { value = distribution(generator); }
}

void normalize_rows(std::vector<float>& values, int width) {
    const std::size_t rows = values.size() / static_cast<std::size_t>(width);
    for (std::size_t row = 0; row < rows; ++row) {
        float* base  = values.data() + row * static_cast<std::size_t>(width);
        double sumsq = 0.0;
        for (int d = 0; d < width; ++d) {
            const double value = static_cast<double>(base[d]);
            sumsq += value * value;
        }
        const double inv = 1.0 / std::sqrt(sumsq);
        for (int d = 0; d < width; ++d) {
            base[d] = static_cast<float>(static_cast<double>(base[d]) * inv);
        }
    }
}

gdn_ref::Inputs make_inputs(const Case& test_case, std::uint32_t seed,
                            InputPattern pattern = InputPattern::Ordinary) {
    gdn_ref::Inputs in;
    in.head_dim    = kStateDim;
    in.qk_heads    = test_case.qk_heads;
    in.value_heads = test_case.value_heads;
    in.tokens      = test_case.tokens;

    const std::size_t qk_size =
        static_cast<std::size_t>(kStateDim * test_case.qk_heads * test_case.tokens);
    const std::size_t value_size =
        static_cast<std::size_t>(kStateDim * test_case.value_heads * test_case.tokens);
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim * kStateDim * test_case.value_heads);
    in.q.resize(qk_size);
    in.k.resize(qk_size);
    in.v.resize(value_size);
    in.g.resize(static_cast<std::size_t>(test_case.value_heads * test_case.tokens));
    in.beta.resize(static_cast<std::size_t>(test_case.value_heads * test_case.tokens));
    in.state.resize(state_size);

    std::mt19937 generator(seed);
    fill_uniform(in.q, generator, -1.0f, 1.0f);
    fill_uniform(in.k, generator, -1.0f, 1.0f);
    fill_uniform(in.v, generator, -0.5f, 0.5f);
    fill_uniform(in.g, generator, -0.10f, -0.005f);
    fill_uniform(in.beta, generator, 0.05f, 0.95f);
    fill_uniform(in.state, generator, -0.02f, 0.02f);

    if (test_case.near_zero_qk) {
        for (float& value : in.q) { value *= 1.0e-4f; }
        for (float& value : in.k) { value *= 1.0e-4f; }
    } else if (!test_case.normalize_qk) {
        // Raw-Q/K mode still receives a stable, entirely valid public input. This host-side
        // generation choice is not part of the oracle.
        normalize_rows(in.q, kStateDim);
        normalize_rows(in.k, kStateDim);
    }

    round_to_bf16(in.q);
    round_to_bf16(in.k);
    round_to_bf16(in.v);
    switch (pattern) {
    case InputPattern::Ordinary:
        break;
    case InputPattern::WeakDecay:
        std::fill(in.g.begin(), in.g.end(), -1.0e-6F);
        break;
    case InputPattern::ZeroDecay:
        std::fill(in.g.begin(), in.g.end(), 0.0F);
        break;
    case InputPattern::SmallBeta:
        std::fill(in.g.begin(), in.g.end(), 0.0F);
        std::fill(in.beta.begin(), in.beta.end(), 0.002F);
        break;
    case InputPattern::StrongDecay:
        std::fill(in.g.begin(), in.g.end(), -80.0F);
        break;
    case InputPattern::NearCollinear: {
        std::fill(in.g.begin(), in.g.end(), 0.0F);
        std::fill(in.beta.begin(), in.beta.end(), 0.98F);
        for (float& value : in.state) value *= 50.0F;
        const std::vector<float> base(in.k.begin(), in.k.begin() + test_case.qk_heads * kStateDim);
        for (int t = 0; t < test_case.tokens; ++t) {
            for (int qh = 0; qh < test_case.qk_heads; ++qh) {
                for (int d = 0; d < kStateDim; ++d) {
                    const auto offset = (t * test_case.qk_heads + qh) * kStateDim + d;
                    in.k[offset]      = base[qh * kStateDim + d] + 0.002F * in.k[offset];
                }
            }
        }
        round_to_bf16(in.k);
        // Make V close to the prediction from the initial state. This stresses cancellation
        // in the update; it does not change the independent mathematical oracle.
        for (int t = 0; t < test_case.tokens; ++t) {
            for (int h = 0; h < test_case.value_heads; ++h) {
                const int qh      = h / (test_case.value_heads / test_case.qk_heads);
                const auto offset = (t * test_case.qk_heads + qh) * kStateDim;
                double sum        = 1.0e-6;
                for (int d = 0; d < kStateDim; ++d)
                    sum += double(in.k[offset + d]) * in.k[offset + d];
                const double inv = test_case.normalize_qk ? 1.0 / std::sqrt(sum) : 1.0;
                for (int row = 0; row < kStateDim; ++row) {
                    double prediction = 0.0;
                    for (int d = 0; d < kStateDim; ++d)
                        prediction += double(in.state[(h * kStateDim + row) * kStateDim + d]) *
                                      in.k[offset + d] * inv;
                    in.v[(t * test_case.value_heads + h) * kStateDim + row] =
                        static_cast<float>(prediction);
                }
            }
        }
        round_to_bf16(in.v);
        break;
    }
    case InputPattern::ZeroState:
        std::fill(in.state.begin(), in.state.end(), 0.0F);
        break;
    case InputPattern::ZeroQuery:
        std::fill(in.q.begin(), in.q.end(), 0.0F);
        break;
    case InputPattern::ZeroSignal:
        std::fill(in.state.begin(), in.state.end(), 0.0F);
        std::fill(in.v.begin(), in.v.end(), 0.0F);
        break;
    case InputPattern::ZeroBeta:
        std::fill(in.g.begin(), in.g.end(), 0.0F);
        std::fill(in.beta.begin(), in.beta.end(), 0.0F);
        break;
    case InputPattern::UnitBeta:
        std::fill(in.g.begin(), in.g.end(), 0.0F);
        std::fill(in.beta.begin(), in.beta.end(), 1.0F);
        break;
    }
    return in;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> doubles(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

template <typename T>
int verify_exact(const std::string& label, const std::vector<T>& got,
                 const std::vector<T>& expected) {
    return ninfer::test::verify_exact(label.c_str(), got, expected);
}

int verify_heads(const std::string& label, std::span<const double> got,
                 std::span<const double> expected, int value_heads, int tokens, bool state,
                 DType state_dtype = DType::FP32) {
    const auto& criterion = state ? gdn_ref::state_criterion(state_dtype) : kOutputCriterion;
    const std::size_t head_size =
        static_cast<std::size_t>(kStateDim) * (state ? kStateDim : tokens);
    if (got.size() != head_size * value_heads || got.size() != expected.size()) {
        throw std::logic_error("GDN per-head comparison shape mismatch");
    }
    std::vector<double> head_got(head_size), head_ref(head_size);
    int failures = 0;
    for (int h = 0; h < value_heads; ++h) {
        if (state) {
            std::copy_n(got.data() + h * head_size, head_size, head_got.data());
            std::copy_n(expected.data() + h * head_size, head_size, head_ref.data());
        } else {
            for (int t = 0; t < tokens; ++t) {
                const auto source = (static_cast<std::size_t>(t) * value_heads + h) * kStateDim;
                std::copy_n(got.data() + source, kStateDim, head_got.data() + t * kStateDim);
                std::copy_n(expected.data() + source, kStateDim, head_ref.data() + t * kStateDim);
            }
        }
        const std::string head_label = label + " head=" + std::to_string(h);
        const auto stats = compute_reduction_stats(head_got.data(), head_ref.data(), head_size);
        const double gross_limit = gross_error_limit(stats, criterion);
        report_reduction_stats(head_label, head_size, stats, criterion);
        if (!reduction_passes(stats, head_size, criterion)) {
            std::cerr << head_label << ": rel_l2=" << stats.relative_l2
                      << " limit=" << criterion.relative_l2
                      << " max_abs=" << stats.maximum_absolute_error
                      << " gross_limit=" << gross_limit << " non_finite=" << stats.first_non_finite
                      << '\n';
            ++failures;
        }
    }
    return failures;
}

std::vector<double> read_f32(const void* device, std::size_t count) {
    return doubles(from_device<float>(device, count));
}

void upload_state(GuardedDeviceBuffer& buffer, const std::vector<float>& values,
                  DType state_dtype) {
    if (state_dtype == DType::BF16) {
        const auto bits = bf16_bits(values);
        buffer.copy_from_host(bits.data(), buffer.bytes());
    } else {
        buffer.copy_from_host(values.data(), buffer.bytes());
    }
}

std::vector<double> read_state(const void* device, std::size_t count, DType state_dtype) {
    return state_dtype == DType::BF16 ? from_device_bf16(device, count) : read_f32(device, count);
}

std::vector<std::byte> read_bytes(const void* device, std::size_t bytes) {
    return from_device<std::byte>(device, bytes);
}

int verify_common_inputs_unchanged(const std::string& label, const gdn_ref::Inputs& in,
                                   const DeviceBuffer& q, const DeviceBuffer& k,
                                   const DeviceBuffer& v, const DeviceBuffer& g,
                                   const DeviceBuffer& beta) {
    int failures = 0;
    failures += verify_exact(label + " q unchanged", from_device<std::uint16_t>(q, in.q.size()),
                             bf16_bits(in.q));
    failures += verify_exact(label + " k unchanged", from_device<std::uint16_t>(k, in.k.size()),
                             bf16_bits(in.k));
    failures += verify_exact(label + " v unchanged", from_device<std::uint16_t>(v, in.v.size()),
                             bf16_bits(in.v));
    failures += verify_exact(label + " g unchanged", from_device<float>(g, in.g.size()), in.g);
    failures +=
        verify_exact(label + " beta unchanged", from_device<float>(beta, in.beta.size()), in.beta);
    return failures;
}

struct DeviceInputs {
    explicit DeviceInputs(const gdn_ref::Inputs& in)
        : q(to_device_bf16(in.q)), k(to_device_bf16(in.k)), v(to_device_bf16(in.v)),
          g(to_device_f32(in.g)), beta(to_device_f32(in.beta)) {}

    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer g;
    DeviceBuffer beta;
};

gdn_ref::Result reference_prefix(const gdn_ref::Inputs& in, int tokens, float scale,
                                 bool normalize_qk) {
    if (tokens == in.tokens) return gdn_ref::evaluate(in, scale, normalize_qk);
    gdn_ref::Inputs prefix;
    prefix.head_dim    = in.head_dim;
    prefix.qk_heads    = in.qk_heads;
    prefix.value_heads = in.value_heads;
    prefix.tokens      = tokens;
    prefix.q.assign(in.q.begin(), in.q.begin() + tokens * in.qk_heads * kStateDim);
    prefix.k.assign(in.k.begin(), in.k.begin() + tokens * in.qk_heads * kStateDim);
    prefix.v.assign(in.v.begin(), in.v.begin() + tokens * in.value_heads * kStateDim);
    prefix.g.assign(in.g.begin(), in.g.begin() + tokens * in.value_heads);
    prefix.beta.assign(in.beta.begin(), in.beta.begin() + tokens * in.value_heads);
    prefix.state = in.state;
    // Re-evaluate a few logical prefixes from the original public state. Neither GPU state nor
    // a cast of an earlier FP64 result is fed back into the oracle at a call boundary.
    return gdn_ref::evaluate(prefix, scale, normalize_qk);
}

int run_case(const Case& test_case, std::uint32_t seed,
             InputPattern pattern = InputPattern::Ordinary, bool inplace = false,
             std::vector<int> partitions = {}, DType state_dtype = DType::FP32) {
    gdn_ref::Inputs in = make_inputs(test_case, seed, pattern);
    if (state_dtype == DType::BF16) round_to_bf16(in.state);
    if (partitions.empty()) partitions.push_back(test_case.tokens);
    if (std::accumulate(partitions.begin(), partitions.end(), 0) != test_case.tokens ||
        *std::min_element(partitions.begin(), partitions.end()) <= 0) {
        throw std::logic_error("invalid GDN continuation partition");
    }
    const std::string label =
        std::string(test_case.name) + " H=" + std::to_string(test_case.qk_heads) + "/" +
        std::to_string(test_case.value_heads) + " T=" + std::to_string(test_case.tokens) +
        (test_case.normalize_qk ? " normalized" : " raw") +
        (inplace ? " inplace" : " distinct-state") +
        (state_dtype == DType::BF16 ? " bf16-state" : " fp32-state");
    std::cout << "GDN_CASE " << label << " calls=" << partitions.size() << std::endl;
    const float scale = 1.0F / std::sqrt(static_cast<float>(kStateDim));
    DeviceInputs device(in);
    const std::size_t state_bytes = in.state.size() * dtype_size(state_dtype);
    GuardedDeviceBuffer state_a(state_bytes);
    GuardedDeviceBuffer state_b(state_bytes);
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    upload_state(state_a, in.state, state_dtype);
    out.fill(0xff);
    Tensor q(device.q.span(), DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.span(), DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.span(), DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.span(), DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.span(), DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor output(out.span(), DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});

    const int maximum_call     = *std::max_element(partitions.begin(), partitions.end());
    const auto workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, 1, maximum_call);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(workspace_bytes, 256));
    scratch.fill(0xff);
    MemoryArena workspace(scratch.span());
    const auto device_execution = cuda_environment().default_execution;

    GuardedDeviceBuffer* current = &state_a;
    int failures                 = 0;
    int begin                    = 0;
    for (int length : partitions) {
        auto* destination = inplace ? current : (current == &state_a ? &state_b : &state_a);
        std::vector<std::byte> source_before;
        if (!inplace) {
            source_before = read_bytes(current->data(), state_bytes);
            destination->fill(0xff);
        }
        Tensor input_state(current->span(), state_dtype,
                           {kStateDim, kStateDim, test_case.value_heads});
        Tensor output_state(destination->span(), state_dtype,
                            {kStateDim, kStateDim, test_case.value_heads});
        Tensor qs = q.slice(2, begin, length), ks = k.slice(2, begin, length);
        Tensor vs = v.slice(2, begin, length), gs = g.slice(1, begin, length);
        Tensor bs = beta.slice(1, begin, length), os = output.slice(2, begin, length);
        if (inplace) {
            ops::gated_delta_net(qs, ks, vs, gs, bs, scale, test_case.normalize_qk, workspace,
                                 output_state, os, device_execution);
        } else {
            ops::gated_delta_net(qs, ks, vs, gs, bs, scale, test_case.normalize_qk, workspace,
                                 input_state, output_state, os, device_execution);
        }
        cuda_synchronize();
        begin += length;
        const std::string checkpoint = label + " prefix=" + std::to_string(begin);
        const auto ref               = reference_prefix(in, begin, scale, test_case.normalize_qk);
        const auto output_count =
            static_cast<std::size_t>(begin) * test_case.value_heads * kStateDim;
        const auto got_output = from_device_bf16(out.data(), output_count);
        const auto got_state  = read_state(destination->data(), in.state.size(), state_dtype);
        failures += verify_heads(checkpoint + " out", got_output, ref.out, test_case.value_heads,
                                 begin, false);
        if (partitions.size() > 1) {
            const auto offset =
                static_cast<std::size_t>(begin - length) * test_case.value_heads * kStateDim;
            failures += verify_heads(checkpoint + " current out",
                                     std::span<const double>(got_output).subspan(offset),
                                     std::span<const double>(ref.out).subspan(offset),
                                     test_case.value_heads, length, false);
        }
        failures += verify_heads(checkpoint + " state", got_state, ref.final_state,
                                 test_case.value_heads, 1, true, state_dtype);
        if (pattern == InputPattern::ZeroQuery || pattern == InputPattern::ZeroSignal) {
            failures += verify_exact(checkpoint + " exact zero output", got_output,
                                     std::vector<double>(output_count, 0.0));
        }
        if (pattern == InputPattern::ZeroSignal) {
            failures += verify_exact(checkpoint + " exact zero state", got_state,
                                     std::vector<double>(in.state.size(), 0.0));
        }
        if (pattern == InputPattern::ZeroBeta) {
            failures += verify_exact(checkpoint + " identity state", got_state, doubles(in.state));
        }
        if (!inplace) {
            failures += verify_exact(checkpoint + " source state unchanged",
                                     read_bytes(current->data(), state_bytes), source_before);
        }
        if (output_count < in.v.size()) {
            const auto* suffix = static_cast<const std::uint16_t*>(out.data()) + output_count;
            failures +=
                verify_exact(checkpoint + " unconsumed output unchanged",
                             from_device<std::uint16_t>(suffix, in.v.size() - output_count),
                             std::vector<std::uint16_t>(in.v.size() - output_count, 0xffffU));
        }
        failures += state_a.verify_guards(checkpoint + " state-a");
        failures += state_b.verify_guards(checkpoint + " state-b");
        failures += out.verify_guards(checkpoint + " out");
        failures += scratch.verify_guards(checkpoint + " workspace");
        if (workspace.used() != 0) {
            std::cerr << checkpoint << ": workspace was not released\n";
            ++failures;
        }
        current = destination;
    }
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    if (workspace.peak_used() > workspace_bytes) {
        std::cerr << label << ": workspace exceeded its interval capacity\n";
        ++failures;
    }
    return failures;
}

int batch_update_case(const Case& test_case, const std::vector<int>& source_slots,
                      const std::vector<int>& destination_slots, int slots, std::uint32_t seed,
                      std::vector<int> valid = {}, bool captured = false,
                      DType state_dtype = DType::FP32) {
    if (test_case.tokens != 1) { throw std::logic_error("batch_update_case requires W=1"); }
    if (source_slots.size() != destination_slots.size()) {
        throw std::logic_error("batch_update_case selector sizes differ");
    }
    const int batch  = static_cast<int>(source_slots.size());
    const int width  = 1;
    const bool dense = valid.empty();
    if (dense) valid.assign(batch, 1);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const std::size_t qk_row_size =
        static_cast<std::size_t>(kStateDim * test_case.qk_heads * width);
    const std::size_t value_row_size =
        static_cast<std::size_t>(kStateDim * test_case.value_heads * width);
    const std::size_t gate_row_size = static_cast<std::size_t>(test_case.value_heads * width);
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim * kStateDim * test_case.value_heads);

    gdn_ref::Inputs aggregate;
    aggregate.head_dim    = kStateDim;
    aggregate.qk_heads    = test_case.qk_heads;
    aggregate.value_heads = test_case.value_heads;
    aggregate.tokens      = static_cast<std::int64_t>(width) * batch;
    aggregate.q.reserve(qk_row_size * static_cast<std::size_t>(batch));
    aggregate.k.reserve(qk_row_size * static_cast<std::size_t>(batch));
    aggregate.v.reserve(value_row_size * static_cast<std::size_t>(batch));
    aggregate.g.reserve(gate_row_size * static_cast<std::size_t>(batch));
    aggregate.beta.reserve(gate_row_size * static_cast<std::size_t>(batch));

    std::vector<gdn_ref::Inputs> rows;
    rows.reserve(static_cast<std::size_t>(batch));
    std::vector<float> initial_states(state_size * static_cast<std::size_t>(slots), 0.125f);
    for (int row = 0; row < batch; ++row) {
        gdn_ref::Inputs input =
            make_inputs(test_case, seed + static_cast<std::uint32_t>(row) * 97U);
        aggregate.q.insert(aggregate.q.end(), input.q.begin(), input.q.end());
        aggregate.k.insert(aggregate.k.end(), input.k.begin(), input.k.end());
        aggregate.v.insert(aggregate.v.end(), input.v.begin(), input.v.end());
        aggregate.g.insert(aggregate.g.end(), input.g.begin(), input.g.end());
        aggregate.beta.insert(aggregate.beta.end(), input.beta.begin(), input.beta.end());
        std::copy(input.state.begin(), input.state.end(),
                  initial_states.begin() +
                      static_cast<std::size_t>(source_slots[static_cast<std::size_t>(row)]) *
                          state_size);
        rows.push_back(std::move(input));
    }
    if (state_dtype == DType::BF16) round_to_bf16(initial_states);

    // Repeated source selectors share the final represented pool contents, independent of row
    // inputs.
    for (int row = 0; row < batch; ++row) {
        const auto begin = initial_states.begin() + std::size_t(source_slots[row]) * state_size;
        rows[row].state.assign(begin, begin + state_size);
    }
    std::vector<gdn_ref::Result> references;
    references.reserve(static_cast<std::size_t>(batch));
    std::vector<bool> written_slots(static_cast<std::size_t>(slots), false);
    for (int row = 0; row < batch; ++row) {
        if (valid[row] == 0) {
            references.emplace_back();
            continue;
        }
        gdn_ref::Result reference =
            gdn_ref::evaluate(rows[static_cast<std::size_t>(row)], static_cast<double>(scale),
                              test_case.normalize_qk);
        written_slots[static_cast<std::size_t>(destination_slots[static_cast<std::size_t>(row)])] =
            true;
        references.push_back(std::move(reference));
    }

    DeviceInputs device(aggregate);
    const std::size_t slot_bytes = state_size * dtype_size(state_dtype);
    GuardedDeviceBuffer states(initial_states.size() * dtype_size(state_dtype));
    GuardedDeviceBuffer out(aggregate.v.size() * sizeof(std::uint16_t));
    upload_state(states, initial_states, state_dtype);
    const auto initial_state_bytes = read_bytes(states.data(), states.bytes());
    out.fill(0xff);

    Tensor q(device.q.span(), DType::BF16, {kStateDim, test_case.qk_heads, width, batch});
    Tensor k(device.k.span(), DType::BF16, {kStateDim, test_case.qk_heads, width, batch});
    Tensor v(device.v.span(), DType::BF16, {kStateDim, test_case.value_heads, width, batch});
    Tensor g(device.g.span(), DType::FP32, {test_case.value_heads, width, batch});
    Tensor beta(device.beta.span(), DType::FP32, {test_case.value_heads, width, batch});
    Tensor states_tensor(states.span(), state_dtype,
                         {kStateDim, kStateDim, test_case.value_heads, slots});
    RecordStateFixture history(states_tensor, 3);
    auto source_offsets      = history.record_offsets(source_slots);
    auto destination_offsets = history.record_offsets(destination_slots);
    for (int row = 0; row < batch; ++row)
        if (valid[row] == 0) source_offsets[row] = destination_offsets[row] = -1;
    DeviceBuffer device_valid;
    Tensor valid_tensor;
    if (!dense) {
        device_valid = to_device(valid);
        valid_tensor = Tensor(device_valid.span(), DType::I32, {batch});
    }
    DeviceBuffer device_source_slots      = to_device(source_offsets);
    DeviceBuffer device_destination_slots = to_device(destination_offsets);
    Tensor source_slots_tensor(device_source_slots.span(), DType::I64, {batch});
    Tensor destination_slots_tensor(device_destination_slots.span(), DType::I64, {batch});
    Tensor out_tensor(out.span(), DType::BF16, {kStateDim, test_case.value_heads, width, batch});
    const auto execution = cuda_environment().context.execution_view();
    const auto launch    = [&] {
        ops::gated_delta_net_batch_update(
            q, k, v, g, beta, scale, test_case.normalize_qk, history.view(source_slots_tensor),
            history.view(destination_slots_tensor), valid_tensor, out_tensor, execution);
    };
    if (captured) {
        device_valid.fill(0);
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        launch();
        cuda_synchronize();
        definition.capture(execution.stream(), launch);
        graph.instantiate(definition);
        graph.launch(execution.stream());
        cuda_synchronize();
        if (from_device<std::uint16_t>(out.data(), aggregate.v.size()) !=
            std::vector<std::uint16_t>(aggregate.v.size(), 0))
            throw std::runtime_error("inactive graph update did not zero output");
        device_valid.copy_from_host(valid.data(), device_valid.size());
        graph.launch(execution.stream());
    } else {
        launch();
    }
    cuda_synchronize();

    const std::string label = std::string(test_case.name) + " batch update B=" +
                              std::to_string(batch) +
                              (state_dtype == DType::BF16 ? " bf16-state" : " fp32-state");
    int failures                         = 0;
    const std::vector<double> got_output = from_device_bf16(out.data(), aggregate.v.size());
    history.copy_back();
    failures += history.verify_guards("recurrent state");
    const auto got_states      = read_state(states.data(), initial_states.size(), state_dtype);
    const auto got_state_bytes = read_bytes(states.data(), states.bytes());
    for (int row = 0; row < batch; ++row) {
        const auto row_label = label + " row=" + std::to_string(row);
        if (valid[row] == 0) {
            failures +=
                verify_exact(row_label + " inactive output",
                             std::vector<double>(got_output.begin() + row * value_row_size,
                                                 got_output.begin() + (row + 1) * value_row_size),
                             std::vector<double>(value_row_size, 0.0));
            continue;
        }
        failures += verify_heads(
            row_label + " out",
            std::span<const double>(got_output).subspan(row * value_row_size, value_row_size),
            references[static_cast<std::size_t>(row)].out, test_case.value_heads, 1, false);
        const std::size_t begin =
            static_cast<std::size_t>(destination_slots[static_cast<std::size_t>(row)]) * state_size;
        failures += verify_heads(
            row_label + " state", std::span<const double>(got_states).subspan(begin, state_size),
            references[static_cast<std::size_t>(row)].final_state, test_case.value_heads, 1, true,
            state_dtype);
    }
    for (int slot = 0; slot < slots; ++slot) {
        if (written_slots[static_cast<std::size_t>(slot)]) continue;
        const std::size_t begin = static_cast<std::size_t>(slot) * slot_bytes;
        failures += verify_exact(
            label + " untouched slot " + std::to_string(slot),
            std::vector<std::byte>(got_state_bytes.begin() + begin,
                                   got_state_bytes.begin() + begin + slot_bytes),
            std::vector<std::byte>(initial_state_bytes.begin() + begin,
                                   initial_state_bytes.begin() + begin + slot_bytes));
    }
    failures += verify_exact(label + " source selectors unchanged",
                             from_device<std::int64_t>(device_source_slots, source_offsets.size()),
                             source_offsets);
    failures += verify_exact(
        label + " destination selectors unchanged",
        from_device<std::int64_t>(device_destination_slots, destination_offsets.size()),
        destination_offsets);
    failures += states.verify_guards((label + " states").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, aggregate, device.q, device.k, device.v,
                                               device.g, device.beta);
    return failures;
}

int graph_case(Case c, int batch, bool pooled, DType state_dtype = DType::FP32) {
    const int slots              = 2 * batch;
    const std::size_t state_size = std::size_t(kStateDim) * kStateDim * c.value_heads;
    const std::size_t slot_bytes = state_size * dtype_size(state_dtype);
    const std::size_t qk_size    = std::size_t(kStateDim) * c.qk_heads * c.tokens;
    const std::size_t value_size = std::size_t(kStateDim) * c.value_heads * c.tokens;
    const std::size_t gate_size  = std::size_t(c.value_heads) * c.tokens;
    Case aggregate_case          = c;
    aggregate_case.tokens *= batch;
    auto in = make_inputs(aggregate_case, 15100U);
    DeviceInputs device(in);
    GuardedDeviceBuffer state(state_size * slots * dtype_size(state_dtype));
    GuardedDeviceBuffer output(value_size * batch * sizeof(std::uint16_t));
    std::vector<std::int32_t> source(batch), destination(batch);
    DeviceBuffer source_device      = to_device(std::vector<std::int64_t>(batch)),
                 destination_device = to_device(std::vector<std::int64_t>(batch));
    Tensor q(device.q.span(), DType::BF16, {kStateDim, c.qk_heads, c.tokens, batch});
    Tensor k(device.k.span(), DType::BF16, {kStateDim, c.qk_heads, c.tokens, batch});
    Tensor v(device.v.span(), DType::BF16, {kStateDim, c.value_heads, c.tokens, batch});
    Tensor g(device.g.span(), DType::FP32, {c.value_heads, c.tokens, batch});
    Tensor beta(device.beta.span(), DType::FP32, {c.value_heads, c.tokens, batch});
    Tensor out(output.span(), DType::BF16, {kStateDim, c.value_heads, c.tokens, batch});
    Tensor states(state.span(), state_dtype, {kStateDim, kStateDim, c.value_heads, slots});
    RecordStateFixture history(states, 3);
    Tensor sources(source_device.span(), DType::I64, {batch});
    Tensor destinations(destination_device.span(), DType::I64, {batch});
    const auto capacity = pooled ? 0
                                 : ops::gated_delta_net_workspace_capacity_bytes(
                                       c.qk_heads, c.value_heads, c.tokens, c.tokens);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 256));
    MemoryArena workspace(scratch.span());
    auto& context        = cuda_environment().context;
    const auto execution = context.execution_view();
    const float scale    = 1.0F / std::sqrt(float(kStateDim));
    const auto launch    = [&] {
        if (pooled) {
            ops::gated_delta_net_batch_update(q, k, v, g, beta, scale, c.normalize_qk,
                                              history.view(sources), history.view(destinations), {},
                                              out, execution);
        } else {
            auto si = history.tensor(0);
            auto so = history.tensor(1);
            ops::gated_delta_net(q, k, v, g, beta, scale, c.normalize_qk, workspace, si, so, out,
                                    execution);
        }
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    int failures = 0;
    for (int replay = 0; replay < 2; ++replay) {
        in                     = make_inputs(aggregate_case, 15100U + 97U * replay);
        const auto upload_bf16 = [](DeviceBuffer& buffer, const std::vector<float>& values) {
            const auto bits = bf16_bits(values);
            buffer.copy_from_host(bits.data(), buffer.size());
        };
        upload_bf16(device.q, in.q);
        upload_bf16(device.k, in.k);
        upload_bf16(device.v, in.v);
        device.g.copy_from_host(in.g.data(), device.g.size());
        device.beta.copy_from_host(in.beta.data(), device.beta.size());
        std::vector<float> initial(state_size * slots);
        for (int slot = 0; slot < slots; ++slot)
            for (std::size_t i = 0; i < state_size; ++i)
                initial[slot * state_size + i] = in.state[i] * (1.0F + 0.25F * slot);
        if (state_dtype == DType::BF16) round_to_bf16(initial);
        for (int row = 0; row < batch; ++row) {
            source[row]      = pooled ? 2 * row + replay : 0;
            destination[row] = pooled ? 2 * row : 1;
        }
        upload_state(state, initial, state_dtype);
        const auto expected_state_bytes = read_bytes(state.data(), state.bytes());
        history.upload();
        const auto source_offsets      = history.record_offsets(source);
        const auto destination_offsets = history.record_offsets(destination);
        source_device.copy_from_host(source_offsets.data(), source_device.size());
        destination_device.copy_from_host(destination_offsets.data(), destination_device.size());
        output.fill(0xff);
        scratch.fill(0xff);
        cuda_synchronize();
        if (!graph.ready()) {
            launch(); // Initialize launch resources before capture, then restore the state.
            context.synchronize();
            history.upload();
            output.fill(0xff);
            scratch.fill(0xff);
            cuda_synchronize();
            definition.capture(context.stream(), launch);
            graph.instantiate(definition);
        }
        graph.launch(context.stream());
        context.synchronize();
        history.copy_back();
        const auto got_state       = read_state(state.data(), initial.size(), state_dtype);
        const auto got_state_bytes = read_bytes(state.data(), state.bytes());
        const auto got_output      = from_device_bf16(output.data(), value_size * batch);
        const std::string label    = std::string(c.name) + " replay=" + std::to_string(replay) +
                                  (state_dtype == DType::BF16 ? " bf16-state" : " fp32-state");
        for (int row = 0; row < batch; ++row) {
            gdn_ref::Inputs input;
            input.head_dim    = kStateDim;
            input.qk_heads    = c.qk_heads;
            input.value_heads = c.value_heads;
            input.tokens      = c.tokens;
            const auto slice  = [row](const std::vector<float>& values, std::size_t size) {
                return std::vector<float>(values.begin() + row * size,
                                           values.begin() + (row + 1) * size);
            };
            input.q    = slice(in.q, qk_size);
            input.k    = slice(in.k, qk_size);
            input.v    = slice(in.v, value_size);
            input.g    = slice(in.g, gate_size);
            input.beta = slice(in.beta, gate_size);
            input.state.assign(initial.begin() + source[row] * state_size,
                               initial.begin() + (source[row] + 1) * state_size);
            const auto reference = gdn_ref::evaluate(input, scale, c.normalize_qk);
            failures += verify_heads(label + " output row=" + std::to_string(row),
                                     std::span(got_output).subspan(row * value_size, value_size),
                                     reference.out, c.value_heads, c.tokens, false);
            failures += verify_heads(
                label + " state row=" + std::to_string(row),
                std::span<const double>(got_state).subspan(destination[row] * state_size,
                                                           state_size),
                reference.final_state, c.value_heads, 1, true, state_dtype);
        }
        for (int slot = 0; slot < slots; ++slot) {
            if (std::find(destination.begin(), destination.end(), slot) != destination.end())
                continue;
            failures += verify_exact(
                label + " untouched state",
                std::vector<std::byte>(got_state_bytes.begin() + slot * slot_bytes,
                                       got_state_bytes.begin() + (slot + 1) * slot_bytes),
                std::vector<std::byte>(expected_state_bytes.begin() + slot * slot_bytes,
                                       expected_state_bytes.begin() + (slot + 1) * slot_bytes));
        }
        failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v,
                                                   device.g, device.beta);
        failures += verify_exact(label + " sources",
                                 from_device<std::int64_t>(source_device, batch), source_offsets);
        failures +=
            verify_exact(label + " destinations",
                         from_device<std::int64_t>(destination_device, batch), destination_offsets);
        failures += history.verify_guards(label);
        failures += output.verify_guards(label);
        failures += scratch.verify_guards(label);
        if (workspace.used() != 0 || workspace.peak_used() > capacity) ++failures;
    }
    return failures;
}

int contract_rejection_cases() {
    DeviceBuffer q_buffer(ninfer::test::cuda_environment().context.device(),
                          kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer k_buffer(ninfer::test::cuda_environment().context.device(),
                          kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer v_buffer(ninfer::test::cuda_environment().context.device(),
                          kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer g_buffer(ninfer::test::cuda_environment().context.device(), 8 * sizeof(float));
    DeviceBuffer beta_buffer(ninfer::test::cuda_environment().context.device(), 8 * sizeof(float));
    DeviceBuffer state_buffer(ninfer::test::cuda_environment().context.device(),
                              kStateDim * kStateDim * 8 * sizeof(float));
    DeviceBuffer out_buffer(ninfer::test::cuda_environment().context.device(),
                            kStateDim * 8 * sizeof(std::uint16_t));
    ninfer::DeviceBuffer workspace_storage(ninfer::test::cuda_environment().context.device(), 256);
    MemoryArena workspace(workspace_storage.span());
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));

    auto is_rejected = [&](int activation_dim, int state_dim, int qk_heads, int value_heads) {
        Tensor q(q_buffer.span(), DType::BF16, {activation_dim, qk_heads, 1});
        Tensor k(k_buffer.span(), DType::BF16, {activation_dim, qk_heads, 1});
        Tensor v(v_buffer.span(), DType::BF16, {activation_dim, value_heads, 1});
        Tensor g(g_buffer.span(), DType::FP32, {value_heads, 1});
        Tensor beta(beta_buffer.span(), DType::FP32, {value_heads, 1});
        Tensor state(state_buffer.span(), DType::FP32, {state_dim, state_dim, value_heads});
        Tensor out(out_buffer.span(), DType::BF16, {activation_dim, value_heads, 1});
        try {
            ops::gated_delta_net(q, k, v, g, beta, scale, true, workspace, state, out,
                                 cuda_environment().default_execution);
        } catch (const std::invalid_argument&) { return true; }
        cuda_synchronize();
        return false;
    };

    int failures = 0;
    if (!is_rejected(64, kStateDim, 4, 8)) {
        std::cerr << "gated_delta_net accepted Q/K/V head dimension 64\n";
        ++failures;
    }
    if (!is_rejected(kStateDim, 64, 4, 8)) {
        std::cerr << "gated_delta_net accepted state dimension 64\n";
        ++failures;
    }
    if (!is_rejected(kStateDim, kStateDim, 4, 6)) {
        std::cerr << "gated_delta_net accepted a non-divisible head map\n";
        ++failures;
    }
    // Reject unsupported (FP16) and mismatched (FP32 vs BF16) recurrent state dtypes.
    auto check_state_dtype_rejection = [&](DType src_dtype, DType dst_dtype, const char* what) {
        Tensor q(q_buffer.span(), DType::BF16, {kStateDim, 4, 1});
        Tensor k(k_buffer.span(), DType::BF16, {kStateDim, 4, 1});
        Tensor v(v_buffer.span(), DType::BF16, {kStateDim, 8, 1});
        Tensor g(g_buffer.span(), DType::FP32, {8, 1});
        Tensor beta(beta_buffer.span(), DType::FP32, {8, 1});
        Tensor out(out_buffer.span(), DType::BF16, {kStateDim, 8, 1});
        Tensor src_state(state_buffer.span(), src_dtype, {kStateDim, kStateDim, 8});
        Tensor dst_state(state_buffer.span(), dst_dtype, {kStateDim, kStateDim, 8});
        bool rejected = false;
        try {
            if (src_dtype == dst_dtype) {
                ops::gated_delta_net(q, k, v, g, beta, scale, true, workspace, dst_state, out,
                                     cuda_environment().default_execution);
            } else {
                ops::gated_delta_net(q, k, v, g, beta, scale, true, workspace, src_state, dst_state,
                                     out, cuda_environment().default_execution);
            }
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) {
            cuda_synchronize();
            std::cerr << "gated_delta_net accepted " << what << '\n';
            ++failures;
        }
    };
    check_state_dtype_rejection(DType::FP16, DType::FP16, "FP16 state");
    check_state_dtype_rejection(DType::FP32, DType::BF16, "mismatched FP32/BF16 states");
    check_state_dtype_rejection(DType::BF16, DType::FP32, "mismatched BF16/FP32 states");
    {
        DeviceBuffer offset_buffer = to_device(std::vector<std::int64_t>{0});
        Tensor offsets(offset_buffer.span(), DType::I64, {1});
        Tensor q(q_buffer.span(), DType::BF16, {kStateDim, 4, 1, 1});
        Tensor k(k_buffer.span(), DType::BF16, {kStateDim, 4, 1, 1});
        Tensor v(v_buffer.span(), DType::BF16, {kStateDim, 8, 1, 1});
        Tensor g(g_buffer.span(), DType::FP32, {8, 1, 1});
        Tensor beta(beta_buffer.span(), DType::FP32, {8, 1, 1});
        Tensor out(out_buffer.span(), DType::BF16, {kStateDim, 8, 1, 1});
        auto make_field = [&](DType dt) {
            const std::size_t bytes = std::size_t(kStateDim) * kStateDim * 8 * dtype_size(dt);
            TensorRegion reg{{0, bytes, 16}, dt, {kStateDim, kStateDim, 8, 1}};
            return StateFieldView{{state_buffer.span(), offsets}, StateFieldLayout{reg, 1, 0}};
        };
        for (const auto [sd, dd, name] : {std::tuple{DType::FP16, DType::FP16, "batch FP16 state"},
                                          std::tuple{DType::FP32, DType::BF16, "batch mixed state"},
                                          std::tuple{DType::BF16, DType::FP32, "batch mixed state 2"}}) {
            bool rejected = false;
            try {
                ops::gated_delta_net_batch_update(q, k, v, g, beta, scale, true, make_field(sd),
                                                  make_field(dd), {}, out,
                                                  cuda_environment().default_execution);
            } catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) {
                cuda_synchronize();
                std::cerr << "gated_delta_net_batch_update accepted " << name << '\n';
                ++failures;
            }
        }
        RaggedFixture ragged({1});
        Tensor rq(q_buffer.span(), DType::BF16, {kStateDim, 4, 1});
        Tensor rk(k_buffer.span(), DType::BF16, {kStateDim, 4, 1});
        Tensor rv(v_buffer.span(), DType::BF16, {kStateDim, 8, 1});
        Tensor rg(g_buffer.span(), DType::FP32, {8, 1});
        Tensor rbeta(beta_buffer.span(), DType::FP32, {8, 1});
        Tensor rout(out_buffer.span(), DType::BF16, {kStateDim, 8, 1});
        for (const auto [sd, dd, name] :
             {std::tuple{DType::FP16, DType::FP16, "ragged FP16 state"},
              std::tuple{DType::FP32, DType::BF16, "ragged mixed state"}}) {
            bool rejected = false;
            try {
                ops::gated_delta_net(rq, rk, rv, rg, rbeta, scale, true, ragged.view, workspace,
                                     make_field(sd), make_field(dd), rout,
                                     cuda_environment().default_execution);
            } catch (const std::invalid_argument&) { rejected = true; }
            if (!rejected) {
                cuda_synchronize();
                std::cerr << "ragged gated_delta_net accepted " << name << '\n';
                ++failures;
            }
        }
    }
    return failures;
}

enum class RaggedStateMode { InPlace, Mixed, Permuted, SharedSource };

int ragged_recurrence(Case c, std::vector<int> lengths, RaggedStateMode mode, bool captured = false,
                      bool adversarial = false, bool poison = false, bool capacity_witness = false,
                      DType state_dtype = DType::FP32) {
    RaggedFixture ragged(std::move(lengths));
    const int rows = ragged.segments(), heads = c.value_heads;
    const int slots                  = 2 * rows + 1;
    c.tokens                         = ragged.tokens();
    const std::size_t state_elements = std::size_t(kStateDim) * kStateDim * heads;
    const std::size_t slot_bytes     = state_elements * dtype_size(state_dtype);
    const std::size_t out_column     = kStateDim * heads;
    const float scale                = 1.0F / std::sqrt(float(kStateDim));
    const auto make_packed_inputs    = [&](unsigned seed) {
        auto packed = make_inputs(c, seed);
        for (int row = 1; adversarial && row < rows; ++row) {
            const int token    = ragged.prefix[row];
            const auto amplify = [token](std::vector<float>& values, int width, float factor) {
                auto first = values.begin() + token * width;
                for (auto it = first; it != first + width; ++it) *it *= factor;
            };
            amplify(packed.q, kStateDim * c.qk_heads, 32.0F);
            amplify(packed.k, kStateDim * c.qk_heads, c.normalize_qk ? 32.0F : 2.0F);
            amplify(packed.v, kStateDim * heads, 32.0F);
            std::fill_n(packed.g.begin() + token * heads, heads, -1.5F);
            std::fill_n(packed.beta.begin() + token * heads, heads, 0.20F);
        }
        if (poison) {
            // A finite nonzero V in padding can be hidden by the zero solver column. An infinity
            // in the neighboring segment exposes a missed zfill through the matrix product.
            std::fill_n(packed.v.begin() + ragged.prefix[1] * out_column, out_column,
                           std::numeric_limits<float>::infinity());
        }
        return packed;
    };
    auto packed = make_packed_inputs(937U);

    std::vector<float> initial(state_elements * slots);
    for (int slot = 0; slot < slots; ++slot) {
        const float factor = slot % 2 == 0 ? 1.0F : -1.0F;
        const auto first   = initial.begin() + slot * state_elements;
        for (std::size_t i = 0; i < state_elements; ++i)
            first[i] = factor * packed.state[(i + slot * 97U) % state_elements];
    }
    if (state_dtype == DType::BF16) round_to_bf16(initial);
    DeviceInputs device(packed);
    GuardedDeviceBuffer states(initial.size() * dtype_size(state_dtype));
    GuardedDeviceBuffer output(packed.v.size() * sizeof(std::uint16_t));
    upload_state(states, initial, state_dtype);
    const auto initial_state_bytes = read_bytes(states.data(), states.bytes());
    RecordStateFixture history(Tensor(states.span(), state_dtype, {128, 128, heads, slots}), 3);
    auto device_source      = to_device(std::vector<std::int64_t>(rows));
    auto device_destination = to_device(std::vector<std::int64_t>(rows));
    const auto src          = history.view(Tensor(device_source.span(), DType::I64, {rows}));
    const auto dst          = history.view(Tensor(device_destination.span(), DType::I64, {rows}));
    Tensor q(device.q.span(), DType::BF16, {128, c.qk_heads, c.tokens});
    Tensor k(device.k.span(), DType::BF16, {128, c.qk_heads, c.tokens});
    Tensor v(device.v.span(), DType::BF16, {128, heads, c.tokens});
    Tensor g(device.g.span(), DType::FP32, {heads, c.tokens});
    Tensor beta(device.beta.span(), DType::FP32, {heads, c.tokens});
    Tensor out(output.span(), DType::BF16, {128, heads, c.tokens});
    const auto bytes = ops::gated_delta_net_ragged_workspace_capacity_bytes(
        c.qk_heads, heads, c.tokens, c.tokens, rows, rows);
    GuardedDeviceBuffer scratch(bytes);
    MemoryArena ws(scratch.span());
    auto& context        = cuda_environment().context;
    const auto execution = context.execution_view();
    const auto launch    = [&] {
        ops::gated_delta_net(q, k, v, g, beta, scale, c.normalize_qk, ragged.view, ws, src, dst,
                                out, execution);
    };

    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    int failures = 0;
    for (int replay = 0; replay <= static_cast<int>(captured); ++replay) {
        std::vector<int> source(rows), destination(rows);
        std::vector<bool> written(slots);
        for (int row = 0; row < rows; ++row) {
            source[row]      = 2 * row + replay;
            destination[row] = source[row];
            switch (mode) {
            case RaggedStateMode::InPlace:
                break;
            case RaggedStateMode::Mixed:
                if (row % 2) destination[row] ^= 1;
                break;
            case RaggedStateMode::Permuted:
                destination[row] = 2 * (rows - 1 - row) + (1 - replay);
                break;
            case RaggedStateMode::SharedSource:
                source[row]      = 2 * rows - replay;
                destination[row] = row;
                break;
            }
            written[destination[row]] = true;
        }
        const auto source_offsets      = history.record_offsets(source);
        const auto destination_offsets = history.record_offsets(destination);
        device_source.copy_from_host(source_offsets.data(), device_source.size());
        device_destination.copy_from_host(destination_offsets.data(), device_destination.size());
        output.fill(0xff);
        scratch.fill(0xff);
        if (replay > 0) {
            packed                 = make_packed_inputs(1147U);
            const auto upload_bf16 = [](DeviceBuffer& buffer, const std::vector<float>& values) {
                const auto bits = bf16_bits(values);
                buffer.copy_from_host(bits.data(), buffer.size());
            };
            upload_bf16(device.q, packed.q);
            upload_bf16(device.k, packed.k);
            upload_bf16(device.v, packed.v);
            device.g.copy_from_host(packed.g.data(), device.g.size());
            device.beta.copy_from_host(packed.beta.data(), device.beta.size());
            upload_state(states, initial, state_dtype);
            history.upload();
        }
        if (captured) {
            if (!graph.ready()) {
                definition.capture(execution.stream(), launch);
                graph.instantiate(definition);
            }
            graph.launch(execution.stream());
        } else {
            launch();
        }
        context.synchronize();
        history.copy_back();
        const auto actual      = from_device_bf16(output.data(), packed.v.size());
        const auto after       = read_state(states.data(), initial.size(), state_dtype);
        const auto after_bytes = read_bytes(states.data(), states.bytes());
        for (int row = 0; row < rows; ++row) {
            if (poison && row == 1) continue;
            const auto segment = [&](const std::vector<float>& values, std::size_t width) {
                return std::vector<float>(values.begin() + ragged.prefix[row] * width,
                                          values.begin() + ragged.prefix[row + 1] * width);
            };
            const auto state_begin = initial.begin() + source[row] * state_elements;
            const auto reference   = gdn_ref::evaluate(
                {kStateDim, c.qk_heads, heads, ragged.lengths[row],
                   segment(packed.q, kStateDim * c.qk_heads),
                   segment(packed.k, kStateDim * c.qk_heads), segment(packed.v, out_column),
                   segment(packed.g, heads), segment(packed.beta, heads),
                   std::vector<float>(state_begin, state_begin + state_elements)},
                scale, c.normalize_qk);
            const std::string label = std::string(c.name) + "/replay=" + std::to_string(replay) +
                                      "/segment=" + std::to_string(row);
            failures += verify_heads(label + "/output",
                                     std::span(actual).subspan(ragged.prefix[row] * out_column,
                                                               ragged.lengths[row] * out_column),
                                     reference.out, heads, ragged.lengths[row], false);
            failures += verify_heads(
                label + "/state",
                std::span<const double>(after).subspan(destination[row] * state_elements,
                                                       state_elements),
                reference.final_state, heads, 1, true, state_dtype);
        }
        for (int slot = 0; slot < slots; ++slot) {
            if (written[slot]) continue;
            const auto begin = slot * slot_bytes;
            failures += verify_exact(
                "ragged unselected state",
                std::vector<std::byte>(after_bytes.begin() + begin,
                                       after_bytes.begin() + begin + slot_bytes),
                std::vector<std::byte>(initial_state_bytes.begin() + begin,
                                       initial_state_bytes.begin() + begin + slot_bytes));
        }
        failures +=
            output.verify_guards("ragged output") + scratch.verify_guards("ragged workspace") +
            states.verify_guards("ragged state") + history.verify_guards("ragged selected fields");
    }
    const bool all_recurrent =
        std::all_of(ragged.lengths.begin(), ragged.lengths.end(), [&](int t) {
            return ops::gated_delta_net_workspace_capacity_bytes(c.qk_heads, heads, t, t) == 0;
        });
    if (ws.used() != 0 || ws.peak_used() > bytes || (all_recurrent && ws.peak_used() != 0) ||
        (capacity_witness && ws.peak_used() != bytes)) {
        std::cerr << "ragged workspace peak=" << ws.peak_used() << " capacity=" << bytes << '\n';
        ++failures;
    }
    return failures;
}

int ragged_capacity_enumeration() {
    constexpr int max_tokens = 37, max_segments = 5;
    int failures = 0;
    for (const auto [qk, heads] : {std::pair{4, 4}, std::pair{8, 24}}) {
        const auto packet = ops::gated_delta_net_workspace_capacity_bytes(qk, heads, 16, 16);
        const auto count  = [&](int tokens) {
            int packets = 0;
            if (ops::gated_delta_net_workspace_capacity_bytes(qk, heads, tokens, tokens) != 0)
                for (int start = 0; start < tokens; start += 16) ++packets;
            return packets;
        };
        const auto partitions = [&](auto&& self, int tokens, int segments, int packets) -> int {
            if (segments == 1) return packets + count(tokens);
            int best = 0;
            for (int length = 1; length <= tokens - segments + 1; ++length)
                best = std::max(best,
                                self(self, tokens - length, segments - 1, packets + count(length)));
            return best;
        };
        int peaks[max_tokens + 1][max_segments + 1]{};
        for (int n = 1; n <= max_tokens; ++n)
            for (int s = 1; s <= std::min(n, max_segments); ++s)
                peaks[n][s] = partitions(partitions, n, s, 0);

        for (int n = 1; n <= max_tokens; ++n) {
            for (int s = 1; s <= std::min(n, max_segments); ++s) {
                for (int min_n : {1, n}) {
                    for (int upper_s = s; upper_s <= max_segments; ++upper_s) {
                        int max_packets = 0;
                        for (int t = min_n; t <= n; ++t)
                            for (int rows = s; rows <= std::min(t, upper_s); ++rows)
                                max_packets = std::max(max_packets, peaks[t][rows]);
                        const auto expected = packet * max_packets;
                        const auto got      = ops::gated_delta_net_ragged_workspace_capacity_bytes(
                            qk, heads, min_n, n, s, upper_s);
                        if (got != expected) {
                            std::cerr << "ragged capacity T=[" << min_n << ',' << n << "] S=[" << s
                                      << ',' << upper_s << "] expected=" << expected
                                      << " actual=" << got << '\n';
                            ++failures;
                        }
                    }
                }
            }
        }
    }
    try {
        (void)ops::gated_delta_net_ragged_workspace_capacity_bytes(4, 6, 1, 50, 1, 2);
        std::cerr << "ragged capacity accepted an invalid head map\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    try {
        (void)ops::gated_delta_net_ragged_workspace_capacity_bytes(4, 4, 1, 10, 11, 12);
        std::cerr << "ragged capacity accepted an empty envelope\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    if (NINFER_TARGET_SPARK) {
        // S=16, T=4096 admits {4081, 15x1}. Its direct window is larger than the
        // packed window used by a uniform payload with the same N and S.
        const auto expected = ops::gated_delta_net_workspace_capacity_bytes(16, 32, 4081, 4081);
        const auto got =
            ops::gated_delta_net_ragged_workspace_capacity_bytes(16, 32, 4096, 4096, 16, 16);
        if (got != expected) {
            std::cerr << "ragged capacity for a direct oversize window is not tight\n";
            ++failures;
        }
    }
    try {
        constexpr int max = std::numeric_limits<int>::max();
        (void)ops::gated_delta_net_ragged_workspace_capacity_bytes(max, max, 1, max, 1, max);
        std::cerr << "ragged capacity accepted a workspace size overflow\n";
        ++failures;
    } catch (const std::overflow_error&) {}
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    OpSuite suite;
    using Mode            = RaggedStateMode;
    const auto add_ragged = [&](Case c, std::vector<int> lengths, Mode mode, bool graph = false,
                                bool adversarial = false, bool poison = false,
                                bool capacity_witness = false) {
        suite.add(std::string("gated_delta_net/ragged/") + c.name,
                  [=] {
                      return ragged_recurrence(c, lengths, mode, graph, adversarial, poison,
                                               capacity_witness);
                  },
                  {{"execution", graph ? "graph" : "eager"}});
    };
    for (int tokens : {10, 11, 15, 16, 65}) {
        for (bool normalize : {false, true}) {
            const auto name =
                "single/t" + std::to_string(tokens) + (normalize ? "/normalized" : "/raw");
            suite.add("gated_delta_net/ragged/" + name, [=] {
                return ragged_recurrence({name.c_str(), 8, 24, tokens, normalize}, {tokens},
                                         normalize ? Mode::InPlace : Mode::Permuted,
                                         normalize && tokens == 16);
            });
        }
    }
    add_ragged({"boundaries", 16, 32, 0, true}, {1, 2, 10, 11, 15, 16, 17, 31, 32, 33, 65},
               Mode::Mixed, false, true);
    add_ragged({"long-first", 4, 4, 0, true}, {4095, 1}, Mode::InPlace);
    add_ragged({"long-last", 8, 24, 0, false}, {1, 4095}, Mode::Permuted);
    add_ragged({"recurrent-multitoken", 16, 48, 0, false}, std::vector<int>(8, 3), Mode::Mixed,
               true);
    add_ragged({"all-singletons", 16, 32, 0, true}, std::vector<int>(32, 1), Mode::SharedSource);
    add_ragged({"sparse-chunked", 16, 48, 0, false}, {256, 1, 1, 1, 256}, Mode::Permuted, true);
    add_ragged({"boundary-isolation", 4, 4, 0, false}, {17, 17, 15, 17, 1, 17}, Mode::Mixed, false,
               true);
    add_ragged({"value-padding-isolation", 4, 4, 0, true}, {17, 17}, Mode::InPlace, true, false,
               true);
    add_ragged({"windowed-capacity", 4, 4, 0, true}, std::vector<int>(33, 17), Mode::Mixed, true,
               false, false, true);
    add_ragged({"oversize-window", 4, 4, 0, false}, {17, 15, 1201, 17, 1, 17}, Mode::Permuted, true,
               true);
    std::vector<int> overflow(160);
    for (int s = 0; s < static_cast<int>(overflow.size()); ++s) overflow[s] = s % 2 ? 17 : 1;
    add_ragged({"large-mixed-batch", 4, 4, 0, true}, overflow, Mode::SharedSource, true, false,
               false, !NINFER_TARGET_SPARK);
    add_ragged({"large-all-chunked-batch", 1, 1, 0, false}, std::vector<int>(65, 17), Mode::InPlace,
               true, false, false, !NINFER_TARGET_SPARK);
    suite.add("gated_delta_net/ragged/capacity-enumeration", ragged_capacity_enumeration);
    suite.add("gdn/contract", [] {
        int failures               = 0;
        const std::size_t interval = ops::gated_delta_net_workspace_capacity_bytes(16, 48, 15, 17);
        const std::size_t witness  = ops::gated_delta_net_workspace_capacity_bytes(16, 48, 17, 17);
        if (interval != witness) {
            std::cerr << "gated_delta_net interval capacity missed the chunk boundary\n";
            ++failures;
        }
        try {
            (void)ops::gated_delta_net_workspace_capacity_bytes(16, 48, 0, 65);
            std::cerr << "gated_delta_net accepted an invalid token interval\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
        try {
            (void)ops::gated_delta_net_workspace_capacity_bytes(4, 6, 1, 65);
            std::cerr << "gated_delta_net workspace accepted a non-divisible head map\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
        failures += contract_rejection_cases();
        return failures;
    });
    const auto add = [&](Case c, unsigned seed, InputPattern pattern = InputPattern::Ordinary,
                         bool inplace = false, std::vector<int> chunks = {}) {
        const auto id = (std::string(c.name).starts_with("local") ? "gdn_local/" : "gdn/") +
                        std::string(c.name) + "/h" + std::to_string(c.qk_heads) + "/v" +
                        std::to_string(c.value_heads) + "/t" + std::to_string(c.tokens) +
                        (c.normalize_qk ? "/normalized" : "/raw") +
                        (inplace ? "/alias" : "/disjoint");
        suite.add(id, [=] { return run_case(c, seed, pattern, inplace, chunks); },
                  {{"execution", "eager"}});
    };
    const auto batch = [&](Case c, std::vector<int> source, std::vector<int> destination, int slots,
                           unsigned seed) {
        std::string id =
            (std::string(c.name).starts_with("local") ? "gdn_batch_local/" : "gdn/batch/") +
            std::string(c.name) + "/b" + std::to_string(source.size()) +
            (c.normalize_qk ? "/normalized" : "/raw");
        for (int x : source) id += "/s" + std::to_string(x);
        for (int x : destination) id += "/d" + std::to_string(x);
        suite.add(id, [=] { return batch_update_case(c, source, destination, slots, seed); });
    };
    for (int batch : {9, 17, 64}) {
        suite.add("gdn/batch/expanded/b" + std::to_string(batch),
                  [=] {
                      std::vector<int> source(batch), destination(batch), valid(batch, 1);
                      std::iota(source.begin(), source.end(), 0);
                      std::iota(destination.begin(), destination.end(), batch);
                      if (batch != 9)
                          for (int row = 0; row < batch; ++row) valid[row] = row % 4 == 0;
                      return batch_update_case(
                          {"expanded", 8, 24, 1, true}, source, destination, 2 * batch,
                          25000U + batch, batch == 9 ? std::vector<int>{} : valid, batch == 17);
                  },
                  {{"suite", "concurrency"}});
    }
    for (bool normalize : {false, true}) {
        for (int t : {1,   2,   3,   4,   5,   6,   7,    8,    9,    10,   11,  12,  13,
                      14,  15,  16,  17,  31,  32,  33,   63,   64,   65,   127, 128, 129,
                      255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2048, 4096})
            for (bool alias : {false, true})
                add({"local", 8, 24, t, normalize}, 21000U + t, InputPattern::Ordinary, alias);
        for (int t : {1, 10, 11, 12, 16, 17, 129, 1024})
            suite.add("gdn_local/graph/t" + std::to_string(t) +
                          (normalize ? "/normalized" : "/raw"),
                      [=] { return graph_case({"local-graph", 8, 24, t, normalize}, 1, false); },
                      {{"execution", "graph"}});
    }
    for (auto pattern : {InputPattern::WeakDecay, InputPattern::ZeroDecay, InputPattern::SmallBeta,
                         InputPattern::NearCollinear, InputPattern::ZeroQuery,
                         InputPattern::ZeroSignal, InputPattern::ZeroBeta, InputPattern::UnitBeta})
        for (bool alias : {false, true}) {
            const char* name = pattern == InputPattern::WeakDecay       ? "local-weak"
                               : pattern == InputPattern::ZeroDecay     ? "local-zero-decay"
                               : pattern == InputPattern::SmallBeta     ? "local-small-beta"
                               : pattern == InputPattern::NearCollinear ? "local-collinear"
                               : pattern == InputPattern::ZeroQuery     ? "local-zero-query"
                               : pattern == InputPattern::ZeroSignal    ? "local-zero-signal"
                               : pattern == InputPattern::ZeroBeta      ? "local-zero-beta"
                                                                        : "local-unit-beta";
            add({name, 8, 24, 257, true}, 23001U + static_cast<unsigned>(pattern), pattern, alias);
        }
    add({"local-continuation", 8, 24, 1025, true}, 24001U, InputPattern::WeakDecay, true,
        {15, 17, 992, 1});
    add({"27b recurrent", 16, 48, 1, true}, 12001U, InputPattern::Ordinary, true);
    add({"27b recurrent", 16, 48, 7, false}, 12007U);
    add({"35b recurrent", 16, 32, 15, true}, 12015U);

    struct Geometry {
        const char* name;
        int qk_heads, value_heads, tokens;
    };

    for (const auto geometry :
         {Geometry{"qwen-27b", 16, 48, 1025}, Geometry{"qwen-35b-a3b", 16, 32, 1025},
          Geometry{"kimi-k3", 96, 96, 1025}, Geometry{"glm5.3-flash", 64, 64, 1025},
          Geometry{"single-head", 1, 1, 65}, Geometry{"shared-qk", 1, 3, 65},
          Geometry{"generic-grouped-map", 5, 15, 65}}) {
        for (bool normalize : {false, true}) {
            add({geometry.name, geometry.qk_heads, geometry.value_heads, geometry.tokens,
                 normalize},
                16180U + geometry.value_heads, InputPattern::Ordinary, !normalize);
        }
    }
    for (int tokens : {15, 16, 17, 31, 32, 33, 63, 64, 65, 1023, 1024, 1025, 4095, 4096}) {
        add({"prefill boundary", 1, 3, tokens, true}, 90000U + tokens, InputPattern::Ordinary,
            tokens % 2 != 0);
    }
    for (int tokens : {15, 16, 17}) {
        add({"raw prefill boundary", 5, 15, tokens, false}, 91000U + tokens);
    }

    for (int heads : {32, 48}) {
        for (int tokens : {127, 128, 129}) {
            add({"prefill tiling boundary", 16, heads, tokens, tokens != 127},
                92000U + heads * 100U + tokens);
        }
    }

    struct Stress {
        const char* name;
        InputPattern pattern;
        int tokens;
        std::uint32_t seed;
    };

    // Retain the previously observed error cases, including long weak/zero decay. The limits
    // above are fixed before qualification, shared by every geometry and every input pattern.
    for (const auto stress :
         {Stress{"weak decay", InputPattern::WeakDecay, 16385, 332211U},
          Stress{"zero decay", InputPattern::ZeroDecay, 16385, 332212U},
          Stress{"small beta", InputPattern::SmallBeta, 16385, 332213U},
          Stress{"strong decay", InputPattern::StrongDecay, 16385, 332214U},
          Stress{"near collinear", InputPattern::NearCollinear, 4097, 332215U}}) {
        add({stress.name, 1, 3, stress.tokens, true}, stress.seed, stress.pattern);
    }
    for (const auto stress : {Stress{"raw weak decay", InputPattern::WeakDecay, 16385, 332211U},
                              Stress{"raw zero decay", InputPattern::ZeroDecay, 16385, 332212U}}) {
        add({stress.name, 1, 3, stress.tokens, false}, stress.seed, stress.pattern, true);
    }
    add({"35b zero decay", 16, 32, 4097, true}, 16032U, InputPattern::ZeroDecay);
    add({"35b weak decay", 16, 32, 4097, true}, 16032U, InputPattern::WeakDecay, true);
    add({"27b strong tail", 16, 48, 65, true}, 12065U, InputPattern::StrongDecay);
    // The first checkpoint is the aligned T=64 prefix of the identical T=65 stress input.
    add({"27b strong continuation", 16, 48, 65, true}, 12065U, InputPattern::StrongDecay, true,
        {64, 1});

    add({"27b prefill continuation", 16, 48, 2049, true}, 14249U, InputPattern::ZeroState, false,
        {1024, 1024, 1});
    add({"weak prefill continuation", 1, 3, 4097, true}, 14297U, InputPattern::WeakDecay, true,
        {1024, 1024, 1024, 1024, 1});
    add({"raw uneven continuation", 5, 15, 1025, false}, 14125U, InputPattern::ZeroDecay, false,
        {15, 17, 992, 1});
    add({"near-zero normalization", 1, 3, 17, true, true}, 14017U);
    for (bool normalize : {false, true}) {
        add({"zero query", 1, 3, 17, normalize}, 14317U, InputPattern::ZeroQuery);
        add({"zero signal", 1, 3, 17, normalize}, 14417U, InputPattern::ZeroSignal, true);
        add({"zero beta", 1, 3, 17, normalize}, 14517U, InputPattern::ZeroBeta);
        add({"unit beta", 1, 3, 257, normalize}, 14657U, InputPattern::UnitBeta, true);
    }

    for (int b = 1; b <= 8; ++b) {
        std::vector<int> source(b), alias(b), distinct(b), shared(b, 0), fork(b);
        for (int row = 0; row < b; ++row) {
            source[row]   = 2 * row + 2;
            alias[row]    = source[row];
            distinct[row] = source[row] + 1;
            fork[row]     = 2 * row + 1;
        }
        for (bool normalize : {false, true}) {
            batch({"local-ordinary", 8, 24, 1, normalize}, source, alias, 2 * b + 4, 25000U + b);
            batch({"local-ordinary", 8, 24, 1, normalize}, source, distinct, 2 * b + 4, 25000U + b);
            batch({"local-fork", 8, 24, 1, normalize}, shared, fork, 2 * b + 4, 26000U + b);
            batch({"local-near-zero", 8, 24, 1, normalize, true}, source, distinct, 2 * b + 4,
                  27000U + b);
            suite.add(
                "gdn_batch_local/graph/b" + std::to_string(b) +
                    (normalize ? "/normalized" : "/raw"),
                [=] { return graph_case({"local-batch-graph", 8, 24, 1, normalize}, b, true); },
                {{"execution", "graph"}});
        }
    }
    // The production decode path updates selected state-pool slots in place at width one.
    batch({"27b selected-slot fused-qk-norm", 16, 48, 1, true}, {7}, {7}, 8, 12101u);
    batch({"35b selected-slot near-zero", 16, 32, 1, true, true}, {6}, {6}, 8, 12201u);
    batch({"35b ordinary", 16, 32, 1, true}, {8, 9, 10, 11, 12, 13, 14, 15},
          {8, 9, 10, 11, 12, 13, 14, 15}, 16, 13001u);
    batch({"35b mixed fork destinations", 16, 32, 1, true}, {0, 2, 4, 6}, {1, 3, 5, 7}, 8, 13101u);

    suite.add("gdn/graph/running-t1",
              [] { return graph_case({"running-t1", 16, 32, 1, false}, 1, false); },
              {{"execution", "graph"}});
    suite.add("gdn/graph/running-t17",
              [] { return graph_case({"running-t17", 16, 48, 17, true}, 1, false); },
              {{"execution", "graph"}});
    suite.add("gdn/graph/batch-update",
              [] { return graph_case({"batch-update", 16, 48, 1, true}, 4, true); },
              {{"execution", "graph"}});

    // BF16 recurrent state storage coverage across recurrent, chunked, batch_update, ragged, and
    // CUDA Graph paths.
    const auto add_bf16 = [&](Case c, unsigned seed, InputPattern pattern = InputPattern::Ordinary,
                              bool inplace = false, std::vector<int> chunks = {}) {
        const auto id = "gdn_bf16/" + std::string(c.name) + "/h" + std::to_string(c.qk_heads) +
                        "/v" + std::to_string(c.value_heads) + "/t" + std::to_string(c.tokens) +
                        (c.normalize_qk ? "/normalized" : "/raw") +
                        (inplace ? "/alias" : "/disjoint");
        suite.add(id, [=] { return run_case(c, seed, pattern, inplace, chunks, DType::BF16); },
                  {{"execution", "eager"}});
    };
    for (int t : {1, 7, 10, 11, 15, 16, 17, 65, 128, 257, 1025}) {
        for (bool normalize : {false, true}) {
            add_bf16({"bf16", 8, 24, t, normalize}, 41000U + t, InputPattern::Ordinary,
                     (t % 2) == 1);
        }
    }
    add_bf16({"27b-recurrent", 16, 48, 1, true}, 42001U, InputPattern::Ordinary, true);
    add_bf16({"27b-chunked", 16, 48, 1025, true}, 42002U, InputPattern::Ordinary, false);
    add_bf16({"35b-chunked", 16, 32, 1025, false}, 42003U, InputPattern::Ordinary, true);
    add_bf16({"zero-query", 8, 24, 17, true}, 42010U, InputPattern::ZeroQuery, false);
    add_bf16({"zero-signal", 8, 24, 17, true}, 42011U, InputPattern::ZeroSignal, true);
    add_bf16({"zero-beta", 8, 24, 17, true}, 42012U, InputPattern::ZeroBeta, false);
    add_bf16({"unit-beta", 8, 24, 257, true}, 42013U, InputPattern::UnitBeta, true);
    add_bf16({"weak-decay", 8, 24, 257, true}, 42014U, InputPattern::WeakDecay, false);
    add_bf16({"continuation", 8, 24, 97, true}, 42020U, InputPattern::Ordinary, true,
             {15, 17, 64, 1});

    suite.add("gdn_bf16/batch/b1-alias",
              [] {
                  return batch_update_case({"bf16-27b", 16, 48, 1, true}, {7}, {7}, 8, 43001U, {},
                                           false, DType::BF16);
              });
    suite.add("gdn_bf16/batch/b4-fork",
              [] {
                  return batch_update_case({"bf16-35b", 16, 32, 1, true}, {0, 0, 2, 2},
                                           {1, 3, 4, 5}, 8, 43002U, {}, false, DType::BF16);
              });
    suite.add("gdn_bf16/batch/b8-distinct-raw",
              [] {
                  return batch_update_case({"bf16-35b-raw", 16, 32, 1, false},
                                           {0, 2, 4, 6, 8, 10, 12, 14}, {1, 3, 5, 7, 9, 11, 13, 15},
                                           16, 43003U, {}, false, DType::BF16);
              });
    suite.add("gdn_bf16/batch/b17-masked-graph",
              [] {
                  std::vector<int> src(17), dst(17), valid(17, 1);
                  std::iota(src.begin(), src.end(), 0);
                  std::iota(dst.begin(), dst.end(), 17);
                  for (int r = 0; r < 17; ++r) valid[r] = r % 3 != 0;
                  return batch_update_case({"bf16-masked", 8, 24, 1, true}, src, dst, 34, 43017U,
                                           valid, true, DType::BF16);
              },
              {{"execution", "graph"}});

    suite.add("gdn_bf16/graph/running-t1",
              [] { return graph_case({"bf16-running-t1", 16, 32, 1, false}, 1, false, DType::BF16); },
              {{"execution", "graph"}});
    suite.add("gdn_bf16/graph/running-t17",
              [] { return graph_case({"bf16-running-t17", 16, 48, 17, true}, 1, false, DType::BF16); },
              {{"execution", "graph"}});
    suite.add("gdn_bf16/graph/batch-update",
              [] { return graph_case({"bf16-batch-update", 16, 48, 1, true}, 4, true, DType::BF16); },
              {{"execution", "graph"}});

    const auto add_ragged_bf16 = [&](Case c, std::vector<int> lengths, Mode mode,
                                     bool graph = false, bool adversarial = false) {
        suite.add(std::string("gated_delta_net/ragged_bf16/") + c.name,
                  [=] {
                      return ragged_recurrence(c, lengths, mode, graph, adversarial, false, false,
                                               DType::BF16);
                  },
                  {{"execution", graph ? "graph" : "eager"}});
    };
    add_ragged_bf16({"boundaries", 16, 32, 0, true}, {1, 2, 10, 15, 16, 17, 33, 65}, Mode::Mixed,
                    false, true);
    add_ragged_bf16({"recurrent-multitoken", 16, 48, 0, false}, std::vector<int>(8, 3),
                    Mode::Mixed, true);
    add_ragged_bf16({"all-singletons", 16, 32, 0, true}, std::vector<int>(16, 1),
                    Mode::SharedSource);
    add_ragged_bf16({"sparse-chunked", 16, 48, 0, false}, {256, 1, 1, 256}, Mode::Permuted, true);

    return suite.main(argc, argv);
}
