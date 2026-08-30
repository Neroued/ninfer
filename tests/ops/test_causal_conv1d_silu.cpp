#include "ninfer/ops/causal_conv1d_silu.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// CausalConv1dSiLU has one A16 convolution-reduction profile. Decode, small-T, sequence, prefill,
// snapshot, and state-alias choices all qualify against this single normwise criterion.
constexpr ReductionCriterion kCausalConvA16Criterion{
    /*relative_l2*/ 1.85e-3,
    /*gross_absolute*/ 1.0e-3,
    /*gross_relative_to_max_reference*/ 3.7e-3,
};

constexpr std::uint8_t kOutputPoison = 0xff;

std::size_t offset(std::int32_t c, std::int32_t column, std::int32_t C) {
    return static_cast<std::size_t>(column) * static_cast<std::size_t>(C) +
           static_cast<std::size_t>(c);
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { result[i] = f32_to_bf16(values[i]); }
    return result;
}

std::vector<float> make_values(std::size_t count, std::uint32_t seed, float low, float high) {
    std::vector<float> values(count);
    fill_uniform(values, seed, low, high);
    round_to_bf16(values);
    return values;
}

struct LogicalInput {
    std::vector<float> x;
    std::vector<float> weight;
};

LogicalInput make_input(std::int32_t C, std::int32_t T, std::uint32_t seed) {
    LogicalInput input{
        make_values(static_cast<std::size_t>(C) * static_cast<std::size_t>(T), seed, -3.0F, 3.0F),
        make_values(static_cast<std::size_t>(C) * 4U, seed + 1U, -1.0F, 1.0F),
    };

    // Deterministic cancellation and sign cases remain logical inputs to the same oracle. They
    // catch tap-order mistakes without introducing a route-specific reference.
    const std::array<std::int32_t, 3> channels{0, C / 2, C - 1};
    for (const std::int32_t c : channels) {
        input.weight[offset(c, 0, C)] = 1.0F;
        input.weight[offset(c, 1, C)] = -1.0F;
        input.weight[offset(c, 2, C)] = 1.0F;
        input.weight[offset(c, 3, C)] = -1.0F;
        for (std::int32_t t = 0; t < T; ++t) {
            input.x[offset(c, t, C)] = (t & 1) == 0 ? 0.75F : -0.75F;
        }
    }
    return input;
}

std::vector<float> make_state(std::int32_t C, std::uint32_t seed) {
    std::vector<float> state = make_values(static_cast<std::size_t>(C) * 3U, seed, -3.0F, 3.0F);
    for (const std::int32_t c : {0, C / 2, C - 1}) {
        state[offset(c, 0, C)] = 0.75F;
        state[offset(c, 1, C)] = 0.75F;
        state[offset(c, 2, C)] = 0.75F;
    }
    return state;
}

struct OracleResult {
    std::vector<double> output;
    std::vector<float> final_state;
    std::vector<float> snapshots;
};

// The one Op oracle: evaluate the complete convolution and SiLU in FP64 over the logical BF16
// inputs, and separately apply the specified width-3 state transition. It has no production
// staging, accumulator, route, or output-rounding behavior.
OracleResult causal_conv_oracle(const std::vector<float>& x, const std::vector<float>& weight,
                                const std::vector<float>& initial_state, std::int32_t C,
                                std::int32_t T, bool record_snapshots) {
    OracleResult result;
    result.output.resize(static_cast<std::size_t>(C) * static_cast<std::size_t>(T));
    result.final_state = initial_state;
    if (record_snapshots) {
        result.snapshots.resize(static_cast<std::size_t>(C) * 3U * static_cast<std::size_t>(T));
    }

    for (std::int32_t t = 0; t < T; ++t) {
        for (std::int32_t c = 0; c < C; ++c) {
            const double x0 = static_cast<double>(result.final_state[offset(c, 0, C)]);
            const double x1 = static_cast<double>(result.final_state[offset(c, 1, C)]);
            const double x2 = static_cast<double>(result.final_state[offset(c, 2, C)]);
            const double x3 = static_cast<double>(x[offset(c, t, C)]);
            double sum      = 0.0;
            sum += static_cast<double>(weight[offset(c, 0, C)]) * x0;
            sum += static_cast<double>(weight[offset(c, 1, C)]) * x1;
            sum += static_cast<double>(weight[offset(c, 2, C)]) * x2;
            sum += static_cast<double>(weight[offset(c, 3, C)]) * x3;
            result.output[offset(c, t, C)] = sum / (1.0 + std::exp(-sum));
        }

        for (std::int32_t c = 0; c < C; ++c) {
            result.final_state[offset(c, 0, C)] = result.final_state[offset(c, 1, C)];
            result.final_state[offset(c, 1, C)] = result.final_state[offset(c, 2, C)];
            result.final_state[offset(c, 2, C)] = x[offset(c, t, C)];
        }

        if (record_snapshots) {
            std::copy(result.final_state.begin(), result.final_state.end(),
                      result.snapshots.begin() +
                          static_cast<std::size_t>(t) * result.final_state.size());
        }
    }
    return result;
}

int verify_bits(const std::string& label, const void* device,
                const std::vector<std::uint16_t>& ref) {
    return verify_exact(label.c_str(), from_device<std::uint16_t>(device, ref.size()), ref);
}

int verify_buffer_guards(const std::string& label, const GuardedDeviceBuffer& buffer) {
    return buffer.verify_guards(label.c_str());
}

int verify_output(const std::string& label, const std::vector<double>& got,
                  const std::vector<double>& reference) {
    return verify_reduction(label.c_str(), got, reference, kCausalConvA16Criterion);
}

enum class StateCall {
    InPlaceEntry,
    DistinctEntry,
    DistinctEntryExactAlias,
};

const char* call_name(StateCall call) {
    switch (call) {
    case StateCall::InPlaceEntry:
        return "in-place-entry";
    case StateCall::DistinctEntry:
        return "distinct-entry";
    case StateCall::DistinctEntryExactAlias:
        return "distinct-entry-exact-alias";
    }
    return "unknown";
}

// Slice one channel range out of the oracle's packed [column][channel] output.
std::vector<double> oracle_slice(const std::vector<double>& packed, std::int32_t C, std::int32_t T,
                                 std::int32_t first, std::int32_t rows) {
    std::vector<double> slice(static_cast<std::size_t>(rows) * static_cast<std::size_t>(T));
    for (std::int32_t column = 0; column < T; ++column) {
        for (std::int32_t row = 0; row < rows; ++row) {
            slice[offset(row, column, rows)] = packed[offset(first + row, column, C)];
        }
    }
    return slice;
}

// The packed distinct-state entry enforces the same family rules the split entry does. One case
// per rule it gained.
int packed_rejection_case(std::int32_t T) {
    constexpr std::int32_t C = 8192;
    const std::size_t n      = static_cast<std::size_t>(C) * static_cast<std::size_t>(T);
    GuardedDeviceBuffer x(n * sizeof(std::uint16_t));
    GuardedDeviceBuffer weight(static_cast<std::size_t>(C) * 4 * sizeof(std::uint16_t));
    GuardedDeviceBuffer state_in(static_cast<std::size_t>(C) * 3 * sizeof(std::uint16_t));
    GuardedDeviceBuffer state_out(static_cast<std::size_t>(C) * 3 * sizeof(std::uint16_t));
    GuardedDeviceBuffer out(n * sizeof(std::uint16_t));

    const Tensor tx(x.data(), DType::BF16, {C, T});
    const Tensor tw(weight.data(), DType::BF16, {C, 4});
    const Tensor tsi(state_in.data(), DType::BF16, {C, 3});

    auto is_rejected = [&](const Tensor& state_in_arg, Tensor state_out_arg, Tensor out_arg) {
        try {
            ops::causal_conv1d_silu(tx, tw, state_in_arg, state_out_arg, out_arg, nullptr);
        } catch (const std::invalid_argument&) { return true; }
        cuda_synchronize();
        return false;
    };

    int failures         = 0;
    auto expect_rejected = [&](bool rejected, const char* what) {
        if (!rejected) {
            std::cout << "FAIL causal_conv1d_silu T=" << T << " accepted " << what << '\n';
            ++failures;
        }
    };
    Tensor tso(state_out.data(), DType::BF16, {C, 3});
    Tensor tout(out.data(), DType::BF16, {C, T});
    Tensor out_over_x(x.data(), DType::BF16, {C, T});
    Tensor shifted_state_out(static_cast<std::uint16_t*>(state_in.data()) + 2, DType::BF16, {C, 3});

    expect_rejected(is_rejected(tsi, shifted_state_out, tout),
                    "a state pair that overlaps without being the same storage");
    expect_rejected(is_rejected(tsi, tso, out_over_x), "an out that overlaps x");
    expect_rejected(is_rejected(tsi, tout, tout), "an out that overlaps conv_state_out");
    return failures;
}

// The entry promises that a bad argument is rejected the same way whatever the column count.
// One case per rejected class, on both sides of the route bound.
int split_rejection_case(std::int32_t T) {
    constexpr std::int32_t C  = 8192;
    constexpr std::int32_t KD = 2048;
    constexpr std::int32_t VD = 4096;
    const std::size_t n       = static_cast<std::size_t>(C) * static_cast<std::size_t>(T);
    GuardedDeviceBuffer x(n * sizeof(std::uint16_t));
    GuardedDeviceBuffer weight(static_cast<std::size_t>(C) * 4 * sizeof(std::uint16_t));
    GuardedDeviceBuffer state_in(static_cast<std::size_t>(C) * 3 * sizeof(std::uint16_t));
    GuardedDeviceBuffer state_out(static_cast<std::size_t>(C) * 3 * sizeof(std::uint16_t));
    GuardedDeviceBuffer q(static_cast<std::size_t>(KD) * T * sizeof(std::uint16_t));
    GuardedDeviceBuffer k(static_cast<std::size_t>(KD) * T * sizeof(std::uint16_t));
    GuardedDeviceBuffer v(static_cast<std::size_t>(VD) * T * sizeof(std::uint16_t));

    const Tensor tx(x.data(), DType::BF16, {C, T});
    const Tensor tw(weight.data(), DType::BF16, {C, 4});
    const Tensor tsi(state_in.data(), DType::BF16, {C, 3});
    Tensor tq(q.data(), DType::BF16, {KD, T});
    Tensor tk(k.data(), DType::BF16, {KD, T});
    Tensor tv(v.data(), DType::BF16, {VD, T});

    const Tensor bad_null(nullptr, DType::BF16, {C, T});
    const Tensor bad_dtype(weight.data(), DType::FP32, {C, 4});
    Tensor bad_rank(q.data(), DType::BF16, {KD, T, 2});
    Tensor bad_profile(q.data(), DType::BF16, {KD + 1, T});
    Tensor other_profile(v.data(), DType::BF16, {6144, T});
    Tensor bad_dest_dtype(q.data(), DType::FP32, {KD, T});
    Tensor null_dest(nullptr, DType::BF16, {KD, T});
    Tensor short_dest(q.data(), DType::BF16, {KD, T - 1});
    Tensor over_x(x.data(), DType::BF16, {KD, T});
    Tensor over_state(state_in.data(), DType::BF16, {KD, T});
    Tensor misaligned(static_cast<std::uint16_t*>(q.data()) + 1, DType::BF16, {KD, T});

    auto is_rejected = [&](const Tensor& x_in, const Tensor& weight_in, const Tensor& state_in_arg,
                           Tensor state_out_arg, Tensor q_in, Tensor k_in, Tensor v_in) {
        try {
            ops::causal_conv1d_silu_split(x_in, weight_in, state_in_arg, state_out_arg, q_in, k_in,
                                          v_in, nullptr);
        } catch (const std::invalid_argument&) { return true; }
        cuda_synchronize();
        return false;
    };
    Tensor tso(state_out.data(), DType::BF16, {C, 3});
    // A state pair that shares all but one element: the contract allows disjoint or exactly the
    // same storage, and this is neither.
    Tensor shifted_state_out(static_cast<std::uint16_t*>(state_in.data()) + 2, DType::BF16, {C, 3});

    int failures = 0;
    auto expect_rejected = [&](bool rejected, const char* what) {
        if (!rejected) {
            std::cout << "FAIL causal_conv1d_silu_split T=" << T << " accepted " << what << "\n";
            ++failures;
        }
    };
    expect_rejected(is_rejected(bad_null, tw, tsi, tso, tq, tk, tv), "a null x");
    expect_rejected(is_rejected(tx, bad_dtype, tsi, tso, tq, tk, tv), "an FP32 weight");
    expect_rejected(is_rejected(tx, tw, tsi, tso, bad_rank, tk, tv), "a rank-3 destination");
    expect_rejected(is_rejected(tx, tw, tsi, tso, bad_profile, tk, tv),
                    "a row profile that is not registered");
    expect_rejected(is_rejected(tx, tw, tsi, tso, tq, tk, other_profile),
                    "the other geometry's row profile on these channels");
    expect_rejected(is_rejected(tx, tw, tsi, tso, bad_dest_dtype, tk, tv), "an FP32 destination");
    expect_rejected(is_rejected(tx, tw, tsi, tso, null_dest, tk, tv), "a null destination");
    expect_rejected(is_rejected(tx, tw, tsi, tso, short_dest, tk, tv),
                    "a destination with fewer columns than x");
    expect_rejected(is_rejected(tx, tw, tsi, tso, tq, tq, tv),
                    "two destinations in the same storage");
    expect_rejected(is_rejected(tx, tw, tsi, tso, over_x, tk, tv), "a destination overlapping x");
    expect_rejected(is_rejected(tx, tw, tsi, tso, over_state, tk, tv),
                    "a destination overlapping the input state");
    expect_rejected(is_rejected(tx, tw, tsi, tso, misaligned, tk, tv),
                    "a destination that is not four-byte aligned");
    expect_rejected(is_rejected(tx, tw, tsi, shifted_state_out, tq, tk, tv),
                    "a state pair that overlaps without being the same storage");
    return failures;
}

// One registered geometry, one column count, one state form. `offset_pairs` shifts every
// destination by one BF16 pair, which the contract admits and the arena never produces.
int split_case(std::int32_t C, std::int32_t q_dim, std::int32_t k_dim, std::int32_t value_dim,
               std::int32_t T, bool alias_state, bool offset_pairs, std::uint32_t seed) {
    const LogicalInput input       = make_input(C, T, seed);
    const std::vector<float> state = make_state(C, seed + 2U);
    const OracleResult oracle      = causal_conv_oracle(input.x, input.weight, state, C, T, false);
    const std::vector<std::uint16_t> x_bits      = bf16_bits(input.x);
    const std::vector<std::uint16_t> weight_bits = bf16_bits(input.weight);
    const std::vector<std::uint16_t> state_bits  = bf16_bits(state);
    const std::vector<std::uint16_t> final_bits  = bf16_bits(oracle.final_state);

    const std::size_t pad     = offset_pairs ? 2U * sizeof(std::uint16_t) : 0U;
    const std::size_t q_bytes = static_cast<std::size_t>(q_dim) * T * sizeof(std::uint16_t);
    const std::size_t k_bytes = static_cast<std::size_t>(k_dim) * T * sizeof(std::uint16_t);
    const std::size_t v_bytes = static_cast<std::size_t>(value_dim) * T * sizeof(std::uint16_t);

    GuardedDeviceBuffer x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer weight(weight_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer state_in(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer state_out(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer q(q_bytes + pad);
    GuardedDeviceBuffer k(k_bytes + pad);
    GuardedDeviceBuffer v(v_bytes + pad);

    x.copy_from_host(x_bits.data(), x.bytes());
    weight.copy_from_host(weight_bits.data(), weight.bytes());
    state_in.copy_from_host(state_bits.data(), state_in.bytes());
    state_out.fill(0x5a);
    q.fill(kOutputPoison);
    k.fill(kOutputPoison);
    v.fill(kOutputPoison);

    auto at = [&](GuardedDeviceBuffer& buffer) {
        return static_cast<void*>(static_cast<char*>(buffer.data()) + pad);
    };
    auto pad_is_untouched = [&](GuardedDeviceBuffer& buffer, const char* label) {
        if (pad == 0) { return 0; }
        std::vector<std::uint8_t> head(pad);
        buffer.copy_to_host(head.data(), pad);
        for (const std::uint8_t byte : head) {
            if (byte != kOutputPoison) {
                std::cout << "FAIL causal_conv1d_silu_split wrote before " << label << '\n';
                return 1;
            }
        }
        return 0;
    };

    Tensor tx(x.data(), DType::BF16, {C, T});
    Tensor tw(weight.data(), DType::BF16, {C, 4});
    Tensor ts_in(state_in.data(), DType::BF16, {C, 3});
    Tensor ts_out((alias_state ? state_in : state_out).data(), DType::BF16, {C, 3});
    Tensor tq(at(q), DType::BF16, {q_dim, T});
    Tensor tk(at(k), DType::BF16, {k_dim, T});
    Tensor tv(at(v), DType::BF16, {value_dim, T});

    ops::causal_conv1d_silu_split(tx, tw, ts_in, ts_out, tq, tk, tv, nullptr);
    cuda_synchronize();

    const std::string tag     = "causal_conv1d_silu_split C=" + std::to_string(C) +
                                " T=" + std::to_string(T) + (alias_state ? " alias" : "") +
                                (offset_pairs ? " offset" : "");
    const std::size_t q_count = static_cast<std::size_t>(q_dim) * static_cast<std::size_t>(T);
    const std::size_t k_count = static_cast<std::size_t>(k_dim) * static_cast<std::size_t>(T);
    const std::size_t v_count = static_cast<std::size_t>(value_dim) * static_cast<std::size_t>(T);

    int failures = 0;
    failures += pad_is_untouched(q, "out0");
    failures += pad_is_untouched(k, "out1");
    failures += pad_is_untouched(v, "out2");
    failures += verify_output(tag + " q", from_device_bf16(at(q), q_count),
                              oracle_slice(oracle.output, C, T, 0, q_dim));
    failures += verify_output(tag + " k", from_device_bf16(at(k), k_count),
                              oracle_slice(oracle.output, C, T, q_dim, k_dim));
    failures += verify_output(tag + " v", from_device_bf16(at(v), v_count),
                              oracle_slice(oracle.output, C, T, q_dim + k_dim, value_dim));
    failures +=
        verify_bits(tag + " final state", (alias_state ? state_in : state_out).data(), final_bits);
    failures += verify_bits(tag + " x preserved", x.data(), x_bits);
    failures += verify_bits(tag + " weight preserved", weight.data(), weight_bits);
    if (!alias_state) {
        failures += verify_bits(tag + " initial state preserved", state_in.data(), state_bits);
    }
    failures += verify_buffer_guards(tag + " x", x);
    failures += verify_buffer_guards(tag + " weight", weight);
    failures += verify_buffer_guards(tag + " state input", state_in);
    failures += verify_buffer_guards(tag + " state output", state_out);
    failures += verify_buffer_guards(tag + " q", q);
    failures += verify_buffer_guards(tag + " k", k);
    failures += verify_buffer_guards(tag + " v", v);
    return failures;
}

int ordinary_case(std::int32_t C, std::int32_t T, StateCall call, std::uint32_t seed) {
    const LogicalInput input       = make_input(C, T, seed);
    const std::vector<float> state = make_state(C, seed + 2U);
    const OracleResult oracle      = causal_conv_oracle(input.x, input.weight, state, C, T, false);
    const std::vector<std::uint16_t> x_bits      = bf16_bits(input.x);
    const std::vector<std::uint16_t> weight_bits = bf16_bits(input.weight);
    const std::vector<std::uint16_t> state_bits  = bf16_bits(state);
    const std::vector<std::uint16_t> final_bits  = bf16_bits(oracle.final_state);

    GuardedDeviceBuffer x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer weight(weight_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer state_in(state_bits.size() * sizeof(std::uint16_t));
    std::unique_ptr<GuardedDeviceBuffer> state_out;
    if (call == StateCall::DistinctEntry) {
        state_out =
            std::make_unique<GuardedDeviceBuffer>(state_bits.size() * sizeof(std::uint16_t));
        state_out->fill(0x5a);
    }
    GuardedDeviceBuffer output(x_bits.size() * sizeof(std::uint16_t));
    x.copy_from_host(x_bits.data(), x.bytes());
    weight.copy_from_host(weight_bits.data(), weight.bytes());
    state_in.copy_from_host(state_bits.data(), state_in.bytes());
    output.fill(kOutputPoison);

    Tensor tx(x.data(), DType::BF16, {C, T});
    Tensor tw(weight.data(), DType::BF16, {C, 4});
    Tensor ts_in(state_in.data(), DType::BF16, {C, 3});
    Tensor ts_out(state_out == nullptr ? state_in.data() : state_out->data(), DType::BF16, {C, 3});
    Tensor tout(output.data(), DType::BF16, {C, T});

    if (call == StateCall::InPlaceEntry) {
        ops::causal_conv1d_silu(tx, tw, ts_in, tout, nullptr);
    } else if (call == StateCall::DistinctEntry) {
        ops::causal_conv1d_silu(tx, tw, ts_in, ts_out, tout, nullptr);
    } else {
        ops::causal_conv1d_silu(tx, tw, ts_in, ts_in, tout, nullptr);
    }
    cuda_synchronize();

    const std::string tag = "causal_conv1d_silu C=" + std::to_string(C) +
                            " T=" + std::to_string(T) + " " + call_name(call);
    int failures = 0;
    failures += verify_output(tag + " output", from_device_bf16(output.data(), x_bits.size()),
                              oracle.output);
    failures += verify_bits(tag + " final state",
                            state_out == nullptr ? state_in.data() : state_out->data(), final_bits);
    failures += verify_bits(tag + " x preserved", x.data(), x_bits);
    failures += verify_bits(tag + " weight preserved", weight.data(), weight_bits);
    if (call == StateCall::DistinctEntry) {
        failures += verify_bits(tag + " initial state preserved", state_in.data(), state_bits);
    }
    failures += verify_buffer_guards(tag + " x", x);
    failures += verify_buffer_guards(tag + " weight", weight);
    failures += verify_buffer_guards(tag + " state input", state_in);
    if (state_out != nullptr) {
        failures += verify_buffer_guards(tag + " state output", *state_out);
    }
    failures += verify_buffer_guards(tag + " output", output);
    return failures;
}

// Continuation prefill reads a selected committed slot and publishes the new running state to slot
// 0. The Op sees two valid disjoint [C,3] tensors; checking their common backing allocation also
// proves that every surrounding state slot remains untouched.
int continuation_slot_case(std::int32_t C, std::int32_t T, std::int32_t slots,
                           std::int32_t read_slot, std::uint32_t seed) {
    const LogicalInput input        = make_input(C, T, seed);
    const std::size_t slot_elements = static_cast<std::size_t>(C) * 3U;
    std::vector<float> states(slot_elements * static_cast<std::size_t>(slots));
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        std::vector<float> one = make_state(C, seed + 10U + static_cast<std::uint32_t>(slot));
        std::copy(one.begin(), one.end(),
                  states.begin() + static_cast<std::size_t>(slot) * slot_elements);
    }

    const auto selected_begin =
        states.begin() + static_cast<std::size_t>(read_slot) * slot_elements;
    const std::vector<float> selected_state(selected_begin, selected_begin + slot_elements);
    const OracleResult oracle =
        causal_conv_oracle(input.x, input.weight, selected_state, C, T, false);
    std::vector<float> expected_states = states;
    std::copy(oracle.final_state.begin(), oracle.final_state.end(), expected_states.begin());

    const std::vector<std::uint16_t> x_bits        = bf16_bits(input.x);
    const std::vector<std::uint16_t> weight_bits   = bf16_bits(input.weight);
    const std::vector<std::uint16_t> state_bits    = bf16_bits(states);
    const std::vector<std::uint16_t> expected_bits = bf16_bits(expected_states);

    GuardedDeviceBuffer x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer weight(weight_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer state(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer output(x_bits.size() * sizeof(std::uint16_t));
    x.copy_from_host(x_bits.data(), x.bytes());
    weight.copy_from_host(weight_bits.data(), weight.bytes());
    state.copy_from_host(state_bits.data(), state.bytes());
    output.fill(kOutputPoison);

    auto* selected_device =
        static_cast<std::uint8_t*>(state.data()) +
        static_cast<std::size_t>(read_slot) * slot_elements * sizeof(std::uint16_t);
    Tensor tx(x.data(), DType::BF16, {C, T});
    Tensor tw(weight.data(), DType::BF16, {C, 4});
    Tensor ts_in(selected_device, DType::BF16, {C, 3});
    Tensor ts_out(state.data(), DType::BF16, {C, 3});
    Tensor tout(output.data(), DType::BF16, {C, T});
    ops::causal_conv1d_silu(tx, tw, ts_in, ts_out, tout, nullptr);
    cuda_synchronize();

    const std::string tag = "causal_conv1d_silu continuation C=" + std::to_string(C) +
                            " T=" + std::to_string(T) + " read_slot=" + std::to_string(read_slot);
    int failures = 0;
    failures += verify_output(tag + " output", from_device_bf16(output.data(), x_bits.size()),
                              oracle.output);
    failures += verify_bits(tag + " all state slots", state.data(), expected_bits);
    failures += verify_bits(tag + " x preserved", x.data(), x_bits);
    failures += verify_bits(tag + " weight preserved", weight.data(), weight_bits);
    failures += verify_buffer_guards(tag + " x", x);
    failures += verify_buffer_guards(tag + " weight", weight);
    failures += verify_buffer_guards(tag + " states", state);
    failures += verify_buffer_guards(tag + " output", output);
    return failures;
}

int snapshot_case(std::int32_t C, std::int32_t T, std::int32_t slots, std::int32_t initial_slot,
                  std::int32_t snapshot_base_slot, std::uint32_t seed) {
    const LogicalInput input        = make_input(C, T, seed);
    const std::size_t slot_elements = static_cast<std::size_t>(C) * 3U;
    std::vector<float> states(slot_elements * static_cast<std::size_t>(slots));
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        std::vector<float> one = make_state(C, seed + 10U + static_cast<std::uint32_t>(slot));
        std::copy(one.begin(), one.end(),
                  states.begin() + static_cast<std::size_t>(slot) * slot_elements);
    }

    const auto selected_begin =
        states.begin() + static_cast<std::size_t>(initial_slot) * slot_elements;
    const std::vector<float> selected_state(selected_begin, selected_begin + slot_elements);
    const OracleResult oracle =
        causal_conv_oracle(input.x, input.weight, selected_state, C, T, true);

    std::vector<float> expected_states = states;
    std::copy(oracle.snapshots.begin(), oracle.snapshots.end(),
              expected_states.begin() +
                  static_cast<std::size_t>(snapshot_base_slot) * slot_elements);

    const std::vector<std::uint16_t> x_bits        = bf16_bits(input.x);
    const std::vector<std::uint16_t> weight_bits   = bf16_bits(input.weight);
    const std::vector<std::uint16_t> state_bits    = bf16_bits(states);
    const std::vector<std::uint16_t> expected_bits = bf16_bits(expected_states);
    const std::int32_t initial_slot_host           = initial_slot;
    const std::int32_t snapshot_base_slot_host     = snapshot_base_slot;

    GuardedDeviceBuffer x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer weight(weight_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer state(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer slot(sizeof(std::int32_t));
    GuardedDeviceBuffer snapshot_base(sizeof(std::int32_t));
    GuardedDeviceBuffer output(x_bits.size() * sizeof(std::uint16_t));
    x.copy_from_host(x_bits.data(), x.bytes());
    weight.copy_from_host(weight_bits.data(), weight.bytes());
    state.copy_from_host(state_bits.data(), state.bytes());
    slot.copy_from_host(&initial_slot_host, sizeof(initial_slot_host));
    snapshot_base.copy_from_host(&snapshot_base_slot_host, sizeof(snapshot_base_slot_host));
    output.fill(kOutputPoison);

    Tensor tx(x.data(), DType::BF16, {C, T});
    Tensor tw(weight.data(), DType::BF16, {C, 4});
    Tensor tstate(state.data(), DType::BF16, {C, 3, slots});
    Tensor tslot(slot.data(), DType::I32, {1});
    Tensor tsnapshot_base(snapshot_base.data(), DType::I32, {1});
    Tensor tout(output.data(), DType::BF16, {C, T});
    ops::causal_conv1d_silu_snapshot(tx, tw, tstate, Tensor{}, tslot, tsnapshot_base, tout,
                                     nullptr);
    cuda_synchronize();

    const std::string tag = "causal_conv1d_silu snapshot C=" + std::to_string(C) +
                            " T=" + std::to_string(T) + " slots=" + std::to_string(slots) +
                            " initial_slot=" + std::to_string(initial_slot) +
                            " snapshot_base_slot=" + std::to_string(snapshot_base_slot);
    int failures = 0;
    failures += verify_output(tag + " output", from_device_bf16(output.data(), x_bits.size()),
                              oracle.output);
    failures += verify_bits(tag + " all state slots", state.data(), expected_bits);
    failures += verify_bits(tag + " x preserved", x.data(), x_bits);
    failures += verify_bits(tag + " weight preserved", weight.data(), weight_bits);
    failures += verify_exact((tag + " initial slot preserved").c_str(),
                             from_device<std::int32_t>(slot.data(), 1),
                             std::vector<std::int32_t>{initial_slot});
    failures += verify_exact((tag + " snapshot base slot preserved").c_str(),
                             from_device<std::int32_t>(snapshot_base.data(), 1),
                             std::vector<std::int32_t>{snapshot_base_slot});
    failures += verify_buffer_guards(tag + " x", x);
    failures += verify_buffer_guards(tag + " weight", weight);
    failures += verify_buffer_guards(tag + " states", state);
    failures += verify_buffer_guards(tag + " initial slot", slot);
    failures += verify_buffer_guards(tag + " snapshot base slot", snapshot_base);
    failures += verify_buffer_guards(tag + " output", output);
    return failures;
}

int batched_snapshot_case(std::int32_t C, std::int32_t width,
                          const std::vector<std::int32_t>& initial_slots,
                          const std::vector<std::int32_t>& snapshot_bases,
                          const std::vector<std::int32_t>& valid_columns, std::int32_t slots,
                          std::uint32_t seed) {
    const std::int32_t batch        = static_cast<std::int32_t>(initial_slots.size());
    const bool masked               = !valid_columns.empty();
    const LogicalInput input        = make_input(C, width * batch, seed);
    const std::size_t slot_elements = static_cast<std::size_t>(C) * 3U;
    const std::size_t row_elements  = static_cast<std::size_t>(C) * width;

    std::vector<float> states(slot_elements * static_cast<std::size_t>(slots));
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        std::vector<float> one = make_state(C, seed + 10U + static_cast<std::uint32_t>(slot));
        std::copy(one.begin(), one.end(),
                  states.begin() + static_cast<std::size_t>(slot) * slot_elements);
    }

    std::vector<double> expected_output(row_elements * static_cast<std::size_t>(batch), 0.0);
    std::vector<float> expected_states = states;
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t valid = masked ? valid_columns[static_cast<std::size_t>(row)] : width;
        const std::size_t input_begin = static_cast<std::size_t>(row) * row_elements;
        std::vector<float> row_input(input.x.begin() + input_begin,
                                     input.x.begin() + input_begin +
                                         static_cast<std::size_t>(valid) * C);
        const std::size_t initial_begin =
            static_cast<std::size_t>(initial_slots[static_cast<std::size_t>(row)]) * slot_elements;
        std::vector<float> initial_state(states.begin() + initial_begin,
                                         states.begin() + initial_begin + slot_elements);
        const OracleResult oracle =
            causal_conv_oracle(row_input, input.weight, initial_state, C, valid, true);
        std::copy(oracle.output.begin(), oracle.output.end(),
                  expected_output.begin() + input_begin);
        const std::size_t destination_begin =
            static_cast<std::size_t>(snapshot_bases[static_cast<std::size_t>(row)]) * slot_elements;
        std::copy(oracle.snapshots.begin(), oracle.snapshots.end(),
                  expected_states.begin() + destination_begin);
    }

    const std::vector<std::uint16_t> x_bits        = bf16_bits(input.x);
    const std::vector<std::uint16_t> weight_bits   = bf16_bits(input.weight);
    const std::vector<std::uint16_t> state_bits    = bf16_bits(states);
    const std::vector<std::uint16_t> expected_bits = bf16_bits(expected_states);

    GuardedDeviceBuffer x(x_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer weight(weight_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer state(state_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer initial(initial_slots.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer bases(snapshot_bases.size() * sizeof(std::int32_t));
    std::unique_ptr<GuardedDeviceBuffer> valid;
    if (masked) {
        valid = std::make_unique<GuardedDeviceBuffer>(valid_columns.size() * sizeof(std::int32_t));
        valid->copy_from_host(valid_columns.data(), valid->bytes());
    }
    GuardedDeviceBuffer output(x_bits.size() * sizeof(std::uint16_t));
    x.copy_from_host(x_bits.data(), x.bytes());
    weight.copy_from_host(weight_bits.data(), weight.bytes());
    state.copy_from_host(state_bits.data(), state.bytes());
    initial.copy_from_host(initial_slots.data(), initial.bytes());
    bases.copy_from_host(snapshot_bases.data(), bases.bytes());
    output.fill(kOutputPoison);

    Tensor tx(x.data(), DType::BF16, {C, width, batch});
    Tensor tw(weight.data(), DType::BF16, {C, 4});
    Tensor tstate(state.data(), DType::BF16, {C, 3, slots});
    Tensor tvalid;
    if (masked) { tvalid = Tensor(valid->data(), DType::I32, {batch}); }
    Tensor tinitial(initial.data(), DType::I32, {batch});
    Tensor tbases(bases.data(), DType::I32, {batch});
    Tensor tout(output.data(), DType::BF16, {C, width, batch});
    ops::causal_conv1d_silu_snapshot(tx, tw, tstate, tvalid, tinitial, tbases, tout, nullptr);
    cuda_synchronize();

    const std::string tag = "causal_conv1d_silu batched snapshot C=" + std::to_string(C) +
                            " W=" + std::to_string(width) + " B=" + std::to_string(batch) +
                            (masked ? " masked" : " dense");
    int failures = 0;
    failures += verify_output(tag + " output", from_device_bf16(output.data(), x_bits.size()),
                              expected_output);
    failures += verify_bits(tag + " all state slots", state.data(), expected_bits);
    failures += verify_bits(tag + " x preserved", x.data(), x_bits);
    failures += verify_bits(tag + " weight preserved", weight.data(), weight_bits);
    failures += verify_exact((tag + " initial selectors preserved").c_str(),
                             from_device<std::int32_t>(initial.data(), initial_slots.size()),
                             initial_slots);
    failures += verify_exact((tag + " snapshot bases preserved").c_str(),
                             from_device<std::int32_t>(bases.data(), snapshot_bases.size()),
                             snapshot_bases);
    if (masked) {
        failures += verify_exact((tag + " valid columns preserved").c_str(),
                                 from_device<std::int32_t>(valid->data(), valid_columns.size()),
                                 valid_columns);
        failures += verify_buffer_guards(tag + " valid columns", *valid);
    }
    failures += verify_buffer_guards(tag + " x", x);
    failures += verify_buffer_guards(tag + " weight", weight);
    failures += verify_buffer_guards(tag + " states", state);
    failures += verify_buffer_guards(tag + " initial selectors", initial);
    failures += verify_buffer_guards(tag + " snapshot bases", bases);
    failures += verify_buffer_guards(tag + " output", output);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;

    // The 27B geometry exercises every ordinary positive-T route boundary and one interior point
    // per route. Every case is compared directly with the same complete oracle.
    constexpr std::int32_t kQwen27Channels = 10240;
    for (const std::int32_t T : {1, 2, 7, 15, 16, 17, 32, 63, 64, 65, 257}) {
        failures += ordinary_case(kQwen27Channels, T, StateCall::InPlaceEntry,
                                  1000U + static_cast<std::uint32_t>(T));
    }

    // Qualify the second public entry directly, including its disjoint and exact-alias state forms.
    failures += ordinary_case(kQwen27Channels, 1, StateCall::DistinctEntry, 2001U);
    failures += ordinary_case(kQwen27Channels, 7, StateCall::DistinctEntry, 2007U);
    failures += ordinary_case(kQwen27Channels, 17, StateCall::DistinctEntryExactAlias, 2017U);
    failures += ordinary_case(kQwen27Channels, 32, StateCall::DistinctEntry, 2032U);
    failures += continuation_slot_case(kQwen27Channels, 65, 6, 4, 2065U);

    // The peer 35B-A3B geometry is a separate real channel extent.
    constexpr std::int32_t kQwen35Channels = 8192;
    failures += ordinary_case(kQwen35Channels, 257, StateCall::InPlaceEntry, 3257U);

    // Snapshot decode, small-T boundary/interior, sequence route, slot 0 initialization, and
    // continuation from selected slots. A nonzero destination base proves that snapshots are not
    // hard-wired to physical slots [0,T); selected source slots may overlap the destination range.
    failures += snapshot_case(kQwen27Channels, 1, 4, 0, 1, 4001U);
    failures += snapshot_case(kQwen27Channels, 2, 5, 4, 1, 4002U);
    failures += snapshot_case(kQwen27Channels, 7, 9, 3, 1, 4007U);
    failures += snapshot_case(kQwen27Channels, 15, 18, 14, 1, 4015U);
    failures += snapshot_case(kQwen27Channels, 16, 18, 17, 1, 4016U);
    failures += snapshot_case(kQwen35Channels, 17, 20, 16, 1, 4017U);
    // Above the small-T bound the snapshot entry takes the sequence-snapshot kernel.
    failures += snapshot_case(kQwen35Channels, 40, 44, 40, 1, 4040U);

    failures += batched_snapshot_case(kQwen35Channels, 1, {8, 9, 10, 11, 12, 13, 14, 15},
                                      {0, 1, 2, 3, 4, 5, 6, 7}, {}, 16, 5001U);
    failures +=
        batched_snapshot_case(kQwen27Channels, 6, {18, 19, 20}, {0, 6, 12}, {6, 3, 1}, 21, 5006U);
    // Row 0 reads slot 15 and overwrites it only after the final valid column. This is the
    // production same-row alias pattern; row 1 remains fully disjoint.
    failures += batched_snapshot_case(kQwen35Channels, 16, {15, 33}, {0, 16}, {16, 7}, 34, 5016U);

    // The split entry: both channel geometries, across the column counts that select each of its
    // two routes and the boundary between them.
    for (const std::int32_t T : {1, 2, 7, 15, 16, 17, 32, 33, 63, 64, 65, 257, 1024}) {
        failures += split_case(kQwen27Channels, 2048, 2048, 6144, T, false, false,
                               6000U + static_cast<std::uint32_t>(T));
        failures += split_case(kQwen35Channels, 2048, 2048, 4096, T, false, false,
                               7000U + static_cast<std::uint32_t>(T));
    }

    // The exact-alias state form, on both geometries, across every route boundary.
    for (const std::int32_t T : {1, 2, 15, 16, 17, 32, 33, 64, 65, 257}) {
        failures += split_case(kQwen27Channels, 2048, 2048, 6144, T, true, false,
                               6100U + static_cast<std::uint32_t>(T));
        failures += split_case(kQwen35Channels, 2048, 2048, 4096, T, true, false,
                               7100U + static_cast<std::uint32_t>(T));
    }

    // Destinations offset by a pair: four-byte aligned, which the contract admits, but not the
    // 256-byte alignment the arena happens to give.
    for (const std::int32_t T : {16, 33, 257}) {
        failures += split_case(kQwen27Channels, 2048, 2048, 6144, T, false, true,
                               6200U + static_cast<std::uint32_t>(T));
        failures += split_case(kQwen35Channels, 2048, 2048, 4096, T, false, true,
                               7200U + static_cast<std::uint32_t>(T));
    }

    // Argument rejection, on both sides of the route bound.
    failures += split_rejection_case(16);
    failures += split_rejection_case(128);
    failures += packed_rejection_case(16);
    failures += packed_rejection_case(128);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " causal_conv1d_silu\n";
    return failures == 0 ? 0 : 1;
}
