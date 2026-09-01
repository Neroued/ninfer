#include "ninfer/ops/gdn_input_proj.h"

#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

// This criterion belongs to the complete A16 fused projection/conv/snapshot Op.
constexpr ReductionCriterion kGdnInputProjConvSnapshotA16Tolerance{3.15e-3, 4.0e-3, 3.2e-3};
constexpr ReductionCriterion kGdnInputProjConvSnapshotA4Tolerance{0.16, 4.0e-3, 0.16};
constexpr ReductionCriterion kFp8GdnInputProjConvSnapshotA16Tolerance{1.0 / 256.0, 1.0 / 256.0,
                                                                      2.0 / 256.0};
constexpr ReductionCriterion kFp8GdnInputProjConvSnapshotA8Tolerance{0.04, 1.0 / 256.0, 0.06};

constexpr std::int32_t kQueryRows = 2048;
constexpr std::int32_t kKeyRows   = 2048;

std::int32_t snapshot_sample_count(std::int32_t tokens) { return tokens <= 4 ? 32 : 7; }

double silu_fp64(double value) {
    if (value >= 0.0) { return value / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return value * exponential / (1.0 + exponential);
}

std::vector<float> make_conv_weight(std::int32_t channels, std::uint32_t seed) {
    std::vector<float> weight(static_cast<std::size_t>(channels) * 4);
    fill_uniform(weight, seed, -0.02F, 0.02F);
    round_to_bf16(weight);
    return weight;
}

std::vector<std::uint16_t> make_state(std::int32_t channels, std::int32_t slots,
                                      std::int32_t initial_slot, std::uint32_t seed) {
    const std::size_t slot_stride = static_cast<std::size_t>(channels) * 3;
    std::vector<std::uint16_t> state(slot_stride * slots, 0xffffU);
    std::vector<float> initial(slot_stride);
    fill_uniform(initial, seed, -0.05F, 0.05F);
    round_to_bf16(initial);
    const std::size_t initial_base = static_cast<std::size_t>(initial_slot) * slot_stride;
    for (std::size_t index = 0; index < initial.size(); ++index) {
        state[initial_base + index] = f32_to_bf16(initial[index]);
    }
    return state;
}

struct SnapshotOracle {
    std::vector<double> query;
    std::vector<double> key;
    std::vector<double> value;
    std::vector<double> state;
};

template <class Projection>
SnapshotOracle
snapshot_oracle(std::int32_t value_rows, std::int32_t tokens, const std::vector<float>& conv_weight,
                std::span<const std::uint16_t> initial_state, Projection&& projection) {
    const std::int32_t channels = kQueryRows + kKeyRows + value_rows;
    SnapshotOracle oracle;
    const std::int32_t sample_count                = snapshot_sample_count(tokens);
    const std::vector<std::int32_t> query_rows     = sampled_rows(kQueryRows, sample_count);
    const std::vector<std::int32_t> key_rows       = sampled_rows(kKeyRows, sample_count);
    const std::vector<std::int32_t> value_selected = sampled_rows(value_rows, sample_count);
    const std::size_t sampled_channel_count =
        query_rows.size() + key_rows.size() + value_selected.size();
    oracle.query.reserve(query_rows.size() * static_cast<std::size_t>(tokens));
    oracle.key.reserve(key_rows.size() * static_cast<std::size_t>(tokens));
    oracle.value.reserve(value_selected.size() * static_cast<std::size_t>(tokens));
    oracle.state.reserve(sampled_channel_count * 3 * static_cast<std::size_t>(tokens));

    const auto evaluate = [&](std::int32_t global_row, std::vector<double>& output) {
        double state0        = bf16_to_f32(initial_state[global_row]);
        double state1        = bf16_to_f32(initial_state[channels + global_row]);
        double state2        = bf16_to_f32(initial_state[2 * channels + global_row]);
        const double weight0 = conv_weight[global_row];
        const double weight1 = conv_weight[channels + global_row];
        const double weight2 = conv_weight[2 * channels + global_row];
        const double weight3 = conv_weight[3 * channels + global_row];
        for (std::int32_t token = 0; token < tokens; ++token) {
            const double projected = projection(global_row, token);
            const double convolved =
                weight0 * state0 + weight1 * state1 + weight2 * state2 + weight3 * projected;
            output.push_back(silu_fp64(convolved));
            oracle.state.push_back(state1);
            oracle.state.push_back(state2);
            oracle.state.push_back(projected);
            state0 = state1;
            state1 = state2;
            state2 = projected;
        }
    };

    for (const std::int32_t row : query_rows) { evaluate(row, oracle.query); }
    for (const std::int32_t row : key_rows) { evaluate(kQueryRows + row, oracle.key); }
    for (const std::int32_t row : value_selected) {
        evaluate(kQueryRows + kKeyRows + row, oracle.value);
    }
    return oracle;
}

std::vector<double> gather_state(const std::vector<std::uint16_t>& full, std::int32_t channels,
                                 std::int32_t value_rows, std::int32_t tokens,
                                 std::int32_t snapshot_base_slot) {
    std::vector<double> gathered;
    const auto append = [&](std::int32_t global_row) {
        for (std::int32_t token = 0; token < tokens; ++token) {
            const std::size_t token_base =
                static_cast<std::size_t>(snapshot_base_slot + token) * 3 * channels;
            for (std::int32_t history = 0; history < 3; ++history) {
                gathered.push_back(bf16_to_f32(
                    full[token_base + static_cast<std::size_t>(history) * channels + global_row]));
            }
        }
    };
    const std::int32_t sample_count = snapshot_sample_count(tokens);
    for (const std::int32_t row : sampled_rows(kQueryRows, sample_count)) { append(row); }
    for (const std::int32_t row : sampled_rows(kKeyRows, sample_count)) {
        append(kQueryRows + row);
    }
    for (const std::int32_t row : sampled_rows(value_rows, sample_count)) {
        append(kQueryRows + kKeyRows + row);
    }
    return gathered;
}

int verify_state_effects(std::string_view label, const std::vector<std::uint16_t>& before,
                         const std::vector<std::uint16_t>& after, std::int32_t channels,
                         std::int32_t tokens, std::int32_t slots, std::int32_t snapshot_base_slot) {
    const std::size_t slot_stride = static_cast<std::size_t>(channels) * 3;
    for (std::int32_t slot = snapshot_base_slot; slot < snapshot_base_slot + tokens; ++slot) {
        const std::size_t base = static_cast<std::size_t>(slot) * slot_stride;
        for (std::size_t index = 0; index < slot_stride; ++index) {
            if (!std::isfinite(bf16_to_f32(after[base + index]))) {
                std::cerr << label << ": state slot " << slot << " was not fully written\n";
                return 1;
            }
        }
    }
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        if (slot >= snapshot_base_slot && slot < snapshot_base_slot + tokens) { continue; }
        const std::size_t base = static_cast<std::size_t>(slot) * slot_stride;
        if (!std::equal(before.begin() + static_cast<std::ptrdiff_t>(base),
                        before.begin() + static_cast<std::ptrdiff_t>(base + slot_stride),
                        after.begin() + static_cast<std::ptrdiff_t>(base))) {
            std::cerr << label << ": state slot " << slot << " was modified\n";
            return 1;
        }
    }
    return 0;
}

int verify_snapshot_outputs(
    std::string_view suffix, const GuardedBf16Tensor& query, const GuardedBf16Tensor& key,
    const GuardedBf16Tensor& value, std::int32_t value_rows, std::int32_t tokens,
    const SnapshotOracle& oracle,
    const ReductionCriterion& criterion = kGdnInputProjConvSnapshotA16Tolerance) {
    int failures                    = 0;
    const std::int32_t sample_count = snapshot_sample_count(tokens);
    failures += query.verify_guards("snapshot query" + std::string(suffix));
    failures += key.verify_guards("snapshot key" + std::string(suffix));
    failures += value.verify_guards("snapshot value" + std::string(suffix));
    failures += query.verify_fully_written("snapshot query" + std::string(suffix));
    failures += key.verify_fully_written("snapshot key" + std::string(suffix));
    failures += value.verify_fully_written("snapshot value" + std::string(suffix));
    failures +=
        compare("snapshot query" + std::string(suffix),
                gather_rows(query.values(), kQueryRows, 0, kQueryRows, tokens, sample_count),
                oracle.query, criterion);
    failures += compare("snapshot key" + std::string(suffix),
                        gather_rows(key.values(), kKeyRows, 0, kKeyRows, tokens, sample_count),
                        oracle.key, criterion);
    failures +=
        compare("snapshot value" + std::string(suffix),
                gather_rows(value.values(), value_rows, 0, value_rows, tokens, sample_count),
                oracle.value, criterion);
    return failures;
}

std::vector<double> gather_batch_rows(const std::vector<double>& full, std::int32_t full_rows,
                                      std::int32_t row_count, std::int32_t width,
                                      std::int32_t batch_row, std::int32_t valid,
                                      std::int32_t sample_count) {
    std::vector<double> gathered;
    const std::vector<std::int32_t> selected = sampled_rows(row_count, sample_count);
    gathered.reserve(selected.size() * static_cast<std::size_t>(valid));
    for (const std::int32_t row : selected) {
        for (std::int32_t column = 0; column < valid; ++column) {
            const std::int32_t flat_column = batch_row * width + column;
            gathered.push_back(full[static_cast<std::size_t>(flat_column) * full_rows + row]);
        }
    }
    return gathered;
}

int verify_zero_tail(std::string_view label, const GuardedBf16Tensor& output, std::int32_t rows,
                     std::int32_t width, std::int32_t batch,
                     const std::vector<std::int32_t>& valid_columns) {
    if (valid_columns.empty()) { return 0; }
    const std::vector<std::uint16_t> bits = output.bits();
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        for (std::int32_t column = valid_columns[static_cast<std::size_t>(batch_row)];
             column < width; ++column) {
            const std::size_t base = static_cast<std::size_t>(batch_row * width + column) * rows;
            for (std::int32_t row = 0; row < rows; ++row) {
                if (bits[base + static_cast<std::size_t>(row)] != 0) {
                    std::cerr << label << ": invalid tail was not exact zero\n";
                    return 1;
                }
            }
        }
    }
    return 0;
}

int verify_batched_state_effects(std::string_view label, const std::vector<std::uint16_t>& before,
                                 const std::vector<std::uint16_t>& after, std::int32_t channels,
                                 std::int32_t slots, std::int32_t width, std::int32_t batch,
                                 const std::vector<std::int32_t>& valid_columns,
                                 const std::vector<std::int32_t>& snapshot_bases) {
    const std::size_t slot_stride = static_cast<std::size_t>(channels) * 3;
    std::vector<bool> written(static_cast<std::size_t>(slots), false);
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        const std::int32_t valid =
            valid_columns.empty() ? width : valid_columns[static_cast<std::size_t>(batch_row)];
        for (std::int32_t column = 0; column < valid; ++column) {
            written[static_cast<std::size_t>(snapshot_bases[static_cast<std::size_t>(batch_row)] +
                                             column)] = true;
        }
    }
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        const std::size_t base = static_cast<std::size_t>(slot) * slot_stride;
        if (!written[static_cast<std::size_t>(slot)]) {
            if (!std::equal(before.begin() + static_cast<std::ptrdiff_t>(base),
                            before.begin() + static_cast<std::ptrdiff_t>(base + slot_stride),
                            after.begin() + static_cast<std::ptrdiff_t>(base))) {
                std::cerr << label << ": unwritten state slot " << slot << " was modified\n";
                return 1;
            }
            continue;
        }
        for (std::size_t index = 0; index < slot_stride; ++index) {
            if (!std::isfinite(bf16_to_f32(after[base + index]))) {
                std::cerr << label << ": state slot " << slot << " was not fully written\n";
                return 1;
            }
        }
    }
    return 0;
}

template <class ProjectQkv, class ProjectZ, class Launch>
int run_batched_case(std::string_view label, std::int32_t hidden, std::int32_t value_rows,
                     std::int32_t z_rows, std::int32_t width, std::int32_t batch,
                     std::vector<std::int32_t> valid_columns, const std::vector<float>& conv_weight,
                     std::size_t workspace_bytes, const ReductionCriterion& criterion,
                     ProjectQkv&& project_qkv, ProjectZ&& project_z, Launch&& launch) {
    const std::int32_t channels          = kQueryRows + kKeyRows + value_rows;
    const std::int32_t aggregate_columns = width * batch;
    const std::int32_t slots             = aggregate_columns + batch;
    const std::size_t slot_stride        = static_cast<std::size_t>(channels) * 3;
    const std::vector<float> activation =
        make_bf16_activation(hidden, aggregate_columns, 1103U + width + batch);
    const std::vector<std::uint16_t> activation_bits  = bf16_bits(activation);
    const std::vector<std::uint16_t> conv_weight_bits = bf16_bits(conv_weight);

    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> snapshot_bases(static_cast<std::size_t>(batch));
    std::vector<std::uint16_t> state_before(slot_stride * slots, 0xffffU);
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        snapshot_bases[static_cast<std::size_t>(batch_row)] = batch_row * width;
        initial_slots[static_cast<std::size_t>(batch_row)] =
            batch_row == 0 ? 0 : aggregate_columns + batch_row;
        std::vector<float> initial(slot_stride);
        fill_uniform(initial, 1201U + static_cast<std::uint32_t>(batch_row), -0.05F, 0.05F);
        round_to_bf16(initial);
        const std::size_t base =
            static_cast<std::size_t>(initial_slots[static_cast<std::size_t>(batch_row)]) *
            slot_stride;
        for (std::size_t index = 0; index < slot_stride; ++index) {
            state_before[base + index] = f32_to_bf16(initial[index]);
        }
    }

    DeviceBuffer device_activation    = to_device(activation_bits);
    DeviceBuffer device_conv_weight   = to_device(conv_weight_bits);
    DeviceBuffer device_initial       = to_device(initial_slots);
    DeviceBuffer device_snapshot_base = to_device(snapshot_bases);
    DeviceBuffer device_valid;
    if (!valid_columns.empty()) { device_valid = to_device(valid_columns); }
    GuardedBf16Tensor state(channels * 3, slots);
    state.copy_from_bits(state_before);
    GuardedBf16Tensor query(kQueryRows, aggregate_columns);
    GuardedBf16Tensor key(kKeyRows, aggregate_columns);
    GuardedBf16Tensor value(value_rows, aggregate_columns);
    GuardedBf16Tensor z(z_rows, aggregate_columns);

    Tensor x(device_activation.p, DType::BF16, {hidden, width, batch});
    Tensor conv(device_conv_weight.p, DType::BF16, {channels, 4});
    Tensor conv_state(state.data(), DType::BF16, {channels, 3, slots});
    Tensor valid;
    if (!valid_columns.empty()) { valid = Tensor(device_valid.p, DType::I32, {batch}); }
    Tensor initial(device_initial.p, DType::I32, {batch});
    Tensor snapshot_base(device_snapshot_base.p, DType::I32, {batch});
    Tensor q(query.data(), DType::BF16, {kQueryRows, width, batch});
    Tensor k(key.data(), DType::BF16, {kKeyRows, width, batch});
    Tensor v(value.data(), DType::BF16, {value_rows, width, batch});
    Tensor z_output(z.data(), DType::BF16, {z_rows, width, batch});
    WorkspaceArena workspace(std::max<std::size_t>(1, workspace_bytes));

    launch(x, conv, conv_state, valid, initial, snapshot_base, q, k, v, z_output, workspace);
    cuda_synchronize();

    int failures                                 = 0;
    const std::vector<double> query_values       = query.values();
    const std::vector<double> key_values         = key.values();
    const std::vector<double> value_values       = value.values();
    const std::vector<double> z_values           = z.values();
    const std::vector<std::uint16_t> state_after = state.bits();
    std::vector<double> query_actual;
    std::vector<double> query_expected;
    std::vector<double> key_actual;
    std::vector<double> key_expected;
    std::vector<double> value_actual;
    std::vector<double> value_expected;
    std::vector<double> state_actual;
    std::vector<double> state_expected;
    std::vector<double> z_actual;
    std::vector<double> z_expected;
    const auto append = [](std::vector<double>& destination, std::vector<double> source) {
        destination.insert(destination.end(), source.begin(), source.end());
    };
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        const std::int32_t valid_extent =
            valid_columns.empty() ? width : valid_columns[static_cast<std::size_t>(batch_row)];
        const std::size_t initial_base =
            static_cast<std::size_t>(initial_slots[static_cast<std::size_t>(batch_row)]) *
            slot_stride;
        const std::span<const std::uint16_t> initial_state(state_before.data() + initial_base,
                                                           slot_stride);
        const SnapshotOracle oracle =
            snapshot_oracle(value_rows, valid_extent, conv_weight, initial_state,
                            [&](std::int32_t row, std::int32_t column) {
                                return project_qkv(row, batch_row * width + column, activation);
                            });
        const std::int32_t sample_count = snapshot_sample_count(valid_extent);
        append(query_actual, gather_batch_rows(query_values, kQueryRows, kQueryRows, width,
                                               batch_row, valid_extent, sample_count));
        append(query_expected, oracle.query);
        append(key_actual, gather_batch_rows(key_values, kKeyRows, kKeyRows, width, batch_row,
                                             valid_extent, sample_count));
        append(key_expected, oracle.key);
        append(value_actual, gather_batch_rows(value_values, value_rows, value_rows, width,
                                               batch_row, valid_extent, sample_count));
        append(value_expected, oracle.value);
        append(state_actual, gather_state(state_after, channels, value_rows, valid_extent,
                                          snapshot_bases[static_cast<std::size_t>(batch_row)]));
        append(state_expected, oracle.state);

        std::vector<double> row_z_expected;
        for (const std::int32_t row : sampled_rows(z_rows, sample_count)) {
            for (std::int32_t column = 0; column < width; ++column) {
                row_z_expected.push_back(project_z(row, batch_row * width + column, activation));
            }
        }
        append(z_actual,
               gather_batch_rows(z_values, z_rows, z_rows, width, batch_row, width, sample_count));
        append(z_expected, std::move(row_z_expected));
    }
    failures += compare(std::string(label) + " query", query_actual, query_expected, criterion);
    failures += compare(std::string(label) + " key", key_actual, key_expected, criterion);
    failures += compare(std::string(label) + " value", value_actual, value_expected, criterion);
    failures += compare(std::string(label) + " state", state_actual, state_expected, criterion);
    failures += compare(std::string(label) + " z", z_actual, z_expected, criterion);

    failures += query.verify_guards(std::string(label) + " query");
    failures += key.verify_guards(std::string(label) + " key");
    failures += value.verify_guards(std::string(label) + " value");
    failures += z.verify_guards(std::string(label) + " z");
    failures += state.verify_guards(std::string(label) + " state");
    failures += query.verify_fully_written(std::string(label) + " query");
    failures += key.verify_fully_written(std::string(label) + " key");
    failures += value.verify_fully_written(std::string(label) + " value");
    failures += z.verify_fully_written(std::string(label) + " z");
    failures += verify_zero_tail(std::string(label) + " query", query, kQueryRows, width, batch,
                                 valid_columns);
    failures +=
        verify_zero_tail(std::string(label) + " key", key, kKeyRows, width, batch, valid_columns);
    failures += verify_zero_tail(std::string(label) + " value", value, value_rows, width, batch,
                                 valid_columns);
    failures += verify_batched_state_effects(label, state_before, state_after, channels, slots,
                                             width, batch, valid_columns, snapshot_bases);
    failures += verify_preserved(std::string(label) + " x", device_activation, activation_bits);
    failures +=
        verify_preserved(std::string(label) + " conv weight", device_conv_weight, conv_weight_bits);
    failures +=
        verify_preserved(std::string(label) + " initial slots", device_initial, initial_slots);
    failures += verify_preserved(std::string(label) + " snapshot bases", device_snapshot_base,
                                 snapshot_bases);
    if (!valid_columns.empty()) {
        failures +=
            verify_preserved(std::string(label) + " valid columns", device_valid, valid_columns);
    }
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_q4_q5_case(DevicePackedWeight& query_key, DevicePackedWeight& value_z_weight,
                   std::int32_t tokens, std::int32_t initial_slot) {
    constexpr std::int32_t kHidden           = 5120;
    constexpr std::int32_t kValueRows        = 6144;
    constexpr std::int32_t kZRows            = 6144;
    constexpr std::int32_t kChannels         = 10240;
    constexpr std::int32_t kSnapshotBaseSlot = 1;
    const std::int32_t slots                 = std::max(tokens + 2, initial_slot + 1);
    const std::vector<float> activation      = make_bf16_activation(kHidden, tokens, 601U + tokens);
    const std::vector<std::uint16_t> activation_bits  = bf16_bits(activation);
    const std::vector<float> conv_weight              = make_conv_weight(kChannels, 607U);
    const std::vector<std::uint16_t> conv_weight_bits = bf16_bits(conv_weight);
    const std::vector<std::uint16_t> state_before =
        make_state(kChannels, slots, initial_slot, 613U + tokens);
    const std::vector<std::int32_t> initial_value{initial_slot};
    const std::vector<std::int32_t> snapshot_base_value{kSnapshotBaseSlot};

    DeviceBuffer device_activation    = to_device(activation_bits);
    DeviceBuffer device_conv_weight   = to_device(conv_weight_bits);
    DeviceBuffer device_initial       = to_device(initial_value);
    DeviceBuffer device_snapshot_base = to_device(snapshot_base_value);
    GuardedBf16Tensor state(kChannels * 3, slots);
    state.copy_from_bits(state_before);
    GuardedBf16Tensor query(kQueryRows, tokens);
    GuardedBf16Tensor key(kKeyRows, tokens);
    GuardedBf16Tensor value(kValueRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor conv(device_conv_weight.p, DType::BF16, {kChannels, 4});
    Tensor conv_state(state.data(), DType::BF16, {kChannels, 3, slots});
    Tensor initial(device_initial.p, DType::I32, {1});
    Tensor snapshot_base(device_snapshot_base.p, DType::I32, {1});
    Tensor q                          = query.tensor();
    Tensor k                          = key.tensor();
    Tensor v                          = value.tensor();
    Tensor z_output                   = z.tensor();
    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, 1, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(1, workspace_bytes));

    ops::gdn_input_proj_conv_snapshot(x, query_key.view(), value_z_weight.view(), conv, conv_state,
                                      Tensor{}, initial, snapshot_base, q, k, v, z_output,
                                      workspace, nullptr);
    cuda_synchronize();

    const std::size_t initial_base = static_cast<std::size_t>(initial_slot) * 3 * kChannels;
    const std::span<const std::uint16_t> initial_state(state_before.data() + initial_base,
                                                       3 * kChannels);
    const SnapshotOracle oracle = snapshot_oracle(
        kValueRows, tokens, conv_weight, initial_state, [&](std::int32_t row, std::int32_t token) {
            const float* token_activation =
                activation.data() + static_cast<std::size_t>(token) * kHidden;
            if (row < kQueryRows + kKeyRows) {
                return quantized_weight::dot_fp64(query_key.host, row, token_activation, kHidden);
            }
            return quantized_weight::dot_fp64(value_z_weight.host, row - kQueryRows - kKeyRows,
                                              token_activation, kHidden);
        });
    const std::vector<std::uint16_t> state_after = state.bits();
    const std::string suffix                     = " Q4/Q5 A16 T=" + std::to_string(tokens) +
                               " initial=" + std::to_string(initial_slot) +
                               " base=" + std::to_string(kSnapshotBaseSlot);
    int failures = verify_snapshot_outputs(suffix, query, key, value, kValueRows, tokens, oracle);
    failures += compare("snapshot state" + suffix,
                        gather_state(state_after, kChannels, kValueRows, tokens, kSnapshotBaseSlot),
                        oracle.state, kGdnInputProjConvSnapshotA16Tolerance);
    failures += state.verify_guards("snapshot state" + suffix);
    failures += verify_state_effects("snapshot state" + suffix, state_before, state_after,
                                     kChannels, tokens, slots, kSnapshotBaseSlot);
    failures += z.verify_guards("snapshot z" + suffix);
    failures += z.verify_fully_written("snapshot z" + suffix);
    failures += compare(
        "snapshot z" + suffix, gather_rows(z.values(), kZRows, 0, kZRows, tokens),
        projection_oracle(value_z_weight.host, kValueRows, kZRows, activation, kHidden, tokens),
        kGdnInputProjConvSnapshotA16Tolerance);
    failures += verify_preserved("snapshot x" + suffix, device_activation, activation_bits);
    failures +=
        verify_preserved("snapshot conv weight" + suffix, device_conv_weight, conv_weight_bits);
    failures += verify_preserved("snapshot initial slot" + suffix, device_initial, initial_value);
    failures +=
        verify_preserved("snapshot base slot" + suffix, device_snapshot_base, snapshot_base_value);
    failures += query_key.verify_preserved("snapshot query/key weight" + suffix);
    failures += value_z_weight.verify_preserved("snapshot value/z weight" + suffix);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "snapshot" << suffix << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_q4_q5() {
    constexpr std::int32_t kHidden = 5120;
    DevicePackedWeight query_key(
        quantized_weight::make_patterned_weight(QType::Q4G64_F16S, 4096, kHidden, 617U));
    DevicePackedWeight value_z_weight(
        quantized_weight::make_patterned_weight(QType::Q5G64_F16S, 12288, kHidden, 619U));
    int failures = 0;
    // Cover every fixed Small-T specialization plus the first composed extent.
    for (const std::int32_t tokens : {1, 2, 3, 4, 5, 6, 7}) {
        const std::int32_t initial_slot = tokens == 5 ? 0 : tokens + 1;
        failures += run_q4_q5_case(query_key, value_z_weight, tokens, initial_slot);
    }
    constexpr std::int32_t kValueRows    = 6144;
    constexpr std::int32_t kZRows        = 6144;
    constexpr std::int32_t kChannels     = 10240;
    const std::vector<float> conv_weight = make_conv_weight(kChannels, 631U);
    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, 8, 1, 1);
    failures += run_batched_case(
        "Q4/Q5 A16 B=8 W=1", kHidden, kValueRows, kZRows, 1, 8, {}, conv_weight, workspace_bytes,
        kGdnInputProjConvSnapshotA16Tolerance,
        [&](std::int32_t row, std::int32_t flat_column, const std::vector<float>& activation) {
            const float* column =
                activation.data() + static_cast<std::size_t>(flat_column) * kHidden;
            if (row < kQueryRows + kKeyRows) {
                return quantized_weight::dot_fp64(query_key.host, row, column, kHidden);
            }
            return quantized_weight::dot_fp64(value_z_weight.host, row - kQueryRows - kKeyRows,
                                              column, kHidden);
        },
        [&](std::int32_t row, std::int32_t flat_column, const std::vector<float>& activation) {
            return quantized_weight::dot_fp64(
                value_z_weight.host, kValueRows + row,
                activation.data() + static_cast<std::size_t>(flat_column) * kHidden, kHidden);
        },
        [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid,
            const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
            Tensor& z, WorkspaceArena& workspace) {
            ops::gdn_input_proj_conv_snapshot(x, query_key.view(), value_z_weight.view(), conv,
                                              state, valid, initial, snapshot_base, q, k, v, z,
                                              workspace, nullptr);
        });
    failures += query_key.verify_preserved("batched Q4/Q5 query/key weight");
    failures += value_z_weight.verify_preserved("batched Q4/Q5 value/z weight");
    return failures;
}

int run_w8_case(DevicePackedWeight& parent, std::int32_t tokens, std::int32_t initial_slot) {
    constexpr std::int32_t kHidden           = 2048;
    constexpr std::int32_t kValueRows        = 4096;
    constexpr std::int32_t kZRows            = 4096;
    constexpr std::int32_t kChannels         = 8192;
    constexpr std::int32_t kSnapshotBaseSlot = 1;
    const std::int32_t slots                 = std::max(tokens + 2, initial_slot + 1);
    const std::vector<float> activation      = make_bf16_activation(kHidden, tokens, 701U + tokens);
    const std::vector<std::uint16_t> activation_bits  = bf16_bits(activation);
    const std::vector<float> conv_weight              = make_conv_weight(kChannels, 709U);
    const std::vector<std::uint16_t> conv_weight_bits = bf16_bits(conv_weight);
    const std::vector<std::uint16_t> state_before =
        make_state(kChannels, slots, initial_slot, 719U + tokens);
    const std::vector<std::int32_t> initial_value{initial_slot};
    const std::vector<std::int32_t> snapshot_base_value{kSnapshotBaseSlot};

    DeviceBuffer device_activation    = to_device(activation_bits);
    DeviceBuffer device_conv_weight   = to_device(conv_weight_bits);
    DeviceBuffer device_initial       = to_device(initial_value);
    DeviceBuffer device_snapshot_base = to_device(snapshot_base_value);
    GuardedBf16Tensor state(kChannels * 3, slots);
    state.copy_from_bits(state_before);
    GuardedBf16Tensor query(kQueryRows, tokens);
    GuardedBf16Tensor key(kKeyRows, tokens);
    GuardedBf16Tensor value(kValueRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor conv(device_conv_weight.p, DType::BF16, {kChannels, 4});
    Tensor conv_state(state.data(), DType::BF16, {kChannels, 3, slots});
    Tensor initial(device_initial.p, DType::I32, {1});
    Tensor snapshot_base(device_snapshot_base.p, DType::I32, {1});
    Tensor q                          = query.tensor();
    Tensor k                          = key.tensor();
    Tensor v                          = value.tensor();
    Tensor z_output                   = z.tensor();
    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, 1, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(1, workspace_bytes));

    ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, conv_state, Tensor{}, initial,
                                      snapshot_base, q, k, v, z_output, workspace, nullptr);
    cuda_synchronize();

    const std::size_t initial_base = static_cast<std::size_t>(initial_slot) * 3 * kChannels;
    const std::span<const std::uint16_t> initial_state(state_before.data() + initial_base,
                                                       3 * kChannels);
    const SnapshotOracle oracle = snapshot_oracle(
        kValueRows, tokens, conv_weight, initial_state, [&](std::int32_t row, std::int32_t token) {
            return quantized_weight::dot_fp64(
                parent.host, row, activation.data() + static_cast<std::size_t>(token) * kHidden,
                kHidden);
        });
    const std::vector<std::uint16_t> state_after = state.bits();
    const std::string suffix                     = " W8 A16 T=" + std::to_string(tokens) +
                               " initial=" + std::to_string(initial_slot) +
                               " base=" + std::to_string(kSnapshotBaseSlot);
    int failures = verify_snapshot_outputs(suffix, query, key, value, kValueRows, tokens, oracle);
    failures += compare("snapshot state" + suffix,
                        gather_state(state_after, kChannels, kValueRows, tokens, kSnapshotBaseSlot),
                        oracle.state, kGdnInputProjConvSnapshotA16Tolerance);
    failures += state.verify_guards("snapshot state" + suffix);
    failures += verify_state_effects("snapshot state" + suffix, state_before, state_after,
                                     kChannels, tokens, slots, kSnapshotBaseSlot);
    failures += z.verify_guards("snapshot z" + suffix);
    failures += z.verify_fully_written("snapshot z" + suffix);
    failures +=
        compare("snapshot z" + suffix, gather_rows(z.values(), kZRows, 0, kZRows, tokens),
                projection_oracle(parent.host, kChannels, kZRows, activation, kHidden, tokens),
                kGdnInputProjConvSnapshotA16Tolerance);
    failures += verify_preserved("snapshot x" + suffix, device_activation, activation_bits);
    failures +=
        verify_preserved("snapshot conv weight" + suffix, device_conv_weight, conv_weight_bits);
    failures += verify_preserved("snapshot initial slot" + suffix, device_initial, initial_value);
    failures +=
        verify_preserved("snapshot base slot" + suffix, device_snapshot_base, snapshot_base_value);
    failures += parent.verify_preserved("snapshot parent weight" + suffix);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "snapshot" << suffix << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_w8() {
    constexpr std::int32_t kHidden = 2048;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::W8G32_F16S, 12288, kHidden, 727U));
    int failures = 0;
    // Representative registered T values around every current execution boundary.
    for (const std::int32_t tokens : {1, 2, 17}) {
        const std::int32_t initial_slot = tokens == 2 ? 0 : tokens + 1;
        failures += run_w8_case(parent, tokens, initial_slot);
    }
    constexpr std::int32_t kValueRows = 4096;
    constexpr std::int32_t kZRows     = 4096;
    constexpr std::int32_t kChannels  = 8192;
    constexpr std::int32_t kWidth     = 16;
    constexpr std::int32_t kBatch     = 2;
    const std::vector<std::int32_t> valid_columns{16, 7};
    const std::vector<float> conv_weight = make_conv_weight(kChannels, 733U);
    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, kBatch, kWidth, kWidth);
    failures += run_batched_case(
        "W8 A16 B=2 W=16 masked", kHidden, kValueRows, kZRows, kWidth, kBatch, valid_columns,
        conv_weight, workspace_bytes, kGdnInputProjConvSnapshotA16Tolerance,
        [&](std::int32_t row, std::int32_t flat_column, const std::vector<float>& activation) {
            return quantized_weight::dot_fp64(
                parent.host, row,
                activation.data() + static_cast<std::size_t>(flat_column) * kHidden, kHidden);
        },
        [&](std::int32_t row, std::int32_t flat_column, const std::vector<float>& activation) {
            return quantized_weight::dot_fp64(
                parent.host, kChannels + row,
                activation.data() + static_cast<std::size_t>(flat_column) * kHidden, kHidden);
        },
        [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid,
            const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
            Tensor& z, WorkspaceArena& workspace) {
            ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, state, valid, initial,
                                              snapshot_base, q, k, v, z, workspace, nullptr);
        });
    failures += parent.verify_preserved("batched W8 parent weight");
    return failures;
}

int run_nvfp4_case(DevicePackedWeight& parent, std::int32_t tokens, ops::LinearPolicy policy,
                   std::int32_t initial_slot) {
    constexpr std::int32_t kHidden           = 5120;
    constexpr std::int32_t kValueRows        = 6144;
    constexpr std::int32_t kZRows            = 6144;
    constexpr std::int32_t kChannels         = 10240;
    constexpr std::int32_t kRows             = kChannels + kZRows;
    constexpr std::int32_t kSnapshotBaseSlot = 1;
    const std::int32_t slots                 = std::max(tokens + 2, initial_slot + 1);
    const std::vector<float> activation =
        make_bf16_activation(kHidden, tokens, 809U + static_cast<std::uint32_t>(tokens));
    const std::vector<std::uint16_t> activation_bits  = bf16_bits(activation);
    const std::vector<float> conv_weight              = make_conv_weight(kChannels, 811U);
    const std::vector<std::uint16_t> conv_weight_bits = bf16_bits(conv_weight);
    const std::vector<std::uint16_t> state_before =
        make_state(kChannels, slots, initial_slot, 821U + static_cast<std::uint32_t>(tokens));
    const std::vector<std::int32_t> initial_value{initial_slot};
    const std::vector<std::int32_t> snapshot_base_value{kSnapshotBaseSlot};

    DeviceBuffer device_activation    = to_device(activation_bits);
    DeviceBuffer device_conv_weight   = to_device(conv_weight_bits);
    DeviceBuffer device_initial       = to_device(initial_value);
    DeviceBuffer device_snapshot_base = to_device(snapshot_base_value);
    GuardedBf16Tensor state(kChannels * 3, slots);
    state.copy_from_bits(state_before);
    GuardedBf16Tensor query(kQueryRows, tokens);
    GuardedBf16Tensor key(kKeyRows, tokens);
    GuardedBf16Tensor value(kValueRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor conv(device_conv_weight.p, DType::BF16, {kChannels, 4});
    Tensor conv_state(state.data(), DType::BF16, {kChannels, 3, slots});
    Tensor initial(device_initial.p, DType::I32, {1});
    Tensor snapshot_base(device_snapshot_base.p, DType::I32, {1});
    Tensor q                          = query.tensor();
    Tensor k                          = key.tensor();
    Tensor v                          = value.tensor();
    Tensor z_output                   = z.tensor();
    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        QType::NVFP4, kRows, kHidden, policy, 1, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, conv_state, Tensor{}, initial,
                                      snapshot_base, q, k, v, z_output, policy, workspace, nullptr);
    cuda_synchronize();

    const std::size_t initial_base = static_cast<std::size_t>(initial_slot) * 3 * kChannels;
    const std::span<const std::uint16_t> initial_state(state_before.data() + initial_base,
                                                       3 * kChannels);
    const SnapshotOracle oracle = snapshot_oracle(
        kValueRows, tokens, conv_weight, initial_state, [&](std::int32_t row, std::int32_t token) {
            return quantized_weight::dot_fp64(
                parent.host, row, activation.data() + static_cast<std::size_t>(token) * kHidden,
                kHidden);
        });
    const bool a4 = policy == ops::LinearPolicy::AllowA4 && tokens >= 4;
    const ReductionCriterion& criterion =
        a4 ? kGdnInputProjConvSnapshotA4Tolerance : kGdnInputProjConvSnapshotA16Tolerance;
    const std::string suffix =
        std::string(" NVFP4 ") + (a4 ? "A4" : "A16") + " T=" + std::to_string(tokens) +
        " initial=" + std::to_string(initial_slot) + " base=" + std::to_string(kSnapshotBaseSlot);
    const std::vector<std::uint16_t> state_after = state.bits();
    int failures =
        verify_snapshot_outputs(suffix, query, key, value, kValueRows, tokens, oracle, criterion);
    failures += compare("snapshot state" + suffix,
                        gather_state(state_after, kChannels, kValueRows, tokens, kSnapshotBaseSlot),
                        oracle.state, criterion);
    failures += state.verify_guards("snapshot state" + suffix);
    failures += verify_state_effects("snapshot state" + suffix, state_before, state_after,
                                     kChannels, tokens, slots, kSnapshotBaseSlot);
    failures += z.verify_guards("snapshot z" + suffix);
    failures += z.verify_fully_written("snapshot z" + suffix);
    failures +=
        compare("snapshot z" + suffix,
                gather_rows(z.values(), kZRows, 0, kZRows, tokens, snapshot_sample_count(tokens)),
                projection_oracle(parent.host, kChannels, kZRows, activation, kHidden, tokens,
                                  snapshot_sample_count(tokens)),
                criterion);
    failures += verify_preserved("snapshot x" + suffix, device_activation, activation_bits);
    failures +=
        verify_preserved("snapshot conv weight" + suffix, device_conv_weight, conv_weight_bits);
    failures += verify_preserved("snapshot initial slot" + suffix, device_initial, initial_value);
    failures +=
        verify_preserved("snapshot base slot" + suffix, device_snapshot_base, snapshot_base_value);
    failures += parent.verify_preserved("snapshot parent weight" + suffix);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "snapshot" << suffix << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_nvfp4() {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kRows   = 16384;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kRows, kHidden, 823U, options));

    int failures = 0;
    failures += run_nvfp4_case(parent, 1, ops::LinearPolicy::A16Only, 2);
#ifndef NINFER_VOLTA_BUILD
    failures += run_nvfp4_case(parent, 3, ops::LinearPolicy::AllowA4, 4);
    failures += run_nvfp4_case(parent, 4, ops::LinearPolicy::AllowA4, 5);
    failures += run_nvfp4_case(parent, 17, ops::LinearPolicy::AllowA4, 0);
    failures += run_nvfp4_case(parent, 1024, ops::LinearPolicy::AllowA4, 1025);
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kChannels  = 10240;
    constexpr std::int32_t kWidth     = 6;
    constexpr std::int32_t kBatch     = 3;
    const std::vector<std::int32_t> valid_columns{6, 3, 1};
    const std::vector<float> conv_weight = make_conv_weight(kChannels, 829U);
    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        QType::NVFP4, kRows, kHidden, ops::LinearPolicy::AllowA4, kBatch, kWidth, kWidth);
    failures += run_batched_case(
        "NVFP4 A4 B=3 W=6 masked", kHidden, kValueRows, kZRows, kWidth, kBatch, valid_columns,
        conv_weight, workspace_bytes, kGdnInputProjConvSnapshotA4Tolerance,
        [&](std::int32_t row, std::int32_t flat_column, const std::vector<float>& activation) {
            return quantized_weight::dot_fp64(
                parent.host, row,
                activation.data() + static_cast<std::size_t>(flat_column) * kHidden, kHidden);
        },
        [&](std::int32_t row, std::int32_t flat_column, const std::vector<float>& activation) {
            return quantized_weight::dot_fp64(
                parent.host, kChannels + row,
                activation.data() + static_cast<std::size_t>(flat_column) * kHidden, kHidden);
        },
        [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid,
            const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
            Tensor& z, WorkspaceArena& workspace) {
            ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, state, valid, initial,
                                              snapshot_base, q, k, v, z, ops::LinearPolicy::AllowA4,
                                              workspace, nullptr);
        });
    failures += parent.verify_preserved("batched NVFP4 parent weight");
#endif
    return failures;
}

int run_fp8_case(DevicePackedWeight& parent, std::int32_t tokens, ops::LinearPolicy policy,
                 std::int32_t initial_slot, bool convenience = false,
                 bool shared_state_selectors = false) {
    constexpr std::int32_t kHidden               = 5120;
    constexpr std::int32_t kValueRows            = 6144;
    constexpr std::int32_t kZRows                = 6144;
    constexpr std::int32_t kChannels             = 10240;
    constexpr std::int32_t kRows                 = kChannels + kZRows;
    constexpr std::int32_t kSeparateSnapshotBase = 1;
    const std::int32_t snapshot_base_slot =
        shared_state_selectors ? initial_slot : kSeparateSnapshotBase;
    const std::int32_t slots = std::max(tokens + snapshot_base_slot + 1, initial_slot + 1);
    const std::vector<float> activation =
        make_bf16_activation(kHidden, tokens, 907U + static_cast<std::uint32_t>(tokens));
    const std::vector<std::uint16_t> activation_bits  = bf16_bits(activation);
    const std::vector<float> conv_weight              = make_conv_weight(kChannels, 911U);
    const std::vector<std::uint16_t> conv_weight_bits = bf16_bits(conv_weight);
    const std::vector<std::uint16_t> state_before =
        make_state(kChannels, slots, initial_slot, 919U + static_cast<std::uint32_t>(tokens));
    const std::vector<std::int32_t> initial_value{initial_slot};
    const std::vector<std::int32_t> snapshot_base_value{snapshot_base_slot};

    DeviceBuffer device_activation    = to_device(activation_bits);
    DeviceBuffer device_conv_weight   = to_device(conv_weight_bits);
    DeviceBuffer device_initial       = to_device(initial_value);
    DeviceBuffer device_snapshot_base = to_device(snapshot_base_value);
    GuardedBf16Tensor state(kChannels * 3, slots);
    state.copy_from_bits(state_before);
    GuardedBf16Tensor query(kQueryRows, tokens);
    GuardedBf16Tensor key(kKeyRows, tokens);
    GuardedBf16Tensor value(kValueRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor conv(device_conv_weight.p, DType::BF16, {kChannels, 4});
    Tensor conv_state(state.data(), DType::BF16, {kChannels, 3, slots});
    Tensor initial(device_initial.p, DType::I32, {1});
    Tensor snapshot_base =
        shared_state_selectors ? initial : Tensor(device_snapshot_base.p, DType::I32, {1});
    Tensor q                          = query.tensor();
    Tensor k                          = key.tensor();
    Tensor v                          = value.tensor();
    Tensor z_output                   = z.tensor();
    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, policy, 1, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    if (convenience) {
        ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, conv_state, Tensor{}, initial,
                                          snapshot_base, q, k, v, z_output, workspace, nullptr);
    } else {
        ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, conv_state, Tensor{}, initial,
                                          snapshot_base, q, k, v, z_output, policy, workspace,
                                          nullptr);
    }
    cuda_synchronize();

    const std::size_t initial_base = static_cast<std::size_t>(initial_slot) * 3 * kChannels;
    const std::span<const std::uint16_t> initial_state(state_before.data() + initial_base,
                                                       3 * kChannels);
    const SnapshotOracle oracle = snapshot_oracle(
        kValueRows, tokens, conv_weight, initial_state, [&](std::int32_t row, std::int32_t token) {
            return quantized_weight::dot_fp64(
                parent.host, row, activation.data() + static_cast<std::size_t>(token) * kHidden,
                kHidden);
        });
    const bool uses_a8                  = policy == ops::LinearPolicy::AllowA8 && tokens >= 10;
    const ReductionCriterion& criterion = uses_a8 ? kFp8GdnInputProjConvSnapshotA8Tolerance
                                                  : kFp8GdnInputProjConvSnapshotA16Tolerance;
    const std::string suffix =
        std::string(" FP8 ") + (uses_a8 ? "A8" : "A16") + " T=" + std::to_string(tokens) +
        " initial=" + std::to_string(initial_slot) + " base=" + std::to_string(snapshot_base_slot);
    const std::vector<std::uint16_t> state_after = state.bits();
    int failures =
        verify_snapshot_outputs(suffix, query, key, value, kValueRows, tokens, oracle, criterion);
    failures +=
        compare("snapshot state" + suffix,
                gather_state(state_after, kChannels, kValueRows, tokens, snapshot_base_slot),
                oracle.state, criterion);
    failures += state.verify_guards("snapshot state" + suffix);
    failures += verify_state_effects("snapshot state" + suffix, state_before, state_after,
                                     kChannels, tokens, slots, snapshot_base_slot);
    failures += z.verify_guards("snapshot z" + suffix);
    failures += z.verify_fully_written("snapshot z" + suffix);
    failures +=
        compare("snapshot z" + suffix,
                gather_rows(z.values(), kZRows, 0, kZRows, tokens, snapshot_sample_count(tokens)),
                projection_oracle(parent.host, kChannels, kZRows, activation, kHidden, tokens,
                                  snapshot_sample_count(tokens)),
                criterion);
    failures += verify_preserved("snapshot x" + suffix, device_activation, activation_bits);
    failures +=
        verify_preserved("snapshot conv weight" + suffix, device_conv_weight, conv_weight_bits);
    failures += verify_preserved("snapshot initial slot" + suffix, device_initial, initial_value);
    failures +=
        verify_preserved("snapshot base slot" + suffix, device_snapshot_base, snapshot_base_value);
    failures += parent.verify_preserved("snapshot parent weight" + suffix);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "snapshot" << suffix << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_fp8() {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kChannels  = 10240;
    constexpr std::int32_t kRows      = kChannels + kZRows;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, 929U));

    int failures = 0;
    failures += run_fp8_case(parent, 1, ops::LinearPolicy::A16Only, 2, true, true);
    failures += run_fp8_case(parent, 3, ops::LinearPolicy::A16Only, 4);
    failures += run_fp8_case(parent, 4, ops::LinearPolicy::A16Only, 5);
    failures += run_fp8_case(parent, 6, ops::LinearPolicy::A16Only, 7);
    failures += run_fp8_case(parent, 7, ops::LinearPolicy::A16Only, 8);
#ifndef NINFER_VOLTA_BUILD
    failures += run_fp8_case(parent, 9, ops::LinearPolicy::AllowA8, 10);
    failures += run_fp8_case(parent, 10, ops::LinearPolicy::AllowA8, 11);
#endif
    failures += run_fp8_case(parent, 10, ops::LinearPolicy::A16Only, 11);
    failures += run_fp8_case(parent, 11, ops::LinearPolicy::A16Only, 12);
#ifndef NINFER_VOLTA_BUILD
    failures += run_fp8_case(parent, 17, ops::LinearPolicy::AllowA8, 1);

    const auto run_batched = [&](std::int32_t width, std::int32_t batch,
                                 std::vector<std::int32_t> valid_columns, std::uint32_t seed) {
        const std::vector<float> conv_weight = make_conv_weight(kChannels, seed);
        const bool uses_a8                   = width * batch >= 9;
        const std::size_t workspace_bytes =
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::AllowA8, batch,
                width, width);
        return run_batched_case(
            "FP8 AllowA8 B=" + std::to_string(batch) + " W=" + std::to_string(width), kHidden,
            kValueRows, kZRows, width, batch, std::move(valid_columns), conv_weight,
            workspace_bytes,
            uses_a8 ? kFp8GdnInputProjConvSnapshotA8Tolerance
                    : kFp8GdnInputProjConvSnapshotA16Tolerance,
            [&](std::int32_t row, std::int32_t flat_column, const std::vector<float>& activation) {
                return quantized_weight::dot_fp64(
                    parent.host, row,
                    activation.data() + static_cast<std::size_t>(flat_column) * kHidden, kHidden);
            },
            [&](std::int32_t row, std::int32_t flat_column, const std::vector<float>& activation) {
                return quantized_weight::dot_fp64(
                    parent.host, kChannels + row,
                    activation.data() + static_cast<std::size_t>(flat_column) * kHidden, kHidden);
            },
            [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace) {
                ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, state, valid, initial,
                                                  snapshot_base, q, k, v, z,
                                                  ops::LinearPolicy::AllowA8, workspace, nullptr);
            });
    };
    failures += run_batched(4, 2, {4, 2}, 937U);
    failures += run_batched(16, 8, {16, 13, 11, 7, 5, 3, 2, 1}, 941U);
    failures += parent.verify_preserved("batched FP8 parent weight");
#endif
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
#ifndef NINFER_VOLTA_BUILD
    const std::size_t q4_interval =
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(2048, 2048, 6144, 1, 1, 6);
    const std::size_t q4_witness =
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(2048, 2048, 6144, 1, 4, 4);
    const std::size_t q4_right_endpoint =
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(2048, 2048, 6144, 1, 6, 6);
    if (q4_interval != q4_witness || q4_witness == 0 || q4_right_endpoint != 0) {
        std::cerr << "Q4/Q5 snapshot interval did not retain its non-monotonic T=4 route\n";
        ++failures;
    }
    if (ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(2048, 2048, 4096, 1, 1, 16) !=
            0 ||
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(2048, 2048, 4096, 1, 1, 17) !=
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(2048, 2048, 4096, 1, 17,
                                                                       17)) {
        std::cerr << "W8 snapshot interval did not preserve its zero/nonzero route boundary\n";
        ++failures;
    }
    const std::size_t nvfp4_a4_4 = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        QType::NVFP4, 16384, 5120, ops::LinearPolicy::AllowA4, 1, 4, 4);
    if (ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            QType::NVFP4, 16384, 5120, ops::LinearPolicy::A16Only, 1, 1, 16) != 0 ||
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            QType::NVFP4, 16384, 5120, ops::LinearPolicy::AllowA4, 1, 1, 3) != 0 ||
        nvfp4_a4_4 == 0 ||
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            QType::NVFP4, 16384, 5120, ops::LinearPolicy::AllowA4, 1, 1, 4) != nvfp4_a4_4) {
        std::cerr << "NVFP4 snapshot interval did not preserve its A16/A4 route boundary\n";
        ++failures;
    }
    const auto fp8_snapshot_capacity = [](ops::LinearPolicy policy, std::int32_t batch,
                                          std::int32_t min_width, std::int32_t max_width) {
        return ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, 16384, 5120, policy, batch, min_width, max_width);
    };
    const std::size_t fp8_a16_w4 = fp8_snapshot_capacity(ops::LinearPolicy::A16Only, 1, 4, 4);
    const std::size_t fp8_a16_w6 = fp8_snapshot_capacity(ops::LinearPolicy::A16Only, 1, 6, 6);
    const std::size_t fp8_a8_w10 = fp8_snapshot_capacity(ops::LinearPolicy::AllowA8, 1, 10, 10);
    if (fp8_snapshot_capacity(ops::LinearPolicy::A16Only, 1, 1, 3) != 0 || fp8_a16_w4 == 0 ||
        fp8_a16_w6 <= fp8_a16_w4 ||
        fp8_snapshot_capacity(ops::LinearPolicy::A16Only, 1, 7, 10) != 0 ||
        fp8_snapshot_capacity(ops::LinearPolicy::A16Only, 1, 1, 9) != fp8_a16_w6 ||
        fp8_snapshot_capacity(ops::LinearPolicy::A16Only, 1, 11, 11) == 0 ||
        fp8_snapshot_capacity(ops::LinearPolicy::AllowA8, 1, 7, 9) != 0 || fp8_a8_w10 == 0 ||
        fp8_snapshot_capacity(ops::LinearPolicy::AllowA8, 1, 1, 10) != fp8_a8_w10 ||
        fp8_snapshot_capacity(ops::LinearPolicy::AllowA8, 2, 5, 5) <=
            fp8_snapshot_capacity(ops::LinearPolicy::AllowA8, 2, 4, 4)) {
        std::cerr << "FP8 snapshot capacity did not preserve measured route witnesses\n";
        ++failures;
    }
#endif
    failures += run_q4_q5();
    failures += run_w8();
    failures += run_nvfp4();
    failures += run_fp8();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_input_proj_conv_snapshot\n";
    return failures == 0 ? 0 : 1;
}
