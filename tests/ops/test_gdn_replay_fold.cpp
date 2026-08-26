#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gdn_replay.h"

#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "ops/gdn_ref.h"
#include "ops/input_projection_test_common.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kStateDim       = 128;
constexpr std::int32_t kQkHeads        = 16;
constexpr std::int32_t kRecordCapacity = 8;
constexpr std::size_t kGuardBytes      = 256;

std::uint32_t mix(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

float signed_pattern(std::uint32_t key, float magnitude = 0.06F) {
    const std::int32_t centered = static_cast<std::int32_t>(mix(key) % 2001U) - 1000;
    return static_cast<float>(centered) * (magnitude / 1000.0F);
}

std::uint16_t bf16_pattern(std::uint32_t key, float magnitude = 0.06F) {
    return f32_to_bf16(signed_pattern(key, magnitude));
}

void* offset_pointer(void* pointer, std::size_t bytes) {
    return static_cast<void*>(static_cast<std::byte*>(pointer) + bytes);
}

const void* offset_pointer(const void* pointer, std::size_t bytes) {
    return static_cast<const void*>(static_cast<const std::byte*>(pointer) + bytes);
}

struct FoldProfile {
    std::int32_t layers;
    std::int32_t value_heads;
    std::int32_t conv_channels;
};

constexpr ReductionCriterion recurrent_state_criterion() {
    return {/*relative_l2=*/2.7e-3, /*gross_absolute=*/1.0e-5,
            /*gross_relative_to_max_reference=*/3.9e-3};
}

int verify_fold_oracle(const FoldProfile profile, std::int32_t width, std::int32_t commit,
                       std::uint32_t seed, const std::vector<std::uint16_t>& key_records,
                       const std::vector<std::uint16_t>& value_records,
                       const std::vector<std::uint32_t>& gate_records,
                       const std::vector<float>& actual) {
    gdn_ref::Inputs input;
    input.head_dim    = kStateDim;
    input.qk_heads    = kQkHeads;
    input.value_heads = profile.value_heads;
    input.tokens      = commit;
    input.q.assign(static_cast<std::size_t>(kStateDim) * kQkHeads * commit, 0.0F);
    input.k.resize(input.q.size());
    input.v.resize(static_cast<std::size_t>(kStateDim) * profile.value_heads * commit);
    input.g.resize(static_cast<std::size_t>(profile.value_heads) * commit);
    input.beta.resize(input.g.size());
    input.state.assign(actual.size(), signed_pattern(seed + 500009U, 0.01F));

    for (std::int32_t token = 0; token < commit; ++token) {
        const std::int64_t column = token;
        for (std::int32_t head = 0; head < kQkHeads; ++head) {
            const std::size_t base =
                static_cast<std::size_t>((column * kQkHeads + head) * kStateDim);
            for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                input.k[base + dim] = bf16_to_f32(key_records[base + dim]);
            }
        }
        for (std::int32_t head = 0; head < profile.value_heads; ++head) {
            const std::size_t value_base =
                static_cast<std::size_t>((column * profile.value_heads + head) * kStateDim);
            for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                input.v[value_base + dim] = bf16_to_f32(value_records[value_base + dim]);
            }
            const std::size_t gate_base =
                static_cast<std::size_t>((column * profile.value_heads + head) * 2);
            const std::size_t destination =
                static_cast<std::size_t>(token) * profile.value_heads + head;
            input.g[destination]    = std::bit_cast<float>(gate_records[gate_base]);
            input.beta[destination] = std::bit_cast<float>(gate_records[gate_base + 1]);
        }
    }

    const gdn_ref::Result expected =
        gdn_ref::evaluate(input, 1.0 / std::sqrt(static_cast<double>(kStateDim)), true);
    const std::vector<double> actual_double(actual.begin(), actual.end());
    return verify_reduction("gdn replay fold independent recurrent-state oracle", actual_double,
                            expected.final_state, recurrent_state_criterion());
}

std::vector<std::int32_t> selected_slots(std::int32_t rows) {
    if (rows == 1) { return {2}; }
    std::vector<std::int32_t> slots{10, 2, 8, 0, 6, 4, 9, 1};
    slots.resize(static_cast<std::size_t>(rows));
    return slots;
}

int run_case(const FoldProfile profile, std::int32_t width, std::int32_t rows,
             const std::vector<std::int32_t>& commits, std::uint32_t seed,
             bool distinct_destination = false) {
    const std::vector<std::int32_t> source_slots = selected_slots(rows);
    if (distinct_destination && rows != 1) {
        throw std::logic_error("distinct replay destination case requires one row");
    }
    std::vector<std::int32_t> destination_slots = source_slots;
    if (distinct_destination) { destination_slots[0] = 1; }
    const std::int32_t slot_count = rows == 1 ? 3 : 11;
    const std::int32_t outer      = profile.layers * kRecordCapacity;
    const std::size_t recurrent_slot_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * profile.value_heads;
    const std::size_t recurrent_slot_bytes = recurrent_slot_elements * sizeof(float);
    const std::size_t conv_slot_elements   = static_cast<std::size_t>(profile.conv_channels) * 3;
    const std::size_t conv_slot_bytes      = conv_slot_elements * sizeof(std::uint16_t);

    const GdnReplayRecordSpec record_spec{
        .layers          = profile.layers,
        .record_capacity = kRecordCapacity,
        .width           = width,
        .conv_channels   = profile.conv_channels,
        .qk_heads        = kQkHeads,
        .value_heads     = profile.value_heads,
        .key_dim         = kStateDim,
        .value_dim       = kStateDim,
    };
    LayoutBuilder record_builder;
    const GdnReplayRecordLayout record_layout =
        plan_gdn_replay_records(record_builder, record_spec);
    const std::size_t record_bytes = record_builder.finish(256);
    DeviceBuffer record_storage(record_bytes + 2 * kGuardBytes);
    record_storage.fill(0xa5);
    void* record_base = offset_pointer(record_storage.p, kGuardBytes);
    cuda_check(cudaMemset(record_base, 0xff, record_bytes), "initialize replay records");
    const GdnReplayRecords records({record_base, record_bytes}, record_layout);

    std::vector<std::uint16_t> conv_records(records.conv.numel(), 0xffffU);
    std::vector<std::uint16_t> key_records(records.key.numel(), 0xffffU);
    std::vector<std::uint16_t> value_records(records.value.numel(), 0xffffU);
    std::vector<std::uint32_t> gate_records(records.gate.numel(), 0xffffffffU);
    for (std::int32_t layer = 0; layer < profile.layers; ++layer) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::int32_t commit = commits[static_cast<std::size_t>(row)];
            const std::int64_t record_outer =
                static_cast<std::int64_t>(layer) * kRecordCapacity + row;
            for (std::int32_t token = 0; token < commit; ++token) {
                const std::int64_t column = record_outer * width + token;
                for (std::int32_t channel = 0; channel < profile.conv_channels; ++channel) {
                    conv_records[static_cast<std::size_t>(column) * profile.conv_channels +
                                 channel] =
                        bf16_pattern(seed + layer * 131U + row * 17U + token * 7U + channel);
                }
                for (std::int32_t head = 0; head < kQkHeads; ++head) {
                    const std::size_t base =
                        static_cast<std::size_t>((column * kQkHeads + head) * kStateDim);
                    for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                        key_records[base + dim] =
                            bf16_pattern(seed + 100003U + layer * 197U + row * 23U + token * 11U +
                                             head * 5U + dim,
                                         0.08F);
                    }
                }
                for (std::int32_t head = 0; head < profile.value_heads; ++head) {
                    const std::size_t vector_base =
                        static_cast<std::size_t>((column * profile.value_heads + head) * kStateDim);
                    for (std::int32_t dim = 0; dim < kStateDim; ++dim) {
                        value_records[vector_base + dim] =
                            bf16_pattern(seed + 200003U + layer * 211U + row * 29U + token * 13U +
                                             head * 7U + dim,
                                         0.08F);
                    }
                    const std::size_t gate_base =
                        static_cast<std::size_t>((column * profile.value_heads + head) * 2);
                    const float g =
                        -0.03F -
                        static_cast<float>(mix(seed + layer * 31U + row * 17U + token * 7U + head) %
                                           900U) /
                            1000.0F;
                    const float beta =
                        0.05F + static_cast<float>(mix(seed + 300007U + layer * 37U + row * 19U +
                                                       token * 11U + head) %
                                                   900U) /
                                    1000.0F;
                    gate_records[gate_base]     = std::bit_cast<std::uint32_t>(g);
                    gate_records[gate_base + 1] = std::bit_cast<std::uint32_t>(beta);
                }
            }
        }
    }
    cuda_check(cudaMemcpy(records.conv.data, conv_records.data(), records.conv.bytes(),
                          cudaMemcpyHostToDevice),
               "upload conv records");
    cuda_check(cudaMemcpy(records.key.data, key_records.data(), records.key.bytes(),
                          cudaMemcpyHostToDevice),
               "upload key records");
    cuda_check(cudaMemcpy(records.value.data, value_records.data(), records.value.bytes(),
                          cudaMemcpyHostToDevice),
               "upload value records");
    cuda_check(cudaMemcpy(records.gate.data, gate_records.data(), records.gate.bytes(),
                          cudaMemcpyHostToDevice),
               "upload gate records");

    LayoutBuilder state_builder;
    const LinearAttentionStatePoolLayout state_layout = plan_linear_attention_state_pool(
        state_builder, {.layers         = static_cast<std::uint32_t>(profile.layers),
                        .conv_channels  = profile.conv_channels,
                        .conv_width     = 3,
                        .value_heads    = profile.value_heads,
                        .value_head_dim = kStateDim,
                        .key_head_dim   = kStateDim,
                        .slot_count     = slot_count,
                        .conv_dtype     = DType::BF16});
    const std::size_t state_bytes = state_builder.finish(256);
    DeviceBuffer state_storage(state_bytes + 2 * kGuardBytes);
    state_storage.fill(0xa5);
    void* state_base = offset_pointer(state_storage.p, kGuardBytes);
    cuda_check(cudaMemset(state_base, 0, state_bytes), "initialize all-layer state");
    LinearAttentionStatePool state_pool({state_base, state_bytes}, state_layout);

    DeviceBuffer expected_conv(static_cast<std::size_t>(profile.layers) * rows * conv_slot_bytes);
    expected_conv.fill(0);
    for (std::int32_t layer = 0; layer < profile.layers; ++layer) {
        for (std::int32_t row = 0; row < rows; ++row) {
            std::vector<std::uint16_t> initial(conv_slot_elements);
            for (std::int32_t history = 0; history < 3; ++history) {
                for (std::int32_t channel = 0; channel < profile.conv_channels; ++channel) {
                    initial[static_cast<std::size_t>(history) * profile.conv_channels + channel] =
                        bf16_pattern(seed + 400009U + layer * 223U + row * 41U + history * 13U +
                                         channel,
                                     0.05F);
                }
            }
            const Tensor destination = state_pool.conv_slot(
                static_cast<std::uint32_t>(layer), source_slots[static_cast<std::size_t>(row)]);
            cuda_check(cudaMemcpy(destination.data, initial.data(), conv_slot_bytes,
                                  cudaMemcpyHostToDevice),
                       "upload initial conv state");

            std::vector<std::uint16_t> expected = initial;
            const std::int32_t commit           = commits[static_cast<std::size_t>(row)];
            if (commit > 0) {
                const std::int64_t record_outer =
                    static_cast<std::int64_t>(layer) * kRecordCapacity + row;
                for (std::int32_t channel = 0; channel < profile.conv_channels; ++channel) {
                    const auto record_value = [&](std::int32_t token) {
                        return conv_records[static_cast<std::size_t>(
                                                (record_outer * width + token) *
                                                profile.conv_channels) +
                                            channel];
                    };
                    if (commit == 1) {
                        expected[channel] = initial[profile.conv_channels + channel];
                        expected[profile.conv_channels + channel] =
                            initial[2LL * profile.conv_channels + channel];
                        expected[2LL * profile.conv_channels + channel] = record_value(0);
                    } else if (commit == 2) {
                        expected[channel] = initial[2LL * profile.conv_channels + channel];
                        expected[profile.conv_channels + channel]       = record_value(0);
                        expected[2LL * profile.conv_channels + channel] = record_value(1);
                    } else {
                        expected[channel]                               = record_value(commit - 3);
                        expected[profile.conv_channels + channel]       = record_value(commit - 2);
                        expected[2LL * profile.conv_channels + channel] = record_value(commit - 1);
                    }
                }
            }
            void* expected_destination = offset_pointer(
                expected_conv.p, static_cast<std::size_t>(layer * rows + row) * conv_slot_bytes);
            cuda_check(cudaMemcpy(expected_destination, expected.data(), conv_slot_bytes,
                                  cudaMemcpyHostToDevice),
                       "upload expected conv state");
        }
    }

    const std::size_t expected_recurrent_bytes =
        static_cast<std::size_t>(profile.layers) * rows * recurrent_slot_bytes;
    DeviceBuffer expected_recurrent(expected_recurrent_bytes);
    expected_recurrent.fill(0);
    DeviceBuffer local_state(recurrent_slot_bytes);
    local_state.fill(0);
    const std::size_t q_elements = static_cast<std::size_t>(kStateDim) * kQkHeads * width;
    const std::size_t out_elements =
        static_cast<std::size_t>(kStateDim) * profile.value_heads * width;
    DeviceBuffer q(q_elements * sizeof(std::uint16_t));
    DeviceBuffer out(out_elements * sizeof(std::uint16_t));
    q.fill(0);
    DeviceBuffer g_row(static_cast<std::size_t>(profile.value_heads) * width * sizeof(float));
    DeviceBuffer beta_row(static_cast<std::size_t>(profile.value_heads) * width * sizeof(float));
    const Tensor q_tensor(q.p, DType::BF16, {kStateDim, kQkHeads, width, 1});
    Tensor local_state_tensor(local_state.p, DType::FP32,
                              {kStateDim, kStateDim, profile.value_heads});
    Tensor output(out.p, DType::BF16, {kStateDim, profile.value_heads, width, 1});
    constexpr float kScale = 1.0F / std::sqrt(128.0F);
    WorkspaceArena reference_workspace(256);

    for (std::int32_t layer = 0; layer < profile.layers; ++layer) {
        const GdnReplayRecordLayer layer_records = records.layer(layer, rows);
        for (std::int32_t row = 0; row < rows; ++row) {
            const float initial_value =
                signed_pattern(seed + 500009U + layer * 227U + row * 43U, 0.01F);
            const std::vector<float> initial_recurrent(recurrent_slot_elements, initial_value);
            const Tensor actual_initial = state_pool.recurrent_slot(
                static_cast<std::uint32_t>(layer), source_slots[static_cast<std::size_t>(row)]);
            cuda_check(cudaMemcpy(actual_initial.data, initial_recurrent.data(),
                                  recurrent_slot_bytes, cudaMemcpyHostToDevice),
                       "upload initial recurrent state");
            cuda_check(cudaMemcpy(local_state.p, initial_recurrent.data(), recurrent_slot_bytes,
                                  cudaMemcpyHostToDevice),
                       "upload local reference initial state");

            const std::int32_t commit = commits[static_cast<std::size_t>(row)];
            void* expected =
                offset_pointer(expected_recurrent.p,
                               static_cast<std::size_t>(layer * rows + row) * recurrent_slot_bytes);
            if (commit == 0) {
                cuda_check(cudaMemcpy(expected, initial_recurrent.data(), recurrent_slot_bytes,
                                      cudaMemcpyHostToDevice),
                           "save unchanged recurrent state");
                continue;
            }
            std::vector<float> g_host(static_cast<std::size_t>(profile.value_heads) * width);
            std::vector<float> beta_host(static_cast<std::size_t>(profile.value_heads) * width);
            const std::int64_t record_outer =
                static_cast<std::int64_t>(layer) * kRecordCapacity + row;
            for (std::int32_t token = 0; token < width; ++token) {
                for (std::int32_t head = 0; head < profile.value_heads; ++head) {
                    const std::size_t source = static_cast<std::size_t>(
                        ((record_outer * width + token) * profile.value_heads + head) * 2);
                    const std::size_t destination =
                        static_cast<std::size_t>(token) * profile.value_heads + head;
                    g_host[destination]    = std::bit_cast<float>(gate_records[source]);
                    beta_host[destination] = std::bit_cast<float>(gate_records[source + 1]);
                }
            }
            g_row.copy_from_host(g_host.data(), g_row.bytes);
            beta_row.copy_from_host(beta_host.data(), beta_row.bytes);
            Tensor query = q_tensor.slice(2, 0, commit).view({kStateDim, kQkHeads, commit});
            Tensor key   = layer_records.key.slice(3, row, 1)
                             .slice(2, 0, commit)
                             .view({kStateDim, kQkHeads, commit});
            Tensor value = layer_records.value.slice(3, row, 1)
                               .slice(2, 0, commit)
                               .view({kStateDim, profile.value_heads, commit});
            Tensor g_tensor(g_row.p, DType::FP32, {profile.value_heads, width, 1});
            Tensor beta_tensor(beta_row.p, DType::FP32, {profile.value_heads, width, 1});
            Tensor g_prefix = g_tensor.slice(1, 0, commit).view({profile.value_heads, commit});
            Tensor beta_prefix =
                beta_tensor.slice(1, 0, commit).view({profile.value_heads, commit});
            Tensor output_prefix =
                output.slice(2, 0, commit).view({kStateDim, profile.value_heads, commit});
            ops::gated_delta_net(query, key, value, g_prefix, beta_prefix, kScale, true,
                                 reference_workspace, local_state_tensor, output_prefix, nullptr);
            cuda_check(cudaMemcpyAsync(expected, local_state.p, recurrent_slot_bytes,
                                       cudaMemcpyDeviceToDevice, nullptr),
                       "save reference recurrent state");
        }
    }

    const std::vector<std::uint8_t> records_before =
        from_device<std::uint8_t>(record_storage, record_storage.bytes);
    std::vector<ops::GdnReplayFoldRow> fold_rows(static_cast<std::size_t>(rows));
    for (std::int32_t row = 0; row < rows; ++row) {
        fold_rows[static_cast<std::size_t>(row)] = {
            source_slots[static_cast<std::size_t>(row)],
            destination_slots[static_cast<std::size_t>(row)],
            commits[static_cast<std::size_t>(row)]};
    }
    const ops::GdnReplayFoldPlan fold_plan(records, state_pool.all_layers_view());
    fold_plan.execute(fold_rows, nullptr);
    cuda_synchronize();

    int failures             = 0;
    const std::string suffix = " L=" + std::to_string(profile.layers) +
                               " Hv=" + std::to_string(profile.value_heads) +
                               " T=" + std::to_string(width) + " B=" + std::to_string(rows);
    std::vector<float> actual_recurrent(recurrent_slot_elements);
    std::vector<float> expected_recurrent_host(recurrent_slot_elements);
    std::vector<std::uint16_t> actual_conv(conv_slot_elements);
    std::vector<std::uint16_t> expected_conv_host(conv_slot_elements);
    for (std::int32_t layer = 0; layer < profile.layers; ++layer) {
        for (std::int32_t row = 0; row < rows; ++row) {
            const Tensor actual_state =
                state_pool.recurrent_slot(static_cast<std::uint32_t>(layer),
                                          destination_slots[static_cast<std::size_t>(row)]);
            cuda_check(cudaMemcpy(actual_recurrent.data(), actual_state.data, recurrent_slot_bytes,
                                  cudaMemcpyDeviceToHost),
                       "download folded recurrent state");
            const void* expected_state =
                offset_pointer(expected_recurrent.p,
                               static_cast<std::size_t>(layer * rows + row) * recurrent_slot_bytes);
            cuda_check(cudaMemcpy(expected_recurrent_host.data(), expected_state,
                                  recurrent_slot_bytes, cudaMemcpyDeviceToHost),
                       "download expected recurrent state");
            if (actual_recurrent != expected_recurrent_host) {
                std::cerr << "fold recurrent state differs from direct prefix" << suffix
                          << " layer=" << layer << " row=" << row << "\n";
                return failures + 1;
            }
            if (layer == 0 && row == 0 && profile.layers == 30 && width == 2 && rows == 1 &&
                commits[0] == 2) {
                failures += verify_fold_oracle(profile, width, commits[0], seed, key_records,
                                               value_records, gate_records, actual_recurrent);
            }

            const Tensor actual_history =
                state_pool.conv_slot(static_cast<std::uint32_t>(layer),
                                     destination_slots[static_cast<std::size_t>(row)]);
            cuda_check(cudaMemcpy(actual_conv.data(), actual_history.data, conv_slot_bytes,
                                  cudaMemcpyDeviceToHost),
                       "download folded conv state");
            const void* expected_history = offset_pointer(
                expected_conv.p, static_cast<std::size_t>(layer * rows + row) * conv_slot_bytes);
            cuda_check(cudaMemcpy(expected_conv_host.data(), expected_history, conv_slot_bytes,
                                  cudaMemcpyDeviceToHost),
                       "download expected conv state");
            if (actual_conv != expected_conv_host) {
                std::cerr << "fold conv history differs" << suffix << " layer=" << layer
                          << " row=" << row << "\n";
                return failures + 1;
            }
            if (distinct_destination) {
                const float initial_value =
                    signed_pattern(seed + 500009U + layer * 227U + row * 43U, 0.01F);
                const Tensor source_state = state_pool.recurrent_slot(
                    static_cast<std::uint32_t>(layer), source_slots[static_cast<std::size_t>(row)]);
                const std::vector<float> source_recurrent =
                    from_device<float>(source_state.data, recurrent_slot_elements);
                if (!std::all_of(source_recurrent.begin(), source_recurrent.end(),
                                 [&](float value) { return value == initial_value; })) {
                    std::cerr << "fold modified recurrent source" << suffix << " layer=" << layer
                              << " row=" << row << "\n";
                    return failures + 1;
                }
                std::vector<std::uint16_t> initial_conv(conv_slot_elements);
                for (std::int32_t history = 0; history < 3; ++history) {
                    for (std::int32_t channel = 0; channel < profile.conv_channels; ++channel) {
                        initial_conv[static_cast<std::size_t>(history) * profile.conv_channels +
                                     channel] =
                            bf16_pattern(seed + 400009U + layer * 223U + row * 41U + history * 13U +
                                             channel,
                                         0.05F);
                    }
                }
                const Tensor source_history = state_pool.conv_slot(
                    static_cast<std::uint32_t>(layer), source_slots[static_cast<std::size_t>(row)]);
                if (from_device<std::uint16_t>(source_history.data, conv_slot_elements) !=
                    initial_conv) {
                    std::cerr << "fold modified convolution source" << suffix << " layer=" << layer
                              << " row=" << row << "\n";
                    return failures + 1;
                }
            }
        }
    }

    std::vector<float> inactive_recurrent(recurrent_slot_elements);
    std::vector<std::uint16_t> inactive_conv(conv_slot_elements);
    for (std::int32_t layer = 0; layer < profile.layers; ++layer) {
        for (std::int32_t slot = 0; slot < slot_count; ++slot) {
            if (std::find(source_slots.begin(), source_slots.end(), slot) != source_slots.end() ||
                std::find(destination_slots.begin(), destination_slots.end(), slot) !=
                    destination_slots.end()) {
                continue;
            }
            const Tensor recurrent =
                state_pool.recurrent_slot(static_cast<std::uint32_t>(layer), slot);
            cuda_check(cudaMemcpy(inactive_recurrent.data(), recurrent.data, recurrent_slot_bytes,
                                  cudaMemcpyDeviceToHost),
                       "download inactive recurrent state");
            if (!std::all_of(inactive_recurrent.begin(), inactive_recurrent.end(),
                             [](float value) { return value == 0.0F; })) {
                std::cerr << "fold modified inactive recurrent slot" << suffix << " layer=" << layer
                          << " slot=" << slot << "\n";
                return failures + 1;
            }
            const Tensor conv = state_pool.conv_slot(static_cast<std::uint32_t>(layer), slot);
            cuda_check(cudaMemcpy(inactive_conv.data(), conv.data, conv_slot_bytes,
                                  cudaMemcpyDeviceToHost),
                       "download inactive conv state");
            if (!std::all_of(inactive_conv.begin(), inactive_conv.end(),
                             [](std::uint16_t value) { return value == 0; })) {
                std::cerr << "fold modified inactive conv slot" << suffix << " layer=" << layer
                          << " slot=" << slot << "\n";
                return failures + 1;
            }
        }
    }

    const std::vector<std::uint8_t> records_after =
        from_device<std::uint8_t>(record_storage, record_storage.bytes);
    if (records_after != records_before) {
        std::cerr << "fold modified record storage" << suffix << "\n";
        ++failures;
    }
    const std::vector<std::uint8_t> state_storage_after =
        from_device<std::uint8_t>(state_storage, state_storage.bytes);
    if (!std::all_of(state_storage_after.begin(),
                     state_storage_after.begin() + static_cast<std::ptrdiff_t>(kGuardBytes),
                     [](std::uint8_t byte) { return byte == 0xa5; }) ||
        !std::all_of(state_storage_after.end() - static_cast<std::ptrdiff_t>(kGuardBytes),
                     state_storage_after.end(), [](std::uint8_t byte) { return byte == 0xa5; })) {
        std::cerr << "fold modified state outer guard" << suffix << "\n";
        ++failures;
    }
    return failures;
}

int run_record_fold_rounds() {
    using ninfer::test::input_projection::DevicePackedWeight;
    using ninfer::test::input_projection::make_bf16_activation;

    constexpr FoldProfile kProfile{30, 32, 8192};
    constexpr std::int32_t kHidden       = 2048;
    constexpr std::int32_t kValueRows    = 4096;
    constexpr std::int32_t kZRows        = 4096;
    constexpr std::int32_t kParentRows   = 12288;
    constexpr std::int32_t kWidth        = 2;
    constexpr std::int32_t kStateSlots   = 3;
    constexpr std::int32_t kInitialSlot  = 2;
    constexpr std::int32_t kSnapshotBase = 0;
    constexpr float kScale               = 1.0F / std::sqrt(128.0F);

    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::W8G32_F16S, kParentRows, kHidden, 1901U));
    const std::vector<float> activation = make_bf16_activation(kHidden, kWidth, 1903U);
    DeviceBuffer device_x               = to_device_bf16(activation);
    std::vector<std::uint16_t> conv_weight_bits(static_cast<std::size_t>(kProfile.conv_channels) *
                                                4);
    for (std::size_t index = 0; index < conv_weight_bits.size(); ++index) {
        conv_weight_bits[index] = bf16_pattern(1907U + static_cast<std::uint32_t>(index), 0.02F);
    }
    DeviceBuffer device_conv_weight = to_device(conv_weight_bits);
    DeviceBuffer device_initial     = to_device(std::vector<std::int32_t>{kInitialSlot});
    DeviceBuffer device_snapshot    = to_device(std::vector<std::int32_t>{kSnapshotBase});

    LayoutBuilder record_builder;
    const GdnReplayRecordLayout record_layout =
        plan_gdn_replay_records(record_builder, {.layers          = kProfile.layers,
                                                 .record_capacity = 1,
                                                 .width           = kWidth,
                                                 .conv_channels   = kProfile.conv_channels,
                                                 .qk_heads        = kQkHeads,
                                                 .value_heads     = kProfile.value_heads,
                                                 .key_dim         = kStateDim,
                                                 .value_dim       = kStateDim});
    DeviceBuffer record_storage(record_builder.finish(256));
    record_storage.fill(0xff);
    GdnReplayRecords records({record_storage.p, record_storage.bytes}, record_layout);

    LayoutBuilder state_builder;
    const LinearAttentionStatePoolLayout state_layout = plan_linear_attention_state_pool(
        state_builder, {.layers         = static_cast<std::uint32_t>(kProfile.layers),
                        .conv_channels  = kProfile.conv_channels,
                        .conv_width     = 3,
                        .value_heads    = kProfile.value_heads,
                        .value_head_dim = kStateDim,
                        .key_head_dim   = kStateDim,
                        .slot_count     = kStateSlots,
                        .conv_dtype     = DType::BF16});
    DeviceBuffer state_storage(state_builder.finish(256));
    state_storage.fill(0);
    LinearAttentionStatePool state_pool({state_storage.p, state_storage.bytes}, state_layout);
    const ops::GdnReplayFoldPlan fold_plan(records, state_pool.all_layers_view());

    const std::size_t recurrent_slot_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * kProfile.value_heads;
    const std::size_t conv_slot_elements = static_cast<std::size_t>(kProfile.conv_channels) * 3;
    for (std::int32_t layer = 0; layer < kProfile.layers; ++layer) {
        const float initial_value = signed_pattern(1911U + layer * 47U, 0.01F);
        const std::vector<float> recurrent(recurrent_slot_elements, initial_value);
        const Tensor recurrent_slot =
            state_pool.recurrent_slot(static_cast<std::uint32_t>(layer), kInitialSlot);
        cuda_check(cudaMemcpy(recurrent_slot.data, recurrent.data(), recurrent_slot.bytes(),
                              cudaMemcpyHostToDevice),
                   "upload pair initial recurrent state");

        std::vector<std::uint16_t> conv(conv_slot_elements);
        for (std::int32_t history = 0; history < 3; ++history) {
            for (std::int32_t channel = 0; channel < kProfile.conv_channels; ++channel) {
                conv[static_cast<std::size_t>(history) * kProfile.conv_channels + channel] =
                    bf16_pattern(1913U + layer * 53U + history * 11U + channel, 0.04F);
            }
        }
        const Tensor conv_slot =
            state_pool.conv_slot(static_cast<std::uint32_t>(layer), kInitialSlot);
        cuda_check(
            cudaMemcpy(conv_slot.data, conv.data(), conv_slot.bytes(), cudaMemcpyHostToDevice),
            "upload pair initial conv state");
    }

    DeviceBuffer snapshot_q(static_cast<std::size_t>(2048) * kWidth * sizeof(std::uint16_t));
    DeviceBuffer snapshot_k(static_cast<std::size_t>(2048) * kWidth * sizeof(std::uint16_t));
    DeviceBuffer snapshot_v(static_cast<std::size_t>(kValueRows) * kWidth * sizeof(std::uint16_t));
    DeviceBuffer snapshot_z(static_cast<std::size_t>(kZRows) * kWidth * sizeof(std::uint16_t));
    DeviceBuffer record_q(snapshot_q.bytes);
    DeviceBuffer record_k(snapshot_k.bytes);
    DeviceBuffer record_v(snapshot_v.bytes);
    DeviceBuffer record_z(snapshot_z.bytes);
    DeviceBuffer snapshot_out(snapshot_v.bytes);
    DeviceBuffer record_out(snapshot_v.bytes);
    DeviceBuffer direct_discard_state(recurrent_slot_elements * sizeof(float));
    DeviceBuffer device_g(static_cast<std::size_t>(kProfile.value_heads) * kWidth * sizeof(float));
    DeviceBuffer device_beta(device_g.bytes);

    Tensor x(device_x.p, DType::BF16, {kHidden, kWidth, 1});
    Tensor conv_weight(device_conv_weight.p, DType::BF16, {kProfile.conv_channels, 4});
    Tensor initial_selector(device_initial.p, DType::I32, {1});
    Tensor snapshot_selector(device_snapshot.p, DType::I32, {1});
    Tensor valid;
    Tensor snapshot_query(snapshot_q.p, DType::BF16, {2048, kWidth, 1});
    Tensor snapshot_key(snapshot_k.p, DType::BF16, {2048, kWidth, 1});
    Tensor snapshot_value(snapshot_v.p, DType::BF16, {kValueRows, kWidth, 1});
    Tensor snapshot_z_tensor(snapshot_z.p, DType::BF16, {kZRows, kWidth, 1});
    Tensor record_query(record_q.p, DType::BF16, {2048, kWidth, 1});
    Tensor record_key(record_k.p, DType::BF16, {2048, kWidth, 1});
    Tensor record_value(record_v.p, DType::BF16, {kValueRows, kWidth, 1});
    Tensor record_z_tensor(record_z.p, DType::BF16, {kZRows, kWidth, 1});
    Tensor snapshot_output(snapshot_out.p, DType::BF16,
                           {kStateDim, kProfile.value_heads, kWidth, 1});
    Tensor record_output(record_out.p, DType::BF16, {kStateDim, kProfile.value_heads, kWidth, 1});
    Tensor g(device_g.p, DType::FP32, {kProfile.value_heads, kWidth, 1});
    Tensor beta(device_beta.p, DType::FP32, {kProfile.value_heads, kWidth, 1});

    const std::size_t snapshot_workspace_bytes =
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(2048, 2048, kValueRows, 1,
                                                                   kWidth, kWidth);
    const std::size_t record_workspace_bytes =
        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(2048, 2048, kValueRows, 1, kWidth,
                                                                 kWidth);
    WorkspaceArena snapshot_workspace(std::max<std::size_t>(256, snapshot_workspace_bytes));
    WorkspaceArena record_workspace(std::max<std::size_t>(256, record_workspace_bytes));
    WorkspaceArena recurrent_workspace(256);

    int failures = 0;
    for (std::int32_t round = 0; round < 2; ++round) {
        const std::int32_t commit = round + 1;
        std::vector<float> g_host(static_cast<std::size_t>(kProfile.value_heads) * kWidth);
        std::vector<float> beta_host(g_host.size());
        for (std::size_t index = 0; index < g_host.size(); ++index) {
            g_host[index]    = -0.04F - static_cast<float>((index + round * 17) % 80) / 100.0F;
            beta_host[index] = 0.08F + static_cast<float>((index * 7 + round * 13) % 80) / 100.0F;
        }
        device_g.copy_from_host(g_host.data(), device_g.bytes);
        device_beta.copy_from_host(beta_host.data(), device_beta.bytes);

        for (std::int32_t layer = 0; layer < kProfile.layers; ++layer) {
            GdnReplayRecordLayer layer_records = records.layer(layer, 1);
            Tensor conv_states = state_pool.layer_view(static_cast<std::uint32_t>(layer)).conv;
            ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv_weight, conv_states, valid,
                                              initial_selector, snapshot_selector, snapshot_query,
                                              snapshot_key, snapshot_value, snapshot_z_tensor,
                                              snapshot_workspace, nullptr);
            ops::gdn_input_proj_conv_record(x, parent.view(), conv_weight, conv_states, valid,
                                            initial_selector, layer_records.conv, record_query,
                                            record_key, record_value, record_z_tensor,
                                            record_workspace, nullptr);

            Tensor snapshot_q_view = snapshot_query.view({kStateDim, kQkHeads, kWidth, 1});
            Tensor snapshot_k_view = snapshot_key.view({kStateDim, kQkHeads, kWidth, 1});
            Tensor snapshot_v_view =
                snapshot_value.view({kStateDim, kProfile.value_heads, kWidth, 1});
            Tensor record_q_view = record_query.view({kStateDim, kQkHeads, kWidth, 1});
            Tensor record_k_view = record_key.view({kStateDim, kQkHeads, kWidth, 1});
            Tensor record_v_view = record_value.view({kStateDim, kProfile.value_heads, kWidth, 1});
            Tensor recurrent_states =
                state_pool.layer_view(static_cast<std::uint32_t>(layer)).recurrent;
            Tensor q_direct    = snapshot_q_view.view({kStateDim, kQkHeads, kWidth});
            Tensor k_direct    = snapshot_k_view.view({kStateDim, kQkHeads, kWidth});
            Tensor v_direct    = snapshot_v_view.view({kStateDim, kProfile.value_heads, kWidth});
            Tensor g_direct    = g.view({kProfile.value_heads, kWidth});
            Tensor beta_direct = beta.view({kProfile.value_heads, kWidth});
            Tensor out_direct  = snapshot_output.view({kStateDim, kProfile.value_heads, kWidth});
            Tensor initial_state =
                state_pool.recurrent_slot(static_cast<std::uint32_t>(layer), kInitialSlot);
            Tensor committed_state =
                state_pool.recurrent_slot(static_cast<std::uint32_t>(layer), commit - 1);
            Tensor discard_state(direct_discard_state.p, DType::FP32,
                                 {kStateDim, kStateDim, kProfile.value_heads});
            Tensor& full_state = commit == kWidth ? committed_state : discard_state;
            ops::gated_delta_net(q_direct, k_direct, v_direct, g_direct, beta_direct, kScale, true,
                                 recurrent_workspace, initial_state, full_state, out_direct,
                                 nullptr);
            if (commit < kWidth) {
                Tensor q_prefix    = q_direct.slice(2, 0, commit);
                Tensor k_prefix    = k_direct.slice(2, 0, commit);
                Tensor v_prefix    = v_direct.slice(2, 0, commit);
                Tensor g_prefix    = g_direct.slice(1, 0, commit);
                Tensor beta_prefix = beta_direct.slice(1, 0, commit);
                Tensor out_prefix  = out_direct.slice(2, 0, commit);
                ops::gated_delta_net(q_prefix, k_prefix, v_prefix, g_prefix, beta_prefix, kScale,
                                     true, recurrent_workspace, initial_state, committed_state,
                                     out_prefix, nullptr);
            }
            ops::gated_delta_net_replay_record(record_q_view, record_k_view, record_v_view, g, beta,
                                               kScale, recurrent_states, valid, initial_selector,
                                               layer_records.key, layer_records.value,
                                               layer_records.gate, record_output, nullptr);
            cuda_synchronize();

            const std::string label = "record-fold pair round=" + std::to_string(round) +
                                      " layer=" + std::to_string(layer);
            const auto compare_bf16 = [&](const DeviceBuffer& lhs, const DeviceBuffer& rhs,
                                          std::size_t elements, const char* field) {
                if (from_device<std::uint16_t>(lhs, elements) !=
                    from_device<std::uint16_t>(rhs, elements)) {
                    std::cerr << label << " " << field << " differs\n";
                    return 1;
                }
                return 0;
            };
            failures += compare_bf16(snapshot_q, record_q, static_cast<std::size_t>(2048) * kWidth,
                                     "query");
            failures +=
                compare_bf16(snapshot_k, record_k, static_cast<std::size_t>(2048) * kWidth, "key");
            failures += compare_bf16(snapshot_v, record_v,
                                     static_cast<std::size_t>(kValueRows) * kWidth, "value");
            failures +=
                compare_bf16(snapshot_z, record_z, static_cast<std::size_t>(kZRows) * kWidth, "z");
            failures +=
                compare_bf16(snapshot_out, record_out,
                             static_cast<std::size_t>(kValueRows) * kWidth, "recurrent output");
            if (failures != 0) { return failures; }
        }

        const std::array fold_rows{ops::GdnReplayFoldRow{kInitialSlot, kInitialSlot, commit}};
        fold_plan.execute(fold_rows, nullptr);
        cuda_synchronize();
        for (std::int32_t layer = 0; layer < kProfile.layers; ++layer) {
            const Tensor folded_recurrent =
                state_pool.recurrent_slot(static_cast<std::uint32_t>(layer), kInitialSlot);
            const Tensor snapshot_recurrent =
                state_pool.recurrent_slot(static_cast<std::uint32_t>(layer), commit - 1);
            if (from_device<float>(folded_recurrent.data, recurrent_slot_elements) !=
                from_device<float>(snapshot_recurrent.data, recurrent_slot_elements)) {
                std::cerr << "record-fold recurrent mismatch round=" << round << " layer=" << layer
                          << "\n";
                return failures + 1;
            }
            const Tensor folded_conv =
                state_pool.conv_slot(static_cast<std::uint32_t>(layer), kInitialSlot);
            const Tensor snapshot_conv =
                state_pool.conv_slot(static_cast<std::uint32_t>(layer), commit - 1);
            if (from_device<std::uint16_t>(folded_conv.data, conv_slot_elements) !=
                from_device<std::uint16_t>(snapshot_conv.data, conv_slot_elements)) {
                std::cerr << "record-fold conv mismatch round=" << round << " layer=" << layer
                          << "\n";
                return failures + 1;
            }
        }
    }
    failures += parent.verify_preserved("record-fold W8 parent weight");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case({48, 48, 10240}, 2, 1, {2}, 1801U, true);
    failures += run_case({48, 48, 10240}, 3, 4, {0, 1, 2, 3}, 1811U);
    failures += run_case({48, 48, 10240}, 6, 8, {0, 1, 2, 3, 6, 4, 1, 5}, 1821U);
    failures += run_case({30, 32, 8192}, 2, 1, {2}, 1831U);
    failures += run_case({30, 32, 8192}, 6, 1, {6}, 1841U);
    failures += run_case({30, 32, 8192}, 6, 2, {2, 5}, 1851U);
    failures += run_case({30, 32, 8192}, 16, 8, {0, 1, 2, 3, 16, 7, 12, 5}, 1861U);
    failures += run_record_fold_rounds();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_replay_fold\n";
    return failures == 0 ? 0 : 1;
}
