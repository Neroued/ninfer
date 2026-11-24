#include "ninfer/ops/gdn_replay.h"
#include "ops/common/host/operands.h"
#include "ops/linear_attention/gated_delta_net/operands.h"
#include "ops/linear_attention/gated_delta_net/replay/launch.h"
#include <array>
#include <algorithm>

namespace ninfer::ops {
namespace {
using detail::gated_delta_net::kStateDim;
using detail::gated_delta_net::kMaximumReplayRows;

void validate_fold_records(const GdnReplayRecords& records) {
    constexpr const char* kOp       = "gdn_replay_fold";
    const GdnReplayRecordSpec& spec = records.spec;
    (void)detail::gated_delta_net::select_fold_geometry(spec);
    if (spec.record_capacity <= 0 || spec.record_capacity > kMaximumReplayRows || spec.width < 2 ||
        spec.width > 16 || spec.key_dim != kStateDim || spec.value_dim != kStateDim) {
        throw std::invalid_argument(std::string(kOp) + ": unsupported record geometry");
    }
    if (spec.layers > std::numeric_limits<std::int32_t>::max() / spec.record_capacity) {
        throw std::overflow_error(std::string(kOp) + ": record outer extent overflows");
    }
    const std::int32_t outer = spec.layers * spec.record_capacity;
    common::operand::tensor(records.conv, DType::BF16, {spec.conv_channels, spec.width, outer}, 256,
                            kOp, "conv records", records.conv.device);
    common::operand::tensor(records.key, DType::BF16, {kStateDim, spec.qk_heads, spec.width, outer},
                            256, kOp, "key records", records.conv.device);
    common::operand::tensor(records.value, DType::BF16,
                            {kStateDim, spec.value_heads, spec.width, outer}, 256, kOp,
                            "value records", records.conv.device);
    common::operand::tensor(records.gate, DType::FP32, {2, spec.value_heads, spec.width, outer},
                            256, kOp, "gate records", records.conv.device);

    const std::array<common::operand::Range, 4> record_ranges{
        common::operand::range(records.conv, kOp), common::operand::range(records.key, kOp),
        common::operand::range(records.value, kOp), common::operand::range(records.gate, kOp)};
    common::operand::disjoint(record_ranges, "gdn_replay_fold: record planes overlap");
}

void validate_fold_states(const GdnReplayRecords& records,
                          LinearAttentionStateAllLayersView states) {
    constexpr const char* op = "gdn_replay_fold";
    const auto& record       = records.spec;
    const auto& state        = states.spec;
    if (state.layers != static_cast<std::uint32_t>(record.layers) ||
        state.conv_channels != record.conv_channels || state.conv_width != 3 ||
        state.value_heads != record.value_heads || state.value_head_dim != kStateDim ||
        state.key_head_dim != kStateDim || state.conv_dtype != DType::BF16 ||
        states.conv.field.layers != state.layers || states.recurrent.field.layers != state.layers)
        throw std::invalid_argument("gdn_replay_fold: state geometry does not match records");
    common::operand::state_field(states.conv, DType::BF16, {record.conv_channels, 3}, 256, op,
                                 "convolution state", records.conv.device, false);
    common::operand::state_field(states.recurrent, DType::FP32,
                                 {kStateDim, kStateDim, record.value_heads}, 256, op,
                                 "recurrent state", records.conv.device, false);
    if (states.conv.records.storage.data != states.recurrent.records.storage.data ||
        states.conv.records.storage.bytes != states.recurrent.records.storage.bytes)
        throw std::invalid_argument("gdn_replay_fold: state fields require one record arena");
    for (std::uint32_t layer = 0; layer < state.layers; ++layer) {
        const auto conv = common::operand::range(states.conv.tensor(0, layer), op);
        for (std::uint32_t other = 0; other < state.layers; ++other)
            common::operand::disjoint(
                conv, common::operand::range(states.recurrent.tensor(0, other), op), op);
    }
}

common::operand::Blocks selected_field(StateFieldView field, std::int64_t offset) {
    constexpr const char* op = "gdn_replay_fold";
    const auto first         = field.tensor(offset);
    (void)field.tensor(offset, field.field.layers - 1);
    if (!common::operand::aligned(first.data, 256))
        throw std::invalid_argument("gdn_replay_fold: unaligned state record");
    return {common::operand::range(first, op), field.field.layer_stride, field.field.layers};
}

detail::gated_delta_net::GdnReplayFoldKernelRows
validate_fold_rows(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                   std::span<const GdnReplayFoldRow> rows) {
    if (rows.empty() || rows.size() > static_cast<std::size_t>(records.spec.record_capacity))
        throw std::invalid_argument("gdn_replay_fold: active row count is out of range");
    detail::gated_delta_net::GdnReplayFoldKernelRows packed{};
    using Fields = std::array<common::operand::Blocks, 2>;
    std::array<Fields, kMaximumReplayRows> sources{}, destinations{};
    for (std::size_t row = 0; row < rows.size(); ++row) {
        const auto& selected = rows[row];
        if (selected.commit_columns < 0 || selected.commit_columns > records.spec.width)
            throw std::invalid_argument("gdn_replay_fold: commit extent is out of range");
        packed.row[row] = {selected.source_offset, selected.destination_offset,
                           selected.commit_columns, 0};
        if (selected.commit_columns == 0) continue;
        sources[row]      = {selected_field(states.conv, selected.source_offset),
                             selected_field(states.recurrent, selected.source_offset)};
        destinations[row] = {selected_field(states.conv, selected.destination_offset),
                             selected_field(states.recurrent, selected.destination_offset)};
        for (const auto& fields : {sources[row], destinations[row]})
            for (const auto& field : fields)
                for (const auto* record :
                     {&records.conv, &records.key, &records.value, &records.gate})
                    if (common::operand::overlaps(
                            field, common::operand::range(*record, "gdn_replay_fold")))
                        throw std::invalid_argument(
                            "gdn_replay_fold: records overlap state fields");
    }
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (rows[row].commit_columns == 0) continue;
        for (std::size_t other = 0; other < rows.size(); ++other) {
            if (rows[other].commit_columns == 0) continue;
            const bool in_place =
                row == other && rows[row].source_offset == rows[row].destination_offset;
            for (const auto& destination : destinations[row]) {
                if (!in_place)
                    for (const auto& source : sources[other])
                        if (common::operand::overlaps(destination, source))
                            throw std::invalid_argument(
                                "gdn_replay_fold: destination overwrites an active source");
                if (other < row)
                    for (const auto& previous : destinations[other])
                        if (common::operand::overlaps(destination, previous))
                            throw std::invalid_argument(
                                "gdn_replay_fold: active destinations overlap");
            }
        }
    }
    return packed;
}

} // namespace

GdnReplayFoldPlan::GdnReplayFoldPlan(const GdnReplayRecords& records,
                                     LinearAttentionStateAllLayersView states)
    : records_(records), states_(states) {
    validate_fold_records(records_);
    validate_fold_states(records_, states_);
}

void GdnReplayFoldPlan::execute(std::span<const GdnReplayFoldRow> rows,
                                CudaExecutionView execution) const {
    common::operand::device(records_.conv.device, execution.device(), "gdn_replay_fold",
                            "state/records");
    const detail::gated_delta_net::GdnReplayFoldKernelRows packed =
        validate_fold_rows(records_, states_, rows);
    // Row descriptors are host values, including under Graph capture. Preserve full validation
    // above; an all-zero commit has no device work or state/record reads.
    if (std::none_of(rows.begin(), rows.end(),
                     [](const GdnReplayFoldRow& row) { return row.commit_columns != 0; }))
        return;
    detail::gated_delta_net::launch_replay_fold(
        records_, states_, packed, static_cast<std::int32_t>(rows.size()), execution.stream());
}

void GdnReplayFoldPlan::execute(const Tensor& source_offsets, const Tensor& destination_offsets,
                                const Tensor& commit_columns, CudaExecutionView execution) const {
    constexpr const char* op = "gdn_replay_fold";
    common::operand::device(records_.conv.device, execution.device(), op, "state/records");
    const auto batch = source_offsets.ne[0];
    if (batch < 1 || batch > records_.spec.record_capacity)
        throw std::invalid_argument("gdn_replay_fold: active row count is out of range");
    for (const auto* offsets : {&source_offsets, &destination_offsets})
        common::operand::tensor(*offsets, DType::I64, {batch}, 8, op, "state offsets",
                                execution.device());
    common::operand::tensor(commit_columns, DType::I32, {batch}, 4, op, "commit columns",
                            execution.device());
    for (const auto* metadata : {&source_offsets, &destination_offsets, &commit_columns})
        for (const auto* record : {&records_.conv, &records_.key, &records_.value, &records_.gate})
            common::operand::disjoint(common::operand::range(*metadata, op),
                                      common::operand::range(*record, op), op);
    detail::gated_delta_net::launch_replay_fold(
        records_, states_,
        detail::gated_delta_net::GdnReplayFoldDeviceRows{
            static_cast<const std::int64_t*>(source_offsets.data),
            static_cast<const std::int64_t*>(destination_offsets.data),
            static_cast<const std::int32_t*>(commit_columns.data)},
        batch, execution.stream());
}

} // namespace ninfer::ops
