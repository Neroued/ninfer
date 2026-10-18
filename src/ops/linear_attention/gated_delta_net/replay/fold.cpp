#include "ninfer/ops/gdn_replay.h"
#include "ops/linear_attention/gated_delta_net/replay/validation.h"
#include "ops/linear_attention/gated_delta_net/replay/launch.h"
#include <array>
#include <algorithm>

namespace ninfer::ops {
namespace {
using namespace detail::gated_delta_net::replay;

void validate_fold_records(const GdnReplayRecords& records) {
    constexpr const char* kOp       = "gdn_replay_fold";
    const GdnReplayRecordSpec& spec = records.spec;
    (void)detail::gated_delta_net::select_fold_geometry(spec);
    if (spec.record_capacity <= 0 || spec.record_capacity > kMaximumRows || spec.width < 2 ||
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
    constexpr const char* kOp = "gdn_replay_fold";
    const auto& record        = records.spec;
    const auto& state         = states.spec;
    if (state.layers != static_cast<std::uint32_t>(record.layers) ||
        state.conv_channels != record.conv_channels || state.conv_width != 3 ||
        state.value_heads != record.value_heads || state.value_head_dim != kStateDim ||
        state.key_head_dim != kStateDim || states.conv_layer0.ne[2] <= 0 ||
        state.conv_dtype != DType::BF16) {
        throw std::invalid_argument(std::string(kOp) + ": state geometry does not match records");
    }
    common::operand::slotted_tensor(states.conv_layer0, DType::BF16,
                                    {record.conv_channels, 3, states.conv_layer0.ne[2]}, 2, 256,
                                    kOp, "conv state layer 0", records.conv.device);
    common::operand::slotted_tensor(
        states.recurrent_layer0, DType::FP32,
        {kStateDim, kStateDim, record.value_heads, states.conv_layer0.ne[2]}, 3, 256, kOp,
        "recurrent state layer 0", records.conv.device);
    if (states.conv_layer_stride_bytes <= 0 || states.recurrent_layer_stride_bytes <= 0 ||
        states.conv_layer_stride_bytes % static_cast<std::int64_t>(sizeof(std::uint16_t)) != 0 ||
        states.recurrent_layer_stride_bytes % static_cast<std::int64_t>(sizeof(float)) != 0 ||
        static_cast<std::uint64_t>(states.conv_layer_stride_bytes) <
            states.conv_layer0.slice(2, 0, 1).bytes() ||
        static_cast<std::uint64_t>(states.recurrent_layer_stride_bytes) <
            states.recurrent_layer0.slice(3, 0, 1).bytes()) {
        throw std::invalid_argument(std::string(kOp) + ": invalid state layer stride");
    }

    for (std::int32_t layer = 0; layer < record.layers; ++layer) {
        const auto conv =
            layer_blocks(states.conv_layer0, states.conv_layer_stride_bytes, layer, kOp);
        const auto recurrent =
            layer_blocks(states.recurrent_layer0, states.recurrent_layer_stride_bytes, layer, kOp);
        for (std::int32_t other = 0; other < record.layers; ++other) {
            const auto other_conv =
                layer_blocks(states.conv_layer0, states.conv_layer_stride_bytes, other, kOp);
            const auto other_recurrent = layer_blocks(
                states.recurrent_layer0, states.recurrent_layer_stride_bytes, other, kOp);
            if (common::operand::overlaps(conv, other_recurrent) ||
                (other < layer && (common::operand::overlaps(conv, other_conv) ||
                                   common::operand::overlaps(recurrent, other_recurrent)))) {
                throw std::invalid_argument("gdn_replay_fold: state layers overlap");
            }
        }
    }
}

void require_records_disjoint_from_states(const GdnReplayRecords& records,
                                          LinearAttentionStateAllLayersView states) {
    const std::array<common::operand::Range, 4> record_ranges{
        common::operand::range(records.conv, "gdn_replay_fold conv records"),
        common::operand::range(records.key, "gdn_replay_fold key records"),
        common::operand::range(records.value, "gdn_replay_fold value records"),
        common::operand::range(records.gate, "gdn_replay_fold gate records")};
    for (const common::operand::Range record : record_ranges) {
        for (std::int32_t layer = 0; layer < records.spec.layers; ++layer) {
            const auto conv = layer_blocks(states.conv_layer0, states.conv_layer_stride_bytes,
                                           layer, "gdn_replay_fold conv state");
            const auto recurrent =
                layer_blocks(states.recurrent_layer0, states.recurrent_layer_stride_bytes, layer,
                             "gdn_replay_fold recurrent state");
            if (common::operand::overlaps(conv, record) ||
                common::operand::overlaps(recurrent, record)) {
                throw std::invalid_argument("gdn_replay_fold: records overlap state storage");
            }
        }
    }
}

detail::gated_delta_net::GdnReplayFoldKernelRows
validate_fold_rows(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                   std::span<const GdnReplayFoldRow> rows) {
    if (rows.empty() || rows.size() > static_cast<std::size_t>(records.spec.record_capacity)) {
        throw std::invalid_argument("gdn_replay_fold: active row count is out of range");
    }
    detail::gated_delta_net::GdnReplayFoldKernelRows packed{};
    for (std::size_t row = 0; row < rows.size(); ++row) {
        if (rows[row].source_state_slot < 0 ||
            rows[row].source_state_slot >= states.conv_layer0.ne[2] ||
            rows[row].destination_state_slot < 0 ||
            rows[row].destination_state_slot >= states.conv_layer0.ne[2]) {
            throw std::invalid_argument("gdn_replay_fold: linear state slot is out of range");
        }
        if (rows[row].commit_columns < 0 || rows[row].commit_columns > records.spec.width) {
            throw std::invalid_argument("gdn_replay_fold: commit extent is out of range");
        }
        for (std::size_t previous = 0; previous < row; ++previous) {
            if (rows[previous].destination_state_slot == rows[row].destination_state_slot ||
                rows[previous].destination_state_slot == rows[row].source_state_slot ||
                rows[row].destination_state_slot == rows[previous].source_state_slot) {
                throw std::invalid_argument(
                    "gdn_replay_fold: active state source/destination bindings overlap");
            }
        }
        packed.row[row] = {rows[row].source_state_slot, rows[row].destination_state_slot,
                           rows[row].commit_columns, 0};
    }
    return packed;
}

} // namespace

GdnReplayFoldPlan::GdnReplayFoldPlan(const GdnReplayRecords& records,
                                     LinearAttentionStateAllLayersView states)
    : records_(records), states_(states) {
    validate_fold_records(records_);
    validate_fold_states(records_, states_);
    require_records_disjoint_from_states(records_, states_);
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

} // namespace ninfer::ops
